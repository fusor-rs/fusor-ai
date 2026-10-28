//! Causal language models: a decoder stack (`transformer`) and the RMSNorm after its last layer
//! (`Decoder`); with the input embeddings and a head over the vocabulary (`Lm`), the reading of
//! rows of embeddings through the stack over a KV cache, in chunks (`Reader`), and the prediction
//! from the last row, for any stack: its final-normed state, the head's logits and the greedy
//! next token (`Prediction`).
//!
//! Tensors under a prefix `p`: the stack's (`p.layers.{i}…`), the final norm `p.norm`, the head
//! `p.head` and, unless the input embeddings are the head's rows (`Embedding::Tied`), their table
//! `p.embed`.

pub mod sampling;

use bundle::{Load, Scope, Store};
use kernels::ops::rms_norm;
use kernels::{Linear, NARROW_COLUMN_ALIGN, QuantizedRows, Table, Vector};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::rotary::Position;
use crate::transformer::{self, Buffers, KvCache, Mask, Positioned, Span, Transformer};

/// A decoder stack and the RMSNorm after its last layer.
pub struct Decoder {
    pub body: Transformer,
    norm: Vector,
    eps: f32,
}

impl Decoder {
    /// The final norm of hidden row `row` (after `forward`, outside or inside a team run).
    pub fn final_norm(&self, buffers: &Buffers, row: usize, out: &mut [f32]) {
        let width = self.norm.values().len();
        let x = &buffers.hidden.read()[row * width..(row + 1) * width];
        rms_norm(x, self.norm.values(), self.eps, out);
    }
}

/// The stack under `p` and its final norm `p.norm`.
impl Load for Decoder {
    type Shape = transformer::Shape;

    fn load(store: &mut Store, prefix: &str, shape: Self::Shape) -> Result<Self, bundle::Error> {
        let body = Transformer::load(store, prefix, shape)?;
        Ok(Self {
            body,
            norm: Scope::new(store, prefix).load("norm", shape.hidden)?,
            eps: shape.norm_eps,
        })
    }
}

/// Where a language model's input embeddings come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Embedding {
    /// Their own table, `p.embed`.
    Own,
    /// The head's rows.
    Tied,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub stack: transformer::Shape,
    pub vocabulary: usize,
    pub embedding: Embedding,
}

pub struct Lm {
    /// `None` when the embeddings are the head's rows.
    embed: Option<Table>,
    pub decoder: Decoder,
    head: Linear,
    hidden: usize,
}

impl Load for Lm {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            stack, vocabulary, ..
        } = shape;
        let mut scope = Scope::new(store, prefix);
        let embed = match shape.embedding {
            Embedding::Own => Some(scope.load("embed", [vocabulary, stack.hidden])?),
            Embedding::Tied => None,
        };
        let head = scope.load("head", [vocabulary, stack.hidden])?;
        Ok(Self {
            embed,
            decoder: Decoder::load(store, prefix, stack)?,
            head,
            hidden: stack.hidden,
        })
    }
}

impl Lm {
    /// The embeddings of `ids`, `ids.len() × hidden`, into `out`.
    pub fn embed(&self, ids: &[u32], out: &mut [f32]) {
        let table = match &self.embed {
            Some(embed) => embed.matrix(),
            None => self.head.matrix(),
        };
        for (&id, row) in ids.iter().zip(out.chunks_exact_mut(self.hidden)) {
            table.dequantize_row(id as usize, row);
        }
    }
}

/// The positions of rows read: their sequence positions, or positions of their own (multimodal
/// rotary embeddings), one per row.
#[derive(Clone, Copy)]
pub enum Positions<'rows> {
    Sequential,
    Given(&'rows [Position]),
}

/// Rows to read: their embeddings, `rows × hidden`, from sequence position `first`.
pub struct Rows<'rows> {
    pub embeddings: &'rows [f32],
    pub first: usize,
    pub positions: Positions<'rows>,
}

/// The prediction from a row of a stack's output: its final-normed state and the head's logits
/// over the vocabulary.
pub struct Prediction {
    /// The final-normed hidden state of the row.
    pub last: SharedBuf,
    pub logits: SharedBuf,
    normed: PerThread<Vec<f32>>,
    quant: PerThread<QuantizedRows>,
}

impl Prediction {
    pub fn new(hidden: usize, vocabulary: usize, threads: usize) -> Self {
        Self {
            last: SharedBuf::new(hidden),
            logits: SharedBuf::new(vocabulary),
            normed: PerThread::new(threads, || vec![0.0; hidden]),
            quant: PerThread::new(threads, QuantizedRows::default),
        }
    }

    /// Inside a team run: `row`'s RMSNorm by `norm` into `last`, and this thread's columns of
    /// `head`'s logits over it.
    pub fn predict(&self, ctx: &Ctx, row: &[f32], norm: (&Vector, f32), head: &Linear) {
        let (weight, eps) = norm;
        let quant = self.quant.get(ctx);
        let normed = self.normed.get(ctx);
        rms_norm(row, weight.values(), eps, normed);
        if ctx.is_first() {
            self.last.part(0..normed.len()).copy_from_slice(normed);
        }
        let matrix = head.matrix();
        let input = quant.input(&matrix, normed);
        let vocabulary = matrix.rows;
        let cols = ctx.split(
            vocabulary.next_multiple_of(NARROW_COLUMN_ALIGN),
            NARROW_COLUMN_ALIGN,
        );
        let cols = cols.start.min(vocabulary)..cols.end.min(vocabulary);
        head.project(&input, cols, |_, col, value| {
            self.logits.set(col, value);
        });
    }

    /// The most likely next token (the first of equals, as `argmax`).
    pub fn greedy(&self) -> u32 {
        let logits = self.logits.read();
        let best = (0..logits.len()).fold(0, |best, token| {
            if logits[token] > logits[best] {
                token
            } else {
                best
            }
        });
        best as u32
    }
}

/// The state of one reading: the KV cache, the activations, and the prediction from the last row
/// read.
pub struct Reader {
    cache: KvCache,
    buffers: Buffers,
    /// Rows per call of the stack.
    chunk: usize,
    pub prediction: Prediction,
}

impl Reader {
    /// A reader of sequences of up to `capacity` rows, `chunk` rows per call of the stack.
    pub fn new(shape: &Shape, capacity: usize, chunk: usize, threads: usize) -> Self {
        Self {
            cache: KvCache::new(shape.stack, capacity),
            buffers: shape.stack.buffers(chunk, threads),
            chunk,
            prediction: Prediction::new(shape.stack.hidden, shape.vocabulary, threads),
        }
    }

    /// Reads `rows` after the positions already cached; `last` and `logits` are those of the
    /// last of them.
    pub fn read(&mut self, team: &Team, lm: &Lm, rows: &Rows) {
        let count = rows.embeddings.len() / lm.hidden;
        for start in (0..count).step_by(self.chunk) {
            let end = (start + self.chunk).min(count);
            let embeddings = &rows.embeddings[start * lm.hidden..end * lm.hidden];
            self.buffers.hidden.get_mut()[..embeddings.len()].copy_from_slice(embeddings);
            let span = Span {
                rows: end - start,
                first: rows.first + start,
                mask: Mask::Causal,
            };
            let positions = match rows.positions {
                Positions::Sequential => Positions::Sequential,
                Positions::Given(positions) => Positions::Given(&positions[start..end]),
            };
            let last = end == count;
            team.run(|ctx| {
                self.forward(ctx, lm, span, positions);
                if last {
                    self.predict(ctx, lm, end - start - 1);
                }
            });
        }
    }

    fn forward(&self, ctx: &Ctx, lm: &Lm, span: Span, positions: Positions) {
        let body = &lm.decoder.body;
        match positions {
            Positions::Sequential => body.forward(ctx, &self.cache, &self.buffers, span),
            Positions::Given(positions) => {
                let at = Positioned { span, positions };
                body.forward_at(ctx, &self.cache, &self.buffers, at);
            }
        }
    }

    /// The prediction from row `row` of the last call.
    fn predict(&self, ctx: &Ctx, lm: &Lm, row: usize) {
        let decoder = &lm.decoder;
        let hidden = &self.buffers.hidden.read()[row * lm.hidden..(row + 1) * lm.hidden];
        let norm = (&decoder.norm, decoder.eps);
        self.prediction.predict(ctx, hidden, norm, &lm.head);
    }
}

#[cfg(test)]
mod tests {
    use kernels::{DType, Linear, Table, Tensor, Vector};
    use team::Team;

    use super::Prediction;

    /// Head row `i` is all `i`, the normed row all ones: logit `i` is `4i`, the last of 9 too.
    #[test]
    fn predicts_every_logit_of_a_vocabulary_of_no_whole_split_units() {
        let (hidden, vocabulary) = (4, 9);
        let rows: Vec<f32> = (0..vocabulary).flat_map(|row| [row as f32; 4]).collect();
        let weight = Tensor::encode(DType::F32, vec![vocabulary, hidden], &rows).unwrap();
        let head = Linear::new(Table::new(weight).unwrap(), None).unwrap();
        let ones = Tensor::encode(DType::F32, vec![hidden], &[1.0; 4]).unwrap();
        let norm = Vector::new(ones).unwrap();
        let team = Team::global();
        let prediction = Prediction::new(hidden, vocabulary, team.threads());
        team.run(|ctx| prediction.predict(ctx, &[1.0; 4], (&norm, 0.0), &head));
        let want: Vec<f32> = (0..vocabulary).map(|row| 4.0 * row as f32).collect();
        assert_eq!(prediction.logits.read(), want);
    }
}
