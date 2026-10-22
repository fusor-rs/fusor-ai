//! Causal language models: a decoder stack (`transformer`) and the RMSNorm after its last layer
//! (`Decoder`); with the input embeddings and a head over the vocabulary (`Lm`), the reading of
//! rows of embeddings through the stack over a KV cache, in chunks, and the prediction from the
//! last row: its final-normed state, the head's logits and the greedy next token (`Reader`).
//!
//! Tensors under a prefix `p`: the stack's (`p.layers.{i}…`), the final norm `p.norm`, the head
//! `p.head` and, unless the input embeddings are the head's rows (`Embedding::Tied`), their table
//! `p.embed`.

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
    vocabulary: usize,
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
            vocabulary,
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

/// The state of one reading: the KV cache, the activations, the last row's final-normed state
/// and logits.
pub struct Reader {
    cache: KvCache,
    buffers: Buffers,
    /// Rows per call of the stack.
    chunk: usize,
    /// The final-normed hidden state of the last row read.
    pub last: SharedBuf,
    pub logits: SharedBuf,
    normed: PerThread<Vec<f32>>,
    quant: PerThread<QuantizedRows>,
}

impl Reader {
    /// A reader of sequences of up to `capacity` rows, `chunk` rows per call of the stack.
    pub fn new(shape: &Shape, capacity: usize, chunk: usize, threads: usize) -> Self {
        let hidden = shape.stack.hidden;
        Self {
            cache: KvCache::new(shape.stack, capacity),
            buffers: shape.stack.buffers(chunk, threads),
            chunk,
            last: SharedBuf::new(hidden),
            logits: SharedBuf::new(shape.vocabulary),
            normed: PerThread::new(threads, || vec![0.0; hidden]),
            quant: PerThread::new(threads, QuantizedRows::default),
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

    /// The final norm of row `row` into `last`, and this thread's columns of its logits.
    fn predict(&self, ctx: &Ctx, lm: &Lm, row: usize) {
        let quant = self.quant.get(ctx);
        let normed = self.normed.get(ctx);
        lm.decoder.final_norm(&self.buffers, row, normed);
        if ctx.is_first() {
            self.last.part(0..lm.hidden).copy_from_slice(normed);
        }
        let input = quant.input(&lm.head.matrix(), normed);
        let cols = ctx.split(lm.vocabulary, NARROW_COLUMN_ALIGN);
        lm.head.project(&input, cols, |_, col, value| {
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
