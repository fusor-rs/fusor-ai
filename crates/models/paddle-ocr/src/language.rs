//! The language model reading a prompt and writing the answer: rows of token embeddings (the
//! image's rows the projector's tokens) through the ERNIE stack with their multimodal rotary
//! positions, then the final norm and the head's logits of the last row.

use blocks::rotary::Position;
use blocks::transformer::{self, KvCache, Mask, Positioned, Span};
use kernels::ops::rms_norm;
use kernels::{NARROW_COLUMN_ALIGN, QuantizedRows};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::model::{HIDDEN, LANGUAGE_SHAPE, Language, VOCABULARY};

/// Prompt rows per call of the stack.
const PREFILL_ROWS: usize = 128;

/// The state of one reading: the KV cache, the activations, the last row's logits.
pub(crate) struct Reader {
    cache: KvCache,
    buffers: transformer::Buffers,
    /// The final-normed hidden state of the last row read.
    pub(crate) last: SharedBuf,
    pub(crate) logits: SharedBuf,
    quant: PerThread<QuantizedRows>,
}

/// Rows to read: their embeddings, `rows × HIDDEN`, from sequence position `first`.
pub(crate) struct Rows<'rows> {
    pub(crate) embeddings: &'rows [f32],
    pub(crate) positions: &'rows [Position],
    pub(crate) first: usize,
}

impl Reader {
    /// A reader of sequences of up to `capacity` rows.
    pub(crate) fn new(capacity: usize, threads: usize) -> Self {
        Self {
            cache: KvCache::new(LANGUAGE_SHAPE, capacity),
            buffers: LANGUAGE_SHAPE.buffers(PREFILL_ROWS, threads),
            last: SharedBuf::new(HIDDEN),
            logits: SharedBuf::new(VOCABULARY),
            quant: PerThread::new(threads, QuantizedRows::default),
        }
    }

    /// Reads `rows` after the positions already cached; `last` and `logits` are those of the
    /// last of them.
    pub(crate) fn read(&mut self, team: &Team, language: &Language, rows: &Rows) {
        let count = rows.positions.len();
        for start in (0..count).step_by(PREFILL_ROWS) {
            let end = (start + PREFILL_ROWS).min(count);
            let embeddings = &rows.embeddings[start * HIDDEN..end * HIDDEN];
            self.buffers.hidden.get_mut()[..embeddings.len()].copy_from_slice(embeddings);
            let at = Positioned {
                span: Span {
                    rows: end - start,
                    first: rows.first + start,
                    mask: Mask::Causal,
                },
                positions: &rows.positions[start..end],
            };
            let last = end == count;
            team.run(|ctx| {
                language
                    .body
                    .forward_at(ctx, &self.cache, &self.buffers, at);
                if last {
                    self.predict(ctx, language, end - start - 1);
                }
            });
        }
    }

    /// The final norm of row `row` into `last`, and this thread's columns of its logits.
    fn predict(&self, ctx: &Ctx, language: &Language, row: usize) {
        let hidden = &self.buffers.hidden.read()[row * HIDDEN..(row + 1) * HIDDEN];
        let quant = self.quant.get(ctx);
        let mut normed = [0.0; HIDDEN];
        rms_norm(
            hidden,
            language.norm.values(),
            LANGUAGE_SHAPE.norm_eps,
            &mut normed,
        );
        if ctx.is_first() {
            self.last.part(0..HIDDEN).copy_from_slice(&normed);
        }
        let input = quant.input(&language.head.matrix(), &normed);
        let cols = ctx.split(VOCABULARY, NARROW_COLUMN_ALIGN);
        language.head.project(&input, cols, |_, col, value| {
            self.logits.set(col, value);
        });
    }

    /// The most likely next token (the first of equals, as `argmax`).
    pub(crate) fn greedy(&self) -> u32 {
        let logits = self.logits.read();
        let best = (0..VOCABULARY).fold(0, |best, token| {
            if logits[token] > logits[best] {
                token
            } else {
                best
            }
        });
        best as u32
    }
}

impl Language {
    /// The embeddings of `ids`, `ids.len() × HIDDEN`, into `out`.
    pub(crate) fn embed(&self, ids: &[u32], out: &mut [f32]) {
        let table = self.embed.matrix();
        for (&id, row) in ids.iter().zip(out.as_chunks_mut::<HIDDEN>().0) {
            table.dequantize_row(id as usize, row);
        }
    }
}
