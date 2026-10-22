//! The language model reading a prompt and writing the answer: rows of token embeddings (the
//! image's rows its visual tokens) through the Qwen3 stack at consecutive positions, then the
//! final norm and the head's logits of the last row.

use blocks::transformer::{self, KvCache, Mask, Span};
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

    /// Reads the rows of `embeddings` (`rows × HIDDEN`) at positions from `first`, after the
    /// positions already cached; `last` and `logits` are those of the last of them.
    pub(crate) fn read(
        &mut self,
        team: &Team,
        language: &Language,
        embeddings: &[f32],
        first: usize,
    ) {
        let count = embeddings.len() / HIDDEN;
        for start in (0..count).step_by(PREFILL_ROWS) {
            let end = (start + PREFILL_ROWS).min(count);
            let rows = &embeddings[start * HIDDEN..end * HIDDEN];
            self.buffers.hidden.get_mut()[..rows.len()].copy_from_slice(rows);
            let span = Span {
                rows: end - start,
                first: first + start,
                mask: Mask::Causal,
            };
            let last = end == count;
            team.run(|ctx| {
                language.body.forward(ctx, &self.cache, &self.buffers, span);
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
    /// The embeddings of `ids`, `ids.len() × HIDDEN`, into `out`: the head's rows.
    pub(crate) fn embed(&self, ids: &[u32], out: &mut [f32]) {
        let table = self.head.matrix();
        for (&id, row) in ids.iter().zip(out.as_chunks_mut::<HIDDEN>().0) {
            table.dequantize_row(id as usize, row);
        }
    }
}
