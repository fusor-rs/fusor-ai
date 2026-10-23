//! The predictor reading one sequence of tokens and their stamps: each token's embedding plus
//! its stamp's through the stack over the sequence's KV cache, the final norm of every row, the
//! dependency layer's keys and values of each, and the `s1` logits of the last row; then, for a
//! drawn `s1`, the dependency layer's output and the `s2` logits.
//!
//! Every call reads the rows of the window after those the stream holds: the stack attends
//! causally, so once a window slides past `MAX_CONTEXT` rows its every row changes and it is
//! read again from row 0 (`Stream::restart`), its rotary positions restarting there as the
//! reference's.

use blocks::transformer::{self, KvCache, Mask, Span};
use kernels::ops::{attend, rms_norm};
use kernels::{QuantizedRows, SharedRows, project, project_input};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::model::{
    DEPENDENCY_HEADS, HIDDEN, MAX_CONTEXT, Model, NORM_EPS, STACK, Stamp, VOCABULARY,
};
use crate::tokenizer::Token;

const DEPENDENCY_HEAD: usize = HIDDEN / DEPENDENCY_HEADS;

/// One sequence's state: the stack's KV cache, and the dependency layer's keys and values of
/// the window's rows read so far, `rows × HIDDEN` each.
pub struct Stream {
    cache: KvCache,
    keys: SharedBuf,
    values: SharedBuf,
    rows: usize,
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            cache: KvCache::new(STACK, MAX_CONTEXT),
            keys: SharedBuf::new(MAX_CONTEXT * HIDDEN),
            values: SharedBuf::new(MAX_CONTEXT * HIDDEN),
            rows: 0,
        }
    }
}

impl Stream {
    /// Forgets the rows read: the next read starts at row 0.
    pub fn restart(&mut self) {
        self.rows = 0;
    }
}

/// The activations every stream's calls share.
pub struct Buffers {
    stack: transformer::Buffers,
    /// Each row's scaled `s1` and `s2` embeddings, `rows × 2·HIDDEN`, and its stamp's.
    pairs: SharedBuf,
    times: SharedBuf,
    /// The final-normed rows of the last call, `rows × HIDDEN`, and its last row.
    normed: SharedBuf,
    last: SharedBuf,
    /// The dependency layer's query, attention and residual sum.
    query: SharedBuf,
    attention: SharedBuf,
    mixed: SharedBuf,
    /// The last call's logits: `s1` after `read`, `s2` after `dependent`.
    pub logits: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    quant: QuantizedRows,
    scores: Vec<f32>,
    row: Vec<f32>,
}

impl Buffers {
    pub fn new(threads: usize) -> Self {
        Self {
            stack: STACK.buffers(MAX_CONTEXT, threads),
            pairs: SharedBuf::new(MAX_CONTEXT * 2 * HIDDEN),
            times: SharedBuf::new(MAX_CONTEXT * HIDDEN),
            normed: SharedBuf::new(MAX_CONTEXT * HIDDEN),
            last: SharedBuf::new(HIDDEN),
            query: SharedBuf::new(HIDDEN),
            attention: SharedBuf::new(HIDDEN),
            mixed: SharedBuf::new(HIDDEN),
            logits: SharedBuf::new(VOCABULARY),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

/// A window of the sequence: its tokens and their stamps.
#[derive(Clone, Copy)]
pub struct Window<'seq> {
    pub tokens: &'seq [Token],
    pub stamps: &'seq [Stamp],
}

impl Model {
    /// Reads the rows of `window` after those `stream` holds: the `s1` logits of the last row
    /// into `buffers.logits`.
    pub fn read(&self, team: &Team, buffers: &mut Buffers, stream: &mut Stream, window: Window) {
        let first = stream.rows;
        let rows = window.tokens.len() - first;
        self.embed_inputs(buffers, &window, first);
        let span = Span {
            rows,
            first,
            mask: Mask::Causal,
        };
        let (buffers, state) = (&*buffers, &*stream);
        team.run(|ctx| {
            self.stack_rows(ctx, buffers, state, span);
            self.keep_rows(ctx, buffers, state, span);
        });
        stream.rows = window.tokens.len();
    }

    /// Each row's scaled half embeddings into `pairs` and its stamp's embedding into `times`.
    fn embed_inputs(&self, buffers: &mut Buffers, window: &Window, first: usize) {
        let scale = (HIDDEN as f64).sqrt() as f32;
        let pairs = buffers.pairs.get_mut().as_chunks_mut::<{ 2 * HIDDEN }>().0;
        let times = buffers.times.get_mut().as_chunks_mut::<HIDDEN>().0;
        let rows = window.tokens[first..].iter().zip(&window.stamps[first..]);
        for ((token, stamp), (pair, time)) in rows.zip(pairs.iter_mut().zip(times)) {
            let (coarse, fine) = pair.split_at_mut(HIDDEN);
            let embedding = &self.embedding;
            embedding
                .s1
                .matrix()
                .dequantize_row(usize::from(token.s1), coarse);
            embedding
                .s2
                .matrix()
                .dequantize_row(usize::from(token.s2), fine);
            pair.iter_mut().for_each(|value| *value *= scale);
            self.time.embed(stamp, time);
        }
    }

    /// The fused embeddings plus the stamps' through the stack, then each row's final norm into
    /// `normed`.
    fn stack_rows(&self, ctx: &Ctx, buffers: &Buffers, stream: &Stream, span: Span) {
        let hidden = &buffers.stack.hidden;
        let times = buffers.times.read();
        let pairs = SharedRows {
            values: &buffers.pairs,
            rows: span.rows,
        };
        let quant = &mut buffers.scratch.get(ctx).quant;
        project(ctx, &self.embedding.fusion, pairs, quant, |index, value| {
            hidden.set(index, value + times[index]);
        });
        ctx.barrier();
        self.stack
            .body
            .forward(ctx, &stream.cache, &buffers.stack, span);
        ctx.barrier();
        for row in ctx.split(span.rows, 1) {
            let normed = buffers.normed.part(row * HIDDEN..(row + 1) * HIDDEN);
            self.stack.final_norm(&buffers.stack, row, normed);
        }
        ctx.barrier();
    }

    /// The dependency layer's keys and values of the normed rows into the stream, at their
    /// window rows, and the `s1` logits of the last.
    fn keep_rows(&self, ctx: &Ctx, buffers: &Buffers, stream: &Stream, span: Span) {
        let offset = span.first * HIDDEN;
        let normed = SharedRows {
            values: &buffers.normed,
            rows: span.rows,
        };
        let quant = &mut buffers.scratch.get(ctx).quant;
        let dependency = &self.dependency;
        for (linear, kept) in [
            (&dependency.key, &stream.keys),
            (&dependency.value, &stream.values),
        ] {
            project(ctx, linear, normed, quant, |index, value| {
                kept.set(offset + index, value);
            });
        }
        let last = &buffers.normed.read()[(span.rows - 1) * HIDDEN..][..HIDDEN];
        if ctx.is_first() {
            buffers.last.part(0..HIDDEN).copy_from_slice(last);
        }
        let input = quant.input(&self.head_s1.matrix(), last);
        project_input(ctx, &self.head_s1, &input, |index, value| {
            buffers.logits.set(index, value);
        });
    }

    /// The `s2` logits for `s1` into `buffers.logits`, after `read` of the stream's last rows.
    pub fn dependent(&self, team: &Team, buffers: &Buffers, stream: &Stream, s1: u16) {
        let mut sibling = [0.0; HIDDEN];
        let table = self.embedding.s1.matrix();
        table.dequantize_row(usize::from(s1), &mut sibling);
        team.run(|ctx| {
            let dependency = &self.dependency;
            let scratch = buffers.scratch.get(ctx);
            let input = scratch.quant.input(&dependency.query.matrix(), &sibling);
            project_input(ctx, &dependency.query, &input, |index, value| {
                buffers.query.set(index, value);
            });
            ctx.barrier();
            attend_heads(ctx, buffers, stream, &mut scratch.scores);
            ctx.barrier();
            self.mix(ctx, buffers, &mut scratch.quant);
            ctx.barrier();
            let row = &mut scratch.row;
            row.resize(HIDDEN, 0.0);
            rms_norm(
                buffers.mixed.read(),
                dependency.norm.values(),
                NORM_EPS,
                row,
            );
            let input = scratch.quant.input(&self.head_s2.matrix(), row);
            project_input(ctx, &self.head_s2, &input, |index, value| {
                buffers.logits.set(index, value);
            });
        });
    }

    /// The last row read plus the dependency layer's output, this thread's columns.
    fn mix(&self, ctx: &Ctx, buffers: &Buffers, quant: &mut QuantizedRows) {
        let last = buffers.last.read();
        let out = &self.dependency.out;
        let input = quant.input(&out.matrix(), buffers.attention.read());
        project_input(ctx, out, &input, |index, value| {
            buffers.mixed.set(index, last[index] + value);
        });
    }
}

/// This thread's heads of the dependency layer's query over the stream's keys and values.
fn attend_heads(ctx: &Ctx, buffers: &Buffers, stream: &Stream, scores: &mut Vec<f32>) {
    let query = buffers.query.read();
    let (keys, values) = (stream.keys.read(), stream.values.read());
    for head in ctx.split(DEPENDENCY_HEADS, 1) {
        let at = head * DEPENDENCY_HEAD;
        let span = at..at + DEPENDENCY_HEAD;
        attend(
            &query[span.clone()],
            stream.rows,
            |row| {
                let slot = row * HIDDEN + at;
                let head = slot..slot + DEPENDENCY_HEAD;
                (&keys[head.clone()], &values[head])
            },
            scores,
            buffers.attention.part(span),
        );
    }
}
