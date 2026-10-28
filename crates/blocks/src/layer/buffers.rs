//! The activations of one forward call, shared by the team, and each thread's scratch.

use kernels::{BLOCK, COLUMN_ALIGN, Linear, QuantizedRows};
use team::{Ctx, PerThread, SharedBuf};

use super::Heads;
use crate::attention;

/// Activations of one forward call of up to `rows` positions.
pub struct Buffers {
    pub(crate) rows: usize,
    pub(crate) hidden_width: usize,
    /// The residual stream, `rows × hidden`: the input of `forward` and its output.
    pub hidden: SharedBuf,
    /// The token mixer's input projection (a fused QKV), `rows × mixer`.
    pub(crate) qkv: SharedBuf,
    pub(crate) attention: SharedBuf,
    /// The feed-forward's inner activations.
    pub(super) inner: SharedBuf,
    /// A projection's input as `prepare` leaves it: the normed rows for f32 weights (a buffer
    /// read as it is needs no copy), else the rows quantized for the weights' format.
    pub(super) normed: SharedBuf,
    pub(super) quants: SharedBuf<i8>,
    pub(super) scales: SharedBuf,
    pub(super) corrections: SharedBuf<i32>,
    pub(crate) scratch: PerThread<Scratch>,
}

#[derive(Default)]
pub(crate) struct Scratch {
    /// This thread's share of the rows, normed.
    pub(crate) normed: Vec<f32>,
    pub(crate) quant: QuantizedRows,
    pub(crate) scores: Vec<f32>,
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
    /// A query or key head on its way through its norm.
    pub(crate) head: Vec<f32>,
    /// A gated feed-forward's `up` projection of this thread's columns.
    pub(super) up_share: Vec<f32>,
    /// A vision encoder's attention over all of an image's patches.
    pub(crate) attention: attention::Scratch,
}

/// The widths of a stack's activations.
pub(crate) struct Widths {
    pub(crate) hidden: usize,
    pub(crate) heads: Heads,
    /// Columns of the token mixer's input projection: `heads.qkv()` for attention.
    pub(crate) mixer: usize,
    pub(crate) feed_forward: usize,
}

impl Buffers {
    pub(crate) fn new(widths: &Widths, rows: usize, threads: usize) -> Self {
        let widest = widths
            .hidden
            .max(widths.heads.width())
            .max(widths.feed_forward);
        Self {
            rows,
            hidden_width: widths.hidden,
            hidden: SharedBuf::new(rows * widths.hidden),
            qkv: SharedBuf::new(rows * widths.mixer),
            attention: SharedBuf::new(rows * widths.heads.width()),
            inner: SharedBuf::new(rows * widths.feed_forward),
            normed: SharedBuf::new(rows * widths.hidden),
            quants: SharedBuf::new(rows * widest),
            scales: SharedBuf::new(rows * widest / BLOCK),
            corrections: SharedBuf::new(rows * widest / BLOCK),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    /// `hidden[row, cols] = x · Wᵀ + b` for this thread's columns (an input embedding).
    pub fn embed_row(&self, ctx: &Ctx, linear: &Linear, x: &[f32], row: usize) {
        let width = self.hidden_width;
        let input = self.scratch.get(ctx).quant.input(&linear.matrix(), x);
        let cols = ctx.split(width, COLUMN_ALIGN);
        linear.project(&input, cols, |_, col, value| {
            self.hidden.set(row * width + col, value);
        });
    }
}
