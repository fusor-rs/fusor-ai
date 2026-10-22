//! The pre-norm layer every transformer block here is made of: `x += out(attention(norm1(x)))`,
//! then `x += feed_forward(norm2(x))`; a layer may RMS-normalize each query and key head before
//! the rotary embeddings (Qwen3).
//!
//! `Stack::forward` runs inside a team run, per layer nine phases separated by barriers: before
//! each of the four projections, every thread prepares its share of the rows as that
//! projection's input (normed where the layer norms them, quantized where the weights are
//! quantized) into buffers the team shares; then QKV by heads, with the rotary embeddings of
//! the query and key heads and what the block's attention keeps of each head; attention; the
//! output projection plus residual by columns; the feed-forward's first projections by their
//! columns; its last projection plus residual by columns.

mod buffers;
mod input;
mod spec;
mod weights;

use kernels::ops::{rms_norm, rope_half, rope_interleaved, silu};
use kernels::{COLUMN_ALIGN, QUAD, project_residual_input};
use team::Ctx;

use crate::rotary::{Position, Rotary};
pub use buffers::Buffers;
pub(crate) use buffers::{Scratch, Widths};
use input::{Source, input, prepare};
pub(crate) use spec::{FeedForwardSpec, HeadNorm, LayerSpec, Rope};
pub(crate) use weights::{Activation, FeedForward, Layer, Norm};

/// The attention heads of a stack: `heads` query heads over `kv_heads` key and value heads, all
/// `head_dim` wide.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Heads {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
}

impl Heads {
    /// Columns of the fused QKV projection: query heads, then key heads, then value heads.
    pub(crate) fn qkv(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    /// Columns of the attention's output.
    pub(crate) fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

pub(crate) struct Stack {
    heads: Heads,
    /// `None` for stacks without rotary embeddings (learned positions, added to the input).
    rope: Option<Rope>,
    rotary: Rotary,
    layers: Vec<Layer>,
}

/// Everything one thread's pass over the layers reads.
pub(crate) struct Pass<'call> {
    pub(crate) ctx: &'call Ctx<'call>,
    pub(crate) buffers: &'call Buffers,
    pub(crate) rows: usize,
    pub(crate) heads: Heads,
}

/// What a block's attention does with the QKV rows of a call.
pub(crate) trait Attention {
    /// The sequence position of row `row`.
    fn position(&self, row: usize) -> usize;

    /// The position row `row`'s rotary angles turn by: its sequence position on every axis.
    fn rotary(&self, row: usize) -> Position {
        Position::uniform(self.position(row))
    }

    /// Keeps head `unit` (query heads, then key heads, then value heads) of a row at `position`
    /// of layer `layer`, once rotated: a KV cache keeps the key and value heads.
    fn keep(&self, layer: usize, unit: usize, position: usize, head: &[f32]);

    /// This thread's share of the attention of layer `layer` into `buffers.attention`.
    fn attend(&self, pass: &Pass, layer: usize);
}

impl Stack {
    pub(crate) fn new(
        heads: Heads,
        rope: Option<Rope>,
        rotary: Rotary,
        layers: Vec<Layer>,
    ) -> Self {
        Self {
            heads,
            rope,
            rotary,
            layers,
        }
    }

    /// Runs every layer over `buffers.hidden[..rows]` (the residual stream, in and out).
    pub(crate) fn forward(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        rows: usize,
        attention: &impl Attention,
    ) {
        assert!(rows <= buffers.rows, "the buffers hold the rows");
        let pass = Pass {
            ctx,
            buffers,
            rows,
            heads: self.heads,
        };
        for (index, layer) in self.layers.iter().enumerate() {
            prepare(&pass, Source::Normed(&layer.norm1), &layer.qkv);
            ctx.barrier();
            self.attention_input(&pass, index, layer, attention);
            ctx.barrier();
            attention.attend(&pass, index);
            ctx.barrier();
            let attended = Source::Buffer(&buffers.attention);
            prepare(&pass, attended, &layer.out);
            ctx.barrier();
            let attended = input(&pass, attended, &layer.out);
            project_residual_input(ctx, &layer.out, &attended, &buffers.hidden);
            ctx.barrier();
            prepare(
                &pass,
                Source::Normed(&layer.norm2),
                layer.feed_forward.first(),
            );
            ctx.barrier();
            feed_forward(&pass, layer);
            ctx.barrier();
            let (inner, down) = (Source::Buffer(&buffers.inner), layer.feed_forward.down());
            prepare(&pass, inner, down);
            ctx.barrier();
            project_residual_input(ctx, down, &input(&pass, inner, down), &buffers.hidden);
            ctx.barrier();
        }
    }

    /// QKV for this thread's heads, RoPE on its query and key heads, each head kept as the
    /// attention needs it.
    fn attention_input(
        &self,
        pass: &Pass,
        index: usize,
        layer: &Layer,
        attention: &impl Attention,
    ) {
        let head_dim = self.heads.head_dim;
        let (buffers, width) = (pass.buffers, self.heads.qkv());
        let scratch = buffers.scratch.get(pass.ctx);
        let input = input(pass, Source::Normed(&layer.norm1), &layer.qkv);
        let units = pass.ctx.split(width, head_dim);
        layer
            .qkv
            .project_quads(&input, units.clone(), |row, first, values| {
                let at = row * width + first;
                buffers.qkv.part(at..at + QUAD).copy_from_slice(&values);
            });
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        scratch.head.resize(head_dim, 0.0);
        let rotated = self.heads.heads + self.heads.kv_heads;
        for row in 0..pass.rows {
            let position = attention.position(row);
            if self.rope.is_some() {
                let rotary = attention.rotary(row);
                self.rotary
                    .angles(rotary, &mut scratch.cos, &mut scratch.sin);
            }
            for start in units.clone().step_by(head_dim) {
                let at = row * width + start;
                let head = buffers.qkv.part(at..at + head_dim);
                let unit = start / head_dim;
                if unit < rotated {
                    self.position(layer, unit, head, scratch);
                }
                attention.keep(index, unit, position, head);
            }
        }
    }

    /// Query or key head `unit` normed, where the layer norms heads, then rotated.
    fn position(&self, layer: &Layer, unit: usize, head: &mut [f32], scratch: &mut Scratch) {
        if let Some(norms) = &layer.head_norms {
            let weight = if unit < self.heads.heads {
                &norms.query
            } else {
                &norms.key
            };
            scratch.head.copy_from_slice(head);
            rms_norm(&scratch.head, weight.values(), norms.eps, head);
        }
        self.rotate(head, scratch);
    }

    fn rotate(&self, head: &mut [f32], scratch: &Scratch) {
        match self.rope {
            Some(Rope::Half) => rope_half(head, &scratch.cos, &scratch.sin),
            Some(Rope::Interleaved) => rope_interleaved(head, &scratch.cos, &scratch.sin),
            None => {}
        }
    }
}

/// `inner[:, cols]`, the feed-forward of `norm2(hidden)` before its last projection, for this
/// thread's columns.
fn feed_forward(pass: &Pass, layer: &Layer) {
    let up_share = &mut pass.buffers.scratch.get(pass.ctx).up_share;
    let input = input(
        pass,
        Source::Normed(&layer.norm2),
        layer.feed_forward.first(),
    );
    let inner = &pass.buffers.inner;
    match &layer.feed_forward {
        FeedForward::Plain { up, activation, .. } => {
            let width = up.matrix().rows;
            let cols = pass.ctx.split(width, COLUMN_ALIGN);
            up.project_quads(&input, cols, |row, first, values| {
                let at = row * width + first;
                inner
                    .part(at..at + QUAD)
                    .copy_from_slice(&activation.apply(values));
            });
        }
        FeedForward::Gated { gate, up, .. } => {
            let width = gate.matrix().rows;
            let cols = pass.ctx.split(width, COLUMN_ALIGN);
            let share = cols.len();
            gate.project_quads(&input, cols.clone(), |row, first, values| {
                let at = row * width + first;
                inner.part(at..at + QUAD).copy_from_slice(&values);
            });
            up_share.resize(pass.rows * share, 0.0);
            up.project_quads(&input, cols.clone(), |row, first, values| {
                let at = row * share + first - cols.start;
                up_share[at..at + QUAD].copy_from_slice(&values);
            });
            for row in 0..pass.rows {
                let gated = inner.part(row * width + cols.start..row * width + cols.end);
                let ups = &up_share[row * share..(row + 1) * share];
                for (gated, up) in gated.iter_mut().zip(ups) {
                    *gated = silu(*gated) * up;
                }
            }
        }
    }
}
