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

use std::ops::Range;

use kernels::ops::{rms_norm, rope_half, rope_interleaved};
use kernels::{
    COLUMN_ALIGN, Input, Linear, QUAD, ROWS_PER_CALL, project_input, project_residual_input,
};
use team::Ctx;

use crate::rotary::{Position, Rotary};
pub use buffers::Buffers;
pub(crate) use buffers::{Scratch, Widths};
pub(crate) use input::{Source, input, prepare};
pub use spec::HeadNorm;
pub(crate) use spec::{FeedForwardSpec, LayerSpec, Rope};
pub use weights::Activation;
pub(crate) use weights::{FeedForward, HeadNorms, Layer, Norm};

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
        self.forward_layers(ctx, buffers, (rows, 0..self.layers.len()), attention);
    }

    /// `forward` through the layers in `layers` only.
    pub(crate) fn forward_layers(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        (rows, layers): (usize, Range<usize>),
        attention: &impl Attention,
    ) {
        assert!(rows <= buffers.rows, "the buffers hold the rows");
        assert!(layers.end <= self.layers.len(), "the stack has the layers");
        let pass = Pass {
            ctx,
            buffers,
            rows,
            heads: self.heads,
        };
        self.turn(&pass, attention);
        for index in layers {
            self.forward_layer(&pass, index, attention);
        }
    }

    /// `buffers.angles` for this thread's rows, then a barrier: the cosines and sines every
    /// layer of the call rotates its heads by, once and not per layer and thread.
    pub(crate) fn turn(&self, pass: &Pass, attention: &impl Attention) {
        if self.rope.is_none() {
            return;
        }
        let head_dim = self.heads.head_dim;
        for row in pass.ctx.split(pass.rows, 1) {
            let angles = pass
                .buffers
                .angles
                .part(row * head_dim..(row + 1) * head_dim);
            let (cos, sin) = angles.split_at_mut(head_dim / 2);
            self.rotary.angles(attention.rotary(row), cos, sin);
        }
        pass.ctx.barrier();
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Runs layer `index` over the pass's rows of `buffers.hidden`, after `turn`.
    pub(crate) fn forward_layer(&self, pass: &Pass, index: usize, attention: &impl Attention) {
        let layer = &self.layers[index];
        prepare(pass, Source::Normed(&layer.norm1), &layer.qkv);
        pass.ctx.barrier();
        self.attention_input(pass, index, layer, attention);
        pass.ctx.barrier();
        attention.attend(pass, index);
        pass.ctx.barrier();
        attended_residual(pass, &layer.out, None);
        feed_forward_residual(pass, &layer.norm2, &layer.feed_forward, None);
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
        scratch.head.resize(head_dim, 0.0);
        let rotated = self.heads.heads + self.heads.kv_heads;
        let angles = buffers.angles.read();
        for row in 0..pass.rows {
            let position = attention.position(row);
            let turn = angles[row * head_dim..(row + 1) * head_dim].split_at(head_dim / 2);
            for start in units.clone().step_by(head_dim) {
                let at = row * width + start;
                let head = buffers.qkv.part(at..at + head_dim);
                let unit = start / head_dim;
                if unit < rotated {
                    self.position(layer, unit, head, (&mut scratch.head, turn));
                }
                attention.keep(index, unit, position, head);
            }
        }
    }

    /// Query or key head `unit` normed through `copy`, where the layer norms heads, then
    /// rotated by its row's cosines and sines.
    fn position(
        &self,
        layer: &Layer,
        unit: usize,
        head: &mut [f32],
        (copy, (cos, sin)): (&mut [f32], (&[f32], &[f32])),
    ) {
        if let Some(norms) = &layer.head_norms {
            let weight = if unit < self.heads.heads {
                &norms.query
            } else {
                &norms.key
            };
            copy.copy_from_slice(head);
            rms_norm(copy, weight.values(), norms.eps, head);
        }
        match self.rope {
            Some(Rope::Half) => rope_half(head, cos, sin),
            Some(Rope::Interleaved) => rope_interleaved(head, cos, sin),
            None => {}
        }
    }
}

/// `hidden += out(attention)`, the token mixer's output (`buffers.attention`) projected back
/// into the residual stream, through `post` where the layer norms it: two phases, or three.
pub(crate) fn attended_residual(pass: &Pass, out: &Linear, post: Option<&Norm>) {
    let attended = Source::Buffer(&pass.buffers.attention);
    prepare(pass, attended, out);
    pass.ctx.barrier();
    let attended = input(pass, attended, out);
    add_residual(pass, out, &attended, post);
}

/// `hidden += feed_forward(norm(hidden))`, through `post` where the layer norms it: four phases,
/// or five.
pub(crate) fn feed_forward_residual(
    pass: &Pass,
    norm: &Norm,
    feed_forward: &FeedForward,
    post: Option<&Norm>,
) {
    prepare(pass, Source::Normed(norm), feed_forward.first());
    pass.ctx.barrier();
    inner(pass, norm, feed_forward);
    pass.ctx.barrier();
    let (inner, down) = (Source::Buffer(&pass.buffers.inner), feed_forward.down());
    prepare(pass, inner, down);
    pass.ctx.barrier();
    let inner = input(pass, inner, down);
    add_residual(pass, down, &inner, post);
}

/// `hidden += post(linear(input))`, or `hidden += linear(input)` without a norm: by columns, or
/// into `buffers.normed` by columns and then by rows. `normed` is free here: the projection reads
/// a buffer, and the next reads of `normed` follow the next `prepare`.
fn add_residual(pass: &Pass, linear: &Linear, input: &Input, post: Option<&Norm>) {
    let (ctx, buffers) = (pass.ctx, pass.buffers);
    let Some(post) = post else {
        project_residual_input(ctx, linear, input, &buffers.hidden);
        ctx.barrier();
        return;
    };
    project_input(ctx, linear, input, |at, value| {
        buffers.normed.set(at, value)
    });
    ctx.barrier();
    let width = linear.matrix().rows;
    let normed = &mut buffers.scratch.get(ctx).normed;
    normed.resize(width, 0.0);
    for row in ctx.split(pass.rows, 1) {
        let at = row * width..(row + 1) * width;
        post.apply(&buffers.normed.read()[at.clone()], normed);
        for (value, added) in buffers.hidden.part(at).iter_mut().zip(normed.iter()) {
            *value += added;
        }
    }
    ctx.barrier();
}

/// `inner[:, cols]`, the feed-forward of `norm(hidden)` before its last projection, for this
/// thread's columns, the kernel's `ROWS_PER_CALL` rows at a time: a block's outputs are still
/// in the cache when the activation sweeps them. The activation follows a block's projection
/// and is not in its store: there the kernel calls it out of line for every quad, and V8
/// builds its constants again at every call.
fn inner(pass: &Pass, norm: &Norm, feed_forward: &FeedForward) {
    let up_share = &mut pass.buffers.scratch.get(pass.ctx).up_share;
    let input = input(pass, Source::Normed(norm), feed_forward.first());
    let inner = &pass.buffers.inner;
    let matrix = feed_forward.first().matrix();
    let width = matrix.rows;
    let cols = pass.ctx.split(width, COLUMN_ALIGN);
    let share = cols.len();
    for start in (0..pass.rows).step_by(ROWS_PER_CALL) {
        let rows = start..(start + ROWS_PER_CALL).min(pass.rows);
        let block = input.block(matrix.cols, rows.clone());
        let store = |row: usize, first: usize, values: [f32; QUAD]| {
            let at = (start + row) * width + first;
            inner.part(at..at + QUAD).copy_from_slice(&values);
        };
        let mine = |row: usize| inner.part(row * width + cols.start..row * width + cols.end);
        match feed_forward {
            FeedForward::Plain { up, activation, .. } => {
                up.project_quads(&block, cols.clone(), store);
                for row in rows {
                    activation.sweep(mine(row));
                }
            }
            FeedForward::Gated {
                gate,
                up,
                activation,
                ..
            } => {
                gate.project_quads(&block, cols.clone(), store);
                up_share.resize(rows.len() * share, 0.0);
                up.project_quads(&block, cols.clone(), |row, first, values| {
                    let at = row * share + first - cols.start;
                    up_share[at..at + QUAD].copy_from_slice(&values);
                });
                for row in rows {
                    let gated = mine(row);
                    activation.sweep(gated);
                    let ups = &up_share[(row - start) * share..(row - start + 1) * share];
                    for (gated, up) in gated.iter_mut().zip(ups) {
                        *gated *= up;
                    }
                }
            }
        }
    }
}
