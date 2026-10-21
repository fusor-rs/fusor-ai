//! The pre-norm layer every transformer block here is made of: `x += out(attention(norm1(x)))`,
//! then `x += feed_forward(norm2(x))`.
//!
//! `Stack::forward` runs inside a team run, per layer five phases separated by barriers: QKV by
//! heads, with the rotary embeddings of the query and key heads and what the block's attention
//! keeps of each head; attention; the output projection plus residual by columns; the
//! feed-forward's first projections by their columns; its last projection plus residual by
//! columns.

use kernels::ops::{rms_norm, rope_half, rope_interleaved, silu};
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, SharedRows, Vector, project_residual};
use team::{Ctx, PerThread, SharedBuf};

use crate::attention;
use crate::norm::LayerNorm;
use crate::rotary::{Position, Rotary};

pub(crate) enum Norm {
    Rms { weight: Vector, eps: f32 },
    Layer { norm: LayerNorm, eps: f32 },
}

impl Norm {
    pub(crate) fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self {
            Norm::Rms { weight, eps } => rms_norm(x, weight.values(), *eps, out),
            Norm::Layer { norm, eps } => norm.apply(x, *eps, out),
        }
    }
}

pub(crate) enum FeedForward {
    /// `down(silu(gate(x)) · up(x))`.
    Gated {
        gate: Linear,
        up: Linear,
        down: Linear,
    },
    /// `down(activation(up(x)))`.
    Plain {
        up: Linear,
        down: Linear,
        activation: fn(f32) -> f32,
    },
}

impl FeedForward {
    fn down(&self) -> &Linear {
        match self {
            FeedForward::Gated { down, .. } | FeedForward::Plain { down, .. } => down,
        }
    }
}

pub(crate) struct Layer {
    pub(crate) norm1: Norm,
    pub(crate) qkv: Linear,
    pub(crate) out: Linear,
    pub(crate) norm2: Norm,
    pub(crate) feed_forward: FeedForward,
}

/// How rotary embeddings pair the elements of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rope {
    /// Element `i` with `i + d/2` (`rotate_half`, Llama/Qwen).
    Half,
    /// Element `2i` with `2i + 1` (x-transformers).
    Interleaved,
}

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

/// Activations of one forward call of up to `rows` positions.
pub struct Buffers {
    rows: usize,
    hidden_width: usize,
    /// The residual stream, `rows × hidden`: the input of `forward` and its output.
    pub hidden: SharedBuf,
    pub(crate) qkv: SharedBuf,
    pub(crate) attention: SharedBuf,
    /// The feed-forward's inner activations.
    inner: SharedBuf,
    pub(crate) scratch: PerThread<Scratch>,
}

#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) normed: Vec<f32>,
    pub(crate) quant: QuantizedRows,
    pub(crate) scores: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// A gated feed-forward's `up` projection of this thread's columns.
    up_share: Vec<f32>,
    /// A vision encoder's attention over all of an image's patches.
    pub(crate) attention: attention::Scratch,
}

/// The widths of a stack's activations.
pub(crate) struct Widths {
    pub(crate) hidden: usize,
    pub(crate) heads: Heads,
    pub(crate) feed_forward: usize,
}

impl Buffers {
    pub(crate) fn new(widths: &Widths, rows: usize, threads: usize) -> Self {
        Self {
            rows,
            hidden_width: widths.hidden,
            hidden: SharedBuf::new(rows * widths.hidden),
            qkv: SharedBuf::new(rows * widths.heads.qkv()),
            attention: SharedBuf::new(rows * widths.heads.width()),
            inner: SharedBuf::new(rows * widths.feed_forward),
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
        let quant = || &mut buffers.scratch.get(ctx).quant;
        let shared = |values| SharedRows { values, rows };
        for (index, layer) in self.layers.iter().enumerate() {
            self.attention_input(&pass, index, layer, attention);
            ctx.barrier();
            attention.attend(&pass, index);
            ctx.barrier();
            let attended = shared(&buffers.attention);
            project_residual(ctx, &layer.out, attended, &buffers.hidden, quant());
            ctx.barrier();
            feed_forward(&pass, layer);
            ctx.barrier();
            let inner = shared(&buffers.inner);
            let down = layer.feed_forward.down();
            project_residual(ctx, down, inner, &buffers.hidden, quant());
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
        normalize(pass, &layer.norm1, &mut scratch.normed);
        let input = scratch.quant.input(&layer.qkv.matrix(), &scratch.normed);
        let units = pass.ctx.split(width, head_dim);
        layer.qkv.project(&input, units.clone(), |row, col, value| {
            buffers.qkv.set(row * width + col, value);
        });
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
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
                    self.rotate(head, scratch);
                }
                attention.keep(index, unit, position, head);
            }
        }
    }

    fn rotate(&self, head: &mut [f32], scratch: &Scratch) {
        match self.rope {
            Some(Rope::Half) => rope_half(head, &scratch.cos, &scratch.sin),
            Some(Rope::Interleaved) => rope_interleaved(head, &scratch.cos, &scratch.sin),
            None => {}
        }
    }
}

/// `norm(hidden)` of the call's rows into this thread's scratch.
fn normalize(pass: &Pass, norm: &Norm, out: &mut Vec<f32>) {
    let len = pass.rows * pass.buffers.hidden_width;
    out.resize(len, 0.0);
    norm.apply(&pass.buffers.hidden.read()[..len], out);
}

/// `inner[:, cols]`, the feed-forward of `norm2(hidden)` before its last projection, for this
/// thread's columns.
fn feed_forward(pass: &Pass, layer: &Layer) {
    let Scratch {
        normed,
        quant,
        up_share,
        ..
    } = pass.buffers.scratch.get(pass.ctx);
    normalize(pass, &layer.norm2, normed);
    let inner = &pass.buffers.inner;
    match &layer.feed_forward {
        FeedForward::Plain { up, activation, .. } => {
            let width = up.matrix().rows;
            let input = quant.input(&up.matrix(), normed);
            let cols = pass.ctx.split(width, COLUMN_ALIGN);
            up.project(&input, cols, |row, col, value| {
                inner.set(row * width + col, activation(value));
            });
        }
        FeedForward::Gated { gate, up, .. } => {
            let width = gate.matrix().rows;
            let input = quant.input(&gate.matrix(), normed);
            let cols = pass.ctx.split(width, COLUMN_ALIGN);
            let share = cols.len();
            gate.project(&input, cols.clone(), |row, col, value| {
                inner.set(row * width + col, value);
            });
            up_share.resize(pass.rows * share, 0.0);
            up.project(&input, cols.clone(), |row, col, value| {
                up_share[row * share + col - cols.start] = value;
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
