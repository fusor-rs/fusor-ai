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

use kernels::ops::{
    gelu_erf, gelu_tanh_lanes, quick_gelu, rms_norm, rope_half, rope_interleaved, silu,
};
use kernels::{
    BLOCK, COLUMN_ALIGN, Input, Linear, MatrixStorage, QUAD, Quantized, QuantizedRows, Vector,
    project_residual_input, quantize_rows,
};
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
        activation: Activation,
    },
}

/// A feed-forward's pointwise activation.
#[derive(Clone, Copy)]
pub(crate) enum Activation {
    GeluTanh,
    GeluErf,
    QuickGelu,
}

impl Activation {
    /// Of four values at a time. Always inlined: a call through a function pointer per quad
    /// of outputs would cost the projection's store a tenth of the kernel's time.
    #[inline(always)]
    fn apply(self, values: [f32; QUAD]) -> [f32; QUAD] {
        match self {
            Activation::GeluTanh => gelu_tanh_lanes(values),
            Activation::GeluErf => values.map(gelu_erf),
            Activation::QuickGelu => values.map(quick_gelu),
        }
    }
}

impl FeedForward {
    /// The projection that reads `norm2(hidden)` (the gate's input is the up's too).
    fn first(&self) -> &Linear {
        match self {
            FeedForward::Gated { gate, .. } => gate,
            FeedForward::Plain { up, .. } => up,
        }
    }

    fn down(&self) -> &Linear {
        match self {
            FeedForward::Gated { down, .. } | FeedForward::Plain { down, .. } => down,
        }
    }
}

pub(crate) struct Layer {
    pub(crate) norm1: Norm,
    pub(crate) qkv: Linear,
    pub(crate) head_norms: Option<HeadNorms>,
    pub(crate) out: Linear,
    pub(crate) norm2: Norm,
    pub(crate) feed_forward: FeedForward,
}

/// RMSNorms over each head of the queries and of the keys, `head_dim` wide.
pub(crate) struct HeadNorms {
    pub(crate) query: Vector,
    pub(crate) key: Vector,
    pub(crate) eps: f32,
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
    /// A projection's input as `prepare` leaves it: the normed rows for f32 weights (a buffer
    /// read as it is needs no copy), else the rows quantized for the weights' format.
    normed: SharedBuf,
    quants: SharedBuf<i8>,
    scales: SharedBuf,
    corrections: SharedBuf<i32>,
    pub(crate) scratch: PerThread<Scratch>,
}

#[derive(Default)]
pub(crate) struct Scratch {
    /// This thread's share of the rows, normed.
    pub(crate) normed: Vec<f32>,
    pub(crate) quant: QuantizedRows,
    pub(crate) scores: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// A query or key head on its way through its norm.
    head: Vec<f32>,
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
        let widest = widths
            .hidden
            .max(widths.heads.width())
            .max(widths.feed_forward);
        Self {
            rows,
            hidden_width: widths.hidden,
            hidden: SharedBuf::new(rows * widths.hidden),
            qkv: SharedBuf::new(rows * widths.heads.qkv()),
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

/// The rows a projection reads: the residual stream through a norm, or a buffer as it is.
#[derive(Clone, Copy)]
enum Source<'layer> {
    Normed(&'layer Norm),
    Buffer(&'layer SharedBuf),
}

/// This thread's share of `source` prepared as `linear`'s input, into the shared buffers:
/// quantized for quantized weights, else normed (a buffer is its own f32 input). The share is
/// of the quantization groups of all the rows (whole rows for f32 weights), so that one row is
/// shared out as evenly as thousands; a row's norm is computed by every thread holding part of
/// it.
fn prepare(pass: &Pass, source: Source, linear: &Linear) {
    let (buffers, matrix) = (pass.buffers, linear.matrix());
    let width = matrix.cols;
    let unit = match matrix.storage {
        MatrixStorage::F32(_) => width,
        MatrixStorage::Quant { format, .. } => format.group(),
    };
    let per_row = width / unit;
    let share = pass.ctx.split(pass.rows * per_row, 1);
    let scratch = buffers.scratch.get(pass.ctx);
    scratch.normed.resize(width, 0.0);
    for row in share.start / per_row..share.end.div_ceil(per_row) {
        let units = share.start.max(row * per_row)..share.end.min((row + 1) * per_row);
        let values = units.start * unit..units.end * unit;
        let within = values.start - row * width..values.end - row * width;
        let source_row: &[f32] = match source {
            Source::Normed(norm) => {
                let hidden = &buffers.hidden.read()[row * width..(row + 1) * width];
                norm.apply(hidden, &mut scratch.normed);
                &scratch.normed
            }
            Source::Buffer(buffer) => &buffer.read()[row * width..(row + 1) * width],
        };
        match (matrix.storage, source) {
            (MatrixStorage::F32(_), Source::Normed(_)) => {
                buffers
                    .normed
                    .part(values)
                    .copy_from_slice(&source_row[within]);
            }
            (MatrixStorage::F32(_), Source::Buffer(_)) => {}
            (MatrixStorage::Quant { format, .. }, _) => quantize_rows(
                &source_row[within],
                format,
                buffers.quants.part(values),
                buffers.scales.part(units.clone()),
                buffers.corrections.part(units),
            ),
        }
    }
}

/// The input `prepare` made for `linear` over the call's rows, once every thread's share is in
/// (after a barrier).
fn input<'pass>(pass: &Pass<'pass>, source: Source<'pass>, linear: &Linear) -> Input<'pass> {
    let (buffers, matrix) = (pass.buffers, linear.matrix());
    let len = pass.rows * matrix.cols;
    match (matrix.storage, source) {
        (MatrixStorage::F32(_), Source::Normed(_)) => Input::F32(&buffers.normed.read()[..len]),
        (MatrixStorage::F32(_), Source::Buffer(buffer)) => Input::F32(&buffer.read()[..len]),
        (MatrixStorage::Quant { format, .. }, _) => {
            let groups = len / format.group();
            Input::Quant(Quantized {
                format,
                quants: &buffers.quants.read()[..len],
                scales: &buffers.scales.read()[..groups],
                corrections: &buffers.corrections.read()[..groups],
            })
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
