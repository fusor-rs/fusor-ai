//! NVIDIA's FastConformer encoder (Parakeet, Canary): a convolutional subsampler that folds
//! several mel frames into one encoder frame, then conformer layers. A layer is a half-step
//! feed-forward, Transformer-XL relative-position self-attention, a convolution module (pointwise
//! to twice the width, GLU, depthwise over time, BatchNorm, SiLU, pointwise back) and a second
//! half-step feed-forward, each pre-LayerNorm with a residual, then a final LayerNorm. The
//! feed-forwards use SiLU.
//!
//! A checkpoint's compiler folds the half steps into the feed-forwards' `down` projections and
//! the BatchNorm into the depthwise kernel and its bias.
//!
//! `forward` runs inside a team run over one utterance, every frame valid: per layer, LayerNorms
//! by rows, projections by output columns (the relative positions' with the queries', keys' and
//! values'), attention by (head, query) pairs and the depthwise convolution by channels, with a
//! barrier between dependent phases. `subsampling` is the subsampler.
//!
//! Tensors under a prefix `p` (norms as `weight` and `bias` vectors):
//! - `p.layers.{i}.ff1.{norm, up, down}` and `ff2.{…}`;
//! - `p.layers.{i}.attention.{norm, qkv, position, out}`, linears, and `bias_u`, `bias_v`, the
//!   per-head content and position biases as `[hidden]` vectors (head-major);
//! - `p.layers.{i}.conv.{norm, pointwise1, pointwise2}`, `depthwise` (`[kernel, hidden]`, tap
//!   major) and `depthwise_bias` (`[hidden]`);
//! - `p.layers.{i}.norm`, the layer's final norm;
//! - the subsampler's, listed in `subsampling`.

mod attention;
pub mod subsampling;

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::{exp, silu};
use kernels::{
    COLUMN_ALIGN, Input, Linear, MatrixStorage, QuantFormat, QuantizedRows, SharedRows, Vector,
    project, project_residual,
};
use team::{Ctx, PerThread, SharedBuf};

use crate::norm::LayerNorm;
pub use subsampling::Subsampling;

/// The dimensions of one encoder's conformer layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub feed_forward: usize,
    /// Taps of the convolution module's depthwise kernel (odd, centred).
    pub kernel: usize,
    pub norm_eps: f32,
}

impl Shape {
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Columns of the fused QKV projection: the queries, then the keys, then the values.
    fn qkv(&self) -> usize {
        3 * self.hidden
    }
}

struct FeedForward {
    norm: LayerNorm,
    up: Linear,
    down: Linear,
}

struct Attention {
    norm: LayerNorm,
    qkv: Linear,
    position: Linear,
    bias_u: Vector,
    bias_v: Vector,
    out: Linear,
}

struct Convolution {
    norm: LayerNorm,
    pointwise1: Linear,
    depthwise: Vector,
    depthwise_bias: Vector,
    pointwise2: Linear,
}

struct Layer {
    ff1: FeedForward,
    attention: Attention,
    conv: Convolution,
    ff2: FeedForward,
    norm: LayerNorm,
}

pub struct Encoder {
    shape: Shape,
    layers: Vec<Layer>,
}

/// Activations of one `forward` over up to `frames` encoder frames.
pub struct Buffers {
    frames: usize,
    /// The residual stream, `frames × hidden`: the input of `forward` and its output.
    pub hidden: SharedBuf,
    normed: SharedBuf,
    qkv: SharedBuf,
    /// Every layer's projected relative positions, `(2 · frames − 1) × hidden`.
    positions: SharedBuf,
    /// The feed-forwards' inner activations and the convolution module's doubled width.
    wide: SharedBuf,
    /// The attention's and the depthwise convolution's outputs, `frames × hidden`.
    mixed: SharedBuf,
    /// The sinusoids of the relative positions `frames − 1` down to `1 − frames`.
    table: Vec<f32>,
    /// Per thread, the sinusoids of the call's positions as the projections read them, prepared
    /// once for every layer.
    table_input: PerThread<QuantizedRows>,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    quant: QuantizedRows,
    normed: Vec<f32>,
    /// A head's query plus the content bias, then plus the position bias.
    queries: Vec<f32>,
    scores: Vec<f32>,
    gated: Vec<f32>,
    sums: Vec<f32>,
}

impl Buffers {
    pub fn new(shape: Shape, frames: usize, threads: usize) -> Self {
        let wide = shape.feed_forward.max(2 * shape.hidden);
        let positions = 2 * frames - 1;
        Self {
            frames,
            hidden: SharedBuf::new(frames * shape.hidden),
            normed: SharedBuf::new(frames * shape.hidden),
            qkv: SharedBuf::new(frames * shape.qkv()),
            positions: SharedBuf::new(positions * shape.hidden),
            wide: SharedBuf::new(frames * wide),
            mixed: SharedBuf::new(frames * shape.hidden),
            table: attention::position_table(frames, shape.hidden),
            table_input: PerThread::new(threads, QuantizedRows::default),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

/// Everything one thread's pass over the layers reads.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    frames: usize,
}

impl Pass<'_> {
    fn rows<'buf>(&self, values: &'buf SharedBuf) -> SharedRows<'buf> {
        SharedRows {
            values,
            rows: self.frames,
        }
    }
}

impl Encoder {
    /// Runs every layer over `buffers.hidden[..frames]` (in and out).
    ///
    /// # Panics
    /// When `frames` is 0 or more than the buffers hold.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, frames: usize) {
        assert!(
            (1..=buffers.frames).contains(&frames),
            "the buffers hold the frames"
        );
        let pass = Pass {
            ctx,
            buffers,
            frames,
        };
        let Some(first) = self.layers.first() else {
            return;
        };
        let rows = 2 * frames - 1;
        let start = (buffers.frames - frames) * self.shape.hidden;
        let table = &buffers.table[start..start + rows * self.shape.hidden];
        let position = first.attention.position.matrix();
        let table = buffers.table_input.get(ctx).input(&position, table);
        for layer in &self.layers {
            self.feed_forward(&pass, &layer.ff1);
            self.attention(&pass, &layer.attention, &table);
            self.convolution(&pass, &layer.conv);
            self.feed_forward(&pass, &layer.ff2);
            self.normalize(&pass, &layer.norm, &buffers.hidden);
            ctx.barrier();
        }
    }

    /// `out = norm(hidden)` for this thread's rows (`out` may be `hidden`).
    fn normalize(&self, pass: &Pass, norm: &LayerNorm, out: &SharedBuf) {
        let hidden = self.shape.hidden;
        let rows = pass.ctx.split(pass.frames, 1);
        let range = rows.start * hidden..rows.end * hidden;
        let normed = &mut pass.buffers.scratch.get(pass.ctx).normed;
        normed.resize(range.len(), 0.0);
        norm.apply(
            &pass.buffers.hidden.read()[range.clone()],
            self.shape.norm_eps,
            normed,
        );
        out.part(range).copy_from_slice(normed);
    }

    /// `hidden += down(silu(up(norm(hidden))))`.
    fn feed_forward(&self, pass: &Pass, weights: &FeedForward) {
        let buffers = pass.buffers;
        self.normalize(pass, &weights.norm, &buffers.normed);
        pass.ctx.barrier();
        let quant = &mut buffers.scratch.get(pass.ctx).quant;
        let normed = pass.rows(&buffers.normed);
        project(pass.ctx, &weights.up, normed, quant, |at, value| {
            buffers.wide.set(at, silu(value));
        });
        pass.ctx.barrier();
        let wide = pass.rows(&buffers.wide);
        project_residual(pass.ctx, &weights.down, wide, &buffers.hidden, quant);
        pass.ctx.barrier();
    }

    /// `hidden += out(attention(norm(hidden)))`.
    fn attention(&self, pass: &Pass, weights: &Attention, table: &Input) {
        let buffers = pass.buffers;
        self.normalize(pass, &weights.norm, &buffers.normed);
        pass.ctx.barrier();
        let scratch = buffers.scratch.get(pass.ctx);
        let normed = pass.rows(&buffers.normed);
        project(
            pass.ctx,
            &weights.qkv,
            normed,
            &mut scratch.quant,
            |at, value| {
                buffers.qkv.set(at, value);
            },
        );
        let outputs = weights.position.matrix().rows;
        let cols = pass.ctx.split(outputs, COLUMN_ALIGN);
        weights.position.project(table, cols, |row, col, value| {
            buffers.positions.set(row * outputs + col, value);
        });
        pass.ctx.barrier();
        attention::attend(self.shape, pass, weights, scratch);
        pass.ctx.barrier();
        let mixed = pass.rows(&buffers.mixed);
        let quant = &mut scratch.quant;
        project_residual(pass.ctx, &weights.out, mixed, &buffers.hidden, quant);
        pass.ctx.barrier();
    }

    /// `hidden += pointwise2(silu(depthwise(glu(pointwise1(norm(hidden))))))`, the BatchNorm
    /// folded into the depthwise kernel and bias.
    fn convolution(&self, pass: &Pass, weights: &Convolution) {
        let buffers = pass.buffers;
        self.normalize(pass, &weights.norm, &buffers.normed);
        pass.ctx.barrier();
        let scratch = buffers.scratch.get(pass.ctx);
        let normed = pass.rows(&buffers.normed);
        project(
            pass.ctx,
            &weights.pointwise1,
            normed,
            &mut scratch.quant,
            |at, value| {
                buffers.wide.set(at, value);
            },
        );
        pass.ctx.barrier();
        self.depthwise(pass, weights, scratch);
        pass.ctx.barrier();
        let mixed = pass.rows(&buffers.mixed);
        let quant = &mut scratch.quant;
        project_residual(pass.ctx, &weights.pointwise2, mixed, &buffers.hidden, quant);
        pass.ctx.barrier();
    }

    /// GLU, the depthwise convolution over time (zeros beyond the frames), its bias and SiLU,
    /// for this thread's channels: `wide` (`2 · hidden` per frame) into `mixed`.
    fn depthwise(&self, pass: &Pass, weights: &Convolution, scratch: &mut Scratch) {
        let Shape { hidden, kernel, .. } = self.shape;
        let channels = pass.ctx.split(hidden, COLUMN_ALIGN);
        let width = channels.len();
        let wide = pass.buffers.wide.read();
        let Scratch { gated, sums, .. } = scratch;
        gated.clear();
        for frame in 0..pass.frames {
            let row = &wide[frame * 2 * hidden..(frame + 1) * 2 * hidden];
            let (values, gates) = row.split_at(hidden);
            let pairs = values[channels.clone()]
                .iter()
                .zip(&gates[channels.clone()]);
            gated.extend(pairs.map(|(value, gate)| value / (1.0 + exp(-gate))));
        }
        let taps = weights.depthwise.values();
        let bias = &weights.depthwise_bias.values()[channels.clone()];
        let reach = kernel / 2;
        sums.resize(width, 0.0);
        for frame in 0..pass.frames {
            sums.copy_from_slice(bias);
            let first = frame.saturating_sub(reach);
            let last = (frame + reach).min(pass.frames - 1);
            for source in first..=last {
                let tap = &taps[(source + reach - frame) * hidden..][channels.clone()];
                let input = &gated[source * width..(source + 1) * width];
                for ((sum, weight), value) in sums.iter_mut().zip(tap).zip(input) {
                    *sum += weight * value;
                }
            }
            let start = frame * hidden + channels.start;
            let out = pass.buffers.mixed.part(start..start + width);
            for (output, sum) in out.iter_mut().zip(sums.iter()) {
                *output = silu(*sum);
            }
        }
    }
}

/// The format of a layer's relative-position projection (`forward` prepares the sinusoids for
/// the first layer's and every layer reads them).
fn position_format(layer: &Layer) -> Option<QuantFormat> {
    match layer.attention.position.matrix().storage {
        MatrixStorage::F32(_) => None,
        MatrixStorage::Quant { format, .. } => Some(format),
    }
}

impl Load for FeedForward {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            feed_forward,
            ..
        } = shape;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm: scope.load("norm", hidden)?,
            up: scope.load("up", [feed_forward, hidden])?,
            down: scope.load("down", [hidden, feed_forward])?,
        })
    }
}

impl Load for Attention {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let hidden = shape.hidden;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm: scope.load("norm", hidden)?,
            qkv: scope.load("qkv", [shape.qkv(), hidden])?,
            position: scope.load("position", [hidden, hidden])?,
            bias_u: scope.load("bias_u", hidden)?,
            bias_v: scope.load("bias_v", hidden)?,
            out: scope.load("out", [hidden, hidden])?,
        })
    }
}

impl Load for Convolution {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let hidden = shape.hidden;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm: scope.load("norm", hidden)?,
            pointwise1: scope.load("pointwise1", [2 * hidden, hidden])?,
            depthwise: scope.load("depthwise", shape.kernel * hidden)?,
            depthwise_bias: scope.load("depthwise_bias", hidden)?,
            pointwise2: scope.load("pointwise2", [hidden, hidden])?,
        })
    }
}

impl Load for Layer {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            ff1: scope.load("ff1", shape)?,
            attention: scope.load("attention", shape)?,
            conv: scope.load("conv", shape)?,
            ff2: scope.load("ff2", shape)?,
            norm: scope.load("norm", shape.hidden)?,
        })
    }
}

impl Load for Encoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        if !shape.hidden.is_multiple_of(shape.heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: a width of {} in {} heads",
                shape.hidden, shape.heads
            )));
        }
        let layers: Vec<Layer> = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| Layer::load(store, name, shape),
        )?;
        if let Some(first) = layers.first()
            && layers
                .iter()
                .any(|layer| position_format(layer) != position_format(first))
        {
            return Err(bundle::Error::Format(format!(
                "{prefix}: relative-position projections of different formats"
            )));
        }
        Ok(Self { shape, layers })
    }
}
