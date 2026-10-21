//! NVIDIA's FastConformer encoder (Parakeet, Canary): a convolutional subsampler that folds
//! several mel frames into one encoder frame, then conformer layers. A layer is a half-step
//! feed-forward, Transformer-XL relative-position self-attention, a convolution module (pointwise
//! to twice the width, GLU, depthwise over time, BatchNorm, SiLU, pointwise back) and a second
//! half-step feed-forward, each pre-LayerNorm with a residual, then a final LayerNorm. The
//! feed-forwards use SiLU; no projection has a bias.
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

use bundle::{Load, Store};
use kernels::ops::{exp, silu};
use kernels::{
    COLUMN_ALIGN, DType, Error, Input, Linear, MatrixStorage, QuantizedRows, SharedRows, Tensor,
};
use kernels::{matmul, project_residual};
use team::{Ctx, PerThread, SharedBuf};

use crate::kyutai::Norm;
pub use subsampling::Subsampling;

/// The dimensions of one encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Mel bins of the input features.
    pub mels: usize,
    /// Channels of the subsampler's convolutions.
    pub channels: usize,
    /// The side of the subsampler's square windows, their stride, and how many stages it has.
    pub subsampling_kernel: usize,
    pub subsampling_stride: usize,
    pub subsampling_stages: usize,
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
}

pub struct FeedForward {
    pub norm: Norm,
    pub up: Linear,
    pub down: Linear,
}

pub struct Attention {
    pub norm: Norm,
    pub qkv: Linear,
    pub position: Linear,
    pub bias_u: Tensor,
    pub bias_v: Tensor,
    pub out: Linear,
}

pub struct Convolution {
    pub norm: Norm,
    pub pointwise1: Linear,
    pub depthwise: Tensor,
    pub depthwise_bias: Tensor,
    pub pointwise2: Linear,
}

pub struct Layer {
    pub ff1: FeedForward,
    pub attention: Attention,
    pub conv: Convolution,
    pub ff2: FeedForward,
    pub norm: Norm,
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
            qkv: SharedBuf::new(frames * 3 * shape.hidden),
            positions: SharedBuf::new(positions * shape.hidden),
            wide: SharedBuf::new(frames * wide),
            mixed: SharedBuf::new(frames * shape.hidden),
            table: attention::position_table(frames, shape.hidden),
            table_input: PerThread::new(threads, QuantizedRows::default),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    pub fn frames(&self) -> usize {
        self.frames
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
    pub fn new(shape: Shape, layers: Vec<Layer>) -> Result<Self, Error> {
        if layers.len() != shape.layers || !shape.hidden.is_multiple_of(shape.heads) {
            return Err(Error::Format(format!(
                "{} layers for a shape of {} layers of {} split in {} heads",
                layers.len(),
                shape.layers,
                shape.hidden,
                shape.heads
            )));
        }
        if layers
            .iter()
            .any(|layer| position_dtype(layer) != position_dtype(&layers[0]))
        {
            return Err(Error::Format(
                "relative-position projections of different formats".into(),
            ));
        }
        Ok(Self { shape, layers })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

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
        let table = buffers.table_input.get(ctx).input(&position, table, rows);
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
    fn normalize(&self, pass: &Pass, norm: &Norm, out: &SharedBuf) {
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
        project(pass, &weights.up, &buffers.normed, quant, |at, value| {
            buffers.wide.set(at, silu(value));
        });
        pass.ctx.barrier();
        project_residual(
            pass.ctx,
            &weights.down,
            pass.rows(&buffers.wide),
            &buffers.hidden,
            quant,
        );
        pass.ctx.barrier();
    }

    /// `hidden += out(attention(norm(hidden)))`.
    fn attention(&self, pass: &Pass, weights: &Attention, table: &Input) {
        let buffers = pass.buffers;
        self.normalize(pass, &weights.norm, &buffers.normed);
        pass.ctx.barrier();
        let scratch = buffers.scratch.get(pass.ctx);
        project(
            pass,
            &weights.qkv,
            &buffers.normed,
            &mut scratch.quant,
            |at, value| {
                buffers.qkv.set(at, value);
            },
        );
        let position = weights.position.matrix();
        let positions = 2 * pass.frames - 1;
        let cols = pass.ctx.split(position.rows, COLUMN_ALIGN);
        matmul(&position, table, positions, cols, |row, col, value| {
            buffers.positions.set(row * position.rows + col, value);
        });
        pass.ctx.barrier();
        attention::attend(self.shape, pass, weights, scratch);
        pass.ctx.barrier();
        let mixed = pass.rows(&buffers.mixed);
        project_residual(
            pass.ctx,
            &weights.out,
            mixed,
            &buffers.hidden,
            &mut scratch.quant,
        );
        pass.ctx.barrier();
    }

    /// `hidden += pointwise2(silu(depthwise(glu(pointwise1(norm(hidden))))))`, the BatchNorm
    /// folded into the depthwise kernel and bias.
    fn convolution(&self, pass: &Pass, weights: &Convolution) {
        let buffers = pass.buffers;
        self.normalize(pass, &weights.norm, &buffers.normed);
        pass.ctx.barrier();
        let scratch = buffers.scratch.get(pass.ctx);
        project(
            pass,
            &weights.pointwise1,
            &buffers.normed,
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
        let taps = weights.depthwise.values().expect("checked at load");
        let bias = &weights.depthwise_bias.values().expect("checked at load")[channels.clone()];
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
fn position_dtype(layer: &Layer) -> DType {
    match layer.attention.position.matrix().storage {
        MatrixStorage::F32(_) => DType::F32,
        MatrixStorage::Quant { dtype, .. } => dtype,
    }
}

/// `linear(input rows)` for this thread's share of the output columns, each value stored at
/// `row · outputs + col`.
fn project(
    pass: &Pass,
    linear: &Linear,
    input: &SharedBuf,
    quant: &mut QuantizedRows,
    mut store: impl FnMut(usize, f32),
) {
    let matrix = linear.matrix();
    let outputs = matrix.rows;
    let x = quant.input(
        &matrix,
        &input.read()[..pass.frames * matrix.cols],
        pass.frames,
    );
    let cols = pass.ctx.split(outputs, COLUMN_ALIGN);
    matmul(&matrix, &x, pass.frames, cols, |row, col, value| {
        store(row * outputs + col, value);
    });
}

impl Load for FeedForward {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm: Norm::load(store, &name("norm"), ())?,
            up: Linear::load(store, &name("up"), ())?,
            down: Linear::load(store, &name("down"), ())?,
        })
    }
}

impl Load for Attention {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm: Norm::load(store, &name("norm"), ())?,
            qkv: Linear::load(store, &name("qkv"), ())?,
            position: Linear::load(store, &name("position"), ())?,
            bias_u: vector(store, &name("bias_u"), shape.hidden)?,
            bias_v: vector(store, &name("bias_v"), shape.hidden)?,
            out: Linear::load(store, &name("out"), ())?,
        })
    }
}

impl Load for Convolution {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm: Norm::load(store, &name("norm"), ())?,
            pointwise1: Linear::load(store, &name("pointwise1"), ())?,
            depthwise: vector(store, &name("depthwise"), shape.kernel * shape.hidden)?,
            depthwise_bias: vector(store, &name("depthwise_bias"), shape.hidden)?,
            pointwise2: Linear::load(store, &name("pointwise2"), ())?,
        })
    }
}

impl Load for Layer {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            ff1: FeedForward::load(store, &name("ff1"), ())?,
            attention: Attention::load(store, &name("attention"), shape)?,
            conv: Convolution::load(store, &name("conv"), shape)?,
            ff2: FeedForward::load(store, &name("ff2"), ())?,
            norm: Norm::load(store, &name("norm"), ())?,
        })
    }
}

impl Load for Encoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let layers = (0..shape.layers)
            .map(|i| Layer::load(store, &format!("{prefix}.layers.{i}"), shape))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(shape, layers)?)
    }
}

/// Tensor `name`, which must hold `len` f32 values.
fn vector(store: &mut Store, name: &str, len: usize) -> Result<Tensor, bundle::Error> {
    let tensor = store.take(name)?;
    let found = tensor.values()?.len();
    if found != len {
        return Err(bundle::Error::Format(format!(
            "{name}: {found} values, not {len}"
        )));
    }
    Ok(tensor)
}
