//! ECAPA-TDNN speaker encoder (Qwen2.5-Omni's and Qwen3-TTS's): a time-delay conv over mel
//! features, `channels.len() − 2` SE-Res2Net blocks (a 1-wide conv, a Res2Net of `scale − 1`
//! convs over the channel groups, a 1-wide conv, a squeeze-excitation of the time mean, plus the
//! block's input), a multi-layer aggregation conv over the blocks' outputs, attentive statistics
//! pooling (the mean and deviation of the frames, weighted by a per-channel softmax over time),
//! and a linear layer into the speaker's embedding.
//!
//! Features and activations are time-major rows of channels. Every conv pads its input by
//! reflection (`padding="same"`) and is followed by a ReLU, except the pooling's score conv; the
//! compiler lays out each as `[out, kernel · in]`, columns oldest tap first.
//!
//! `embed` runs inside a team run, one phase per conv or reduction: each conv is split by rows
//! while there are enough and by columns after (`causal::project_windows`), the reductions by
//! channels.
//!
//! Tensors under a prefix `p`: `p.blocks.0.conv`, `p.blocks.{b}.{tdnn1, tdnn2, se.squeeze,
//! se.excite, res2net.{j}}`, `p.mfa`, `p.pool.{attention, score}` and `p.out`, each a linear
//! layer.

use bundle::{Load, Store};
use kernels::Linear;
use team::{Ctx, PerThread, SharedBuf};

use crate::causal::{Scratch, project_windows};

/// The pooling's variance floor.
const VARIANCE_FLOOR: f32 = 1e-12;

/// The dimensions of one speaker encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Channels of the input features.
    pub mels: usize,
    /// Output channels of the first conv, the SE-Res2Net blocks and the aggregation conv.
    pub channels: [usize; 5],
    pub kernels: [usize; 5],
    pub dilations: [usize; 5],
    /// Channels inside the pooling's attention.
    pub attention: usize,
    /// Groups of a Res2Net.
    pub scale: usize,
    /// Channels inside a squeeze-excitation.
    pub squeeze: usize,
    pub embedding: usize,
}

impl Shape {
    /// The SE-Res2Net blocks' count.
    fn blocks(&self) -> usize {
        self.channels.len() - 2
    }

    /// Channels of an SE-Res2Net block, the same in all.
    fn width(&self) -> usize {
        self.channels[1]
    }

    /// Channels of the aggregation conv's input: the blocks' outputs side by side.
    fn aggregated(&self) -> usize {
        self.width() * self.blocks()
    }

    /// Channels of the aggregation conv's output, which the pooling reads.
    fn pooled(&self) -> usize {
        self.channels[self.channels.len() - 1]
    }
}

struct Block {
    tdnn1: Linear,
    res2net: Vec<Linear>,
    tdnn2: Linear,
    squeeze: Linear,
    excite: Linear,
}

pub struct Ecapa {
    shape: Shape,
    first: Linear,
    blocks: Vec<Block>,
    mfa: Linear,
    attention: Linear,
    score: Linear,
    out: Linear,
}

/// Each conv a linear layer of `[out, kernel · in]`.
impl Load for Ecapa {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let width = shape.width();
        let group = width / shape.scale;
        let mut blocks = Vec::new();
        for block in 1..=shape.blocks() {
            let name = format!("{prefix}.blocks.{block}");
            let res2net = (0..shape.scale - 1)
                .map(|conv| {
                    let matrix = [group, shape.kernels[block] * group];
                    Linear::load(store, &format!("{name}.res2net.{conv}"), matrix)
                })
                .collect::<Result<_, _>>()?;
            blocks.push(Block {
                tdnn1: Linear::load(store, &format!("{name}.tdnn1"), [width, width])?,
                res2net,
                tdnn2: Linear::load(store, &format!("{name}.tdnn2"), [width, width])?,
                squeeze: Linear::load(
                    store,
                    &format!("{name}.se.squeeze"),
                    [shape.squeeze, width],
                )?,
                excite: Linear::load(store, &format!("{name}.se.excite"), [width, shape.squeeze])?,
            });
        }
        let pooled = shape.pooled();
        let last = shape.channels.len() - 1;
        let first = [shape.channels[0], shape.kernels[0] * shape.mels];
        let aggregation = [pooled, shape.kernels[last] * shape.aggregated()];
        Ok(Self {
            first: Linear::load(store, &format!("{prefix}.blocks.0.conv"), first)?,
            blocks,
            mfa: Linear::load(store, &format!("{prefix}.mfa"), aggregation)?,
            attention: Linear::load(
                store,
                &format!("{prefix}.pool.attention"),
                [shape.attention, 3 * pooled],
            )?,
            score: Linear::load(
                store,
                &format!("{prefix}.pool.score"),
                [pooled, shape.attention],
            )?,
            out: Linear::load(
                store,
                &format!("{prefix}.out"),
                [shape.embedding, 2 * pooled],
            )?,
            shape,
        })
    }
}

/// The activations of one utterance of up to `frames` feature frames.
pub struct Buffers {
    frames: usize,
    first: SharedBuf,
    /// The SE-Res2Net block in flight: the first 1-wide conv's output, the Res2Net's, the second
    /// 1-wide conv's.
    tdnn1: SharedBuf,
    res2net: SharedBuf,
    tdnn2: SharedBuf,
    aggregated: SharedBuf,
    features: SharedBuf,
    attended: SharedBuf,
    scores: SharedBuf,
    /// The time mean of the block's output, the squeezed mean, and the excitation per channel.
    mean: SharedBuf,
    squeezed: SharedBuf,
    excitation: SharedBuf,
    /// The pooling's mean and deviation of the frames, then of the attention-weighted frames.
    moments: SharedBuf,
    pooled: SharedBuf,
    /// The speaker's embedding.
    pub embedding: SharedBuf,
    scratch: PerThread<Scratch>,
}

impl Buffers {
    pub fn new(shape: &Shape, frames: usize, threads: usize) -> Self {
        let width = shape.width();
        let pooled = shape.pooled();
        Self {
            frames,
            first: SharedBuf::new(frames * shape.channels[0]),
            tdnn1: SharedBuf::new(frames * width),
            res2net: SharedBuf::new(frames * width),
            tdnn2: SharedBuf::new(frames * width),
            aggregated: SharedBuf::new(frames * shape.aggregated()),
            features: SharedBuf::new(frames * pooled),
            attended: SharedBuf::new(frames * shape.attention),
            scores: SharedBuf::new(frames * pooled),
            mean: SharedBuf::new(width),
            squeezed: SharedBuf::new(shape.squeeze),
            excitation: SharedBuf::new(width),
            moments: SharedBuf::new(2 * pooled),
            pooled: SharedBuf::new(2 * pooled),
            embedding: SharedBuf::new(shape.embedding),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

/// Where a conv's input channels are: `values` holds rows of `stride` values and the channels
/// are the `inputs` from `offset` on in each.
#[derive(Clone, Copy)]
struct Part<'rows> {
    values: &'rows [f32],
    stride: usize,
    offset: usize,
}

/// A conv's reflect-padded windows over the sum of its `parts` (a plain conv has one).
struct Window<'rows> {
    parts: &'rows [Part<'rows>],
    kernel: usize,
    dilation: usize,
    frames: usize,
    inputs: usize,
}

impl Window<'_> {
    /// The rows output frame `row` reads, oldest first.
    fn fill(&self, row: usize, window: &mut [f32]) {
        let left = (self.dilation * (self.kernel - 1) / 2) as isize;
        for (tap, values) in window.chunks_exact_mut(self.inputs).enumerate() {
            let position = (row + tap * self.dilation) as isize - left;
            let source = reflect(position, self.frames);
            values.fill(0.0);
            for part in self.parts {
                let channels = &part.values[source * part.stride + part.offset..][..self.inputs];
                for (value, channel) in values.iter_mut().zip(channels) {
                    *value += channel;
                }
            }
        }
    }
}

/// Channel `channel` of every row of `values` (rows of `width`).
fn column(values: &[f32], channel: usize, width: usize) -> impl Iterator<Item = f32> + '_ {
    values.iter().skip(channel).step_by(width).copied()
}

/// The frame `position` of a sequence of `frames` extended by reflection around its ends.
fn reflect(position: isize, frames: usize) -> usize {
    let last = frames as isize - 1;
    let folded = if position < 0 {
        -position
    } else if position > last {
        2 * last - position
    } else {
        position
    };
    folded as usize
}

/// A team run's view of one utterance.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    frames: usize,
}

impl Pass<'_> {
    /// `store(frame, channel, product)` of every frame of `window`, then a barrier.
    fn convolve(&self, linear: &Linear, window: &Window, store: impl Fn(usize, usize, f32)) {
        self.product(linear, |row, out| window.fill(row, out), store);
    }

    /// `store(frame, channel, product)` of every frame's row `gather` fills, then a barrier.
    fn product(
        &self,
        linear: &Linear,
        gather: impl Fn(usize, &mut [f32]),
        store: impl Fn(usize, usize, f32),
    ) {
        let scratch = self.buffers.scratch.get(self.ctx);
        project_windows(self.ctx, (linear, self.frames), scratch, gather, store);
        self.ctx.barrier();
    }

    /// `store(channel, product)` of one row `input`, then a barrier.
    fn project_row(&self, linear: &Linear, input: &[f32], store: impl Fn(usize, f32)) {
        let scratch = self.buffers.scratch.get(self.ctx);
        let gather = |_: usize, out: &mut [f32]| out.copy_from_slice(input);
        let store = |_: usize, col: usize, value: f32| store(col, value);
        project_windows(self.ctx, (linear, 1), scratch, gather, store);
        self.ctx.barrier();
    }

    /// This thread's channels of the time mean of `values` (rows of `width`) into `out`, then
    /// a barrier.
    fn time_mean(&self, values: &[f32], width: usize, out: &SharedBuf) {
        for channel in self.ctx.split(width, 1) {
            let sum: f32 = column(values, channel, width).sum();
            out.set(channel, sum / self.frames as f32);
        }
        self.ctx.barrier();
    }

    /// The windows of a conv over the sum of `parts`, each `inputs` channels, of `taps`
    /// (kernel, dilation).
    fn window<'rows>(
        &self,
        parts: &'rows [Part<'rows>],
        taps: (usize, usize),
        inputs: usize,
    ) -> Window<'rows> {
        Window {
            parts,
            kernel: taps.0,
            dilation: taps.1,
            frames: self.frames,
            inputs,
        }
    }
}

impl Ecapa {
    /// Inside a team run: the embedding of `frames × mels` time-major `features` into
    /// `buffers.embedding`.
    ///
    /// # Panics
    /// With more frames than the buffers hold, or no more than a conv's padding.
    pub fn embed(&self, ctx: &Ctx, buffers: &Buffers, features: &[f32], frames: usize) {
        assert!(frames <= buffers.frames, "the buffers hold the utterance");
        let pass = Pass {
            ctx,
            buffers,
            frames,
        };
        let shape = &self.shape;
        let part = Part {
            values: features,
            stride: shape.mels,
            offset: 0,
        };
        let parts = [part];
        let window = pass.window(&parts, (shape.kernels[0], shape.dilations[0]), shape.mels);
        let width = shape.channels[0];
        pass.convolve(&self.first, &window, |frame, channel, value| {
            buffers.first.set(frame * width + channel, value.max(0.0));
        });
        let mut previous = Part {
            values: buffers.first.read(),
            stride: width,
            offset: 0,
        };
        for (index, block) in self.blocks.iter().enumerate() {
            self.block(&pass, (index + 1, block), previous);
            previous = Part {
                values: buffers.aggregated.read(),
                stride: shape.aggregated(),
                offset: index * shape.width(),
            };
        }
        self.pool(&pass);
    }

    /// SE-Res2Net block `index` (1-based) over `previous`, its output into its columns of
    /// `aggregated`.
    fn block(&self, pass: &Pass, (index, block): (usize, &Block), previous: Part) {
        let buffers = pass.buffers;
        let shape = &self.shape;
        let width = shape.width();
        let group = width / shape.scale;
        let taps = (shape.kernels[index], shape.dilations[index]);
        let window = pass.window(std::slice::from_ref(&previous), (1, 1), width);
        pass.convolve(&block.tdnn1, &window, |frame, channel, value| {
            let value = value.max(0.0);
            buffers.tdnn1.set(frame * width + channel, value);
            if channel < group {
                buffers.res2net.set(frame * width + channel, value);
            }
        });
        let split = buffers.tdnn1.read();
        let summed = buffers.res2net.read();
        for (conv, linear) in block.res2net.iter().enumerate() {
            let offset = (conv + 1) * group;
            let input = Part {
                values: split,
                stride: width,
                offset,
            };
            let carried = Part {
                values: summed,
                stride: width,
                offset: offset - group,
            };
            let parts = [input, carried];
            let used = if conv == 0 { 1 } else { 2 };
            let window = pass.window(&parts[..used], taps, group);
            pass.convolve(linear, &window, |frame, channel, value| {
                buffers
                    .res2net
                    .set(frame * width + offset + channel, value.max(0.0));
            });
        }
        let mixed = Part {
            values: summed,
            stride: width,
            offset: 0,
        };
        let window = pass.window(std::slice::from_ref(&mixed), (1, 1), width);
        pass.convolve(&block.tdnn2, &window, |frame, channel, value| {
            buffers.tdnn2.set(frame * width + channel, value.max(0.0));
        });
        self.excite(pass, (index, block), previous);
    }

    /// The squeeze-excitation of the block's output, scaled, plus the block's input `previous`,
    /// into the block's columns of `aggregated`.
    fn excite(&self, pass: &Pass, (index, block): (usize, &Block), previous: Part) {
        let buffers = pass.buffers;
        let width = self.shape.width();
        let mixed = buffers.tdnn2.read();
        pass.time_mean(mixed, width, &buffers.mean);
        let mean = buffers.mean.read();
        pass.project_row(&block.squeeze, mean, |channel, value| {
            buffers.squeezed.set(channel, value.max(0.0));
        });
        pass.project_row(&block.excite, buffers.squeezed.read(), |channel, value| {
            buffers
                .excitation
                .set(channel, 1.0 / (1.0 + (-value).exp()));
        });
        let stride = self.shape.aggregated();
        let offset = (index - 1) * width;
        let excitation = buffers.excitation.read();
        for frame in pass.ctx.split(pass.frames, 1) {
            let skip = &previous.values[frame * previous.stride + previous.offset..][..width];
            let out = buffers
                .aggregated
                .part(frame * stride + offset..frame * stride + offset + width);
            let row = &mixed[frame * width..(frame + 1) * width];
            for (channel, value) in out.iter_mut().enumerate() {
                *value = row[channel] * excitation[channel] + skip[channel];
            }
        }
        pass.ctx.barrier();
    }

    /// The aggregation conv, the attentive statistics pooling and the embedding.
    fn pool(&self, pass: &Pass) {
        let buffers = pass.buffers;
        let shape = &self.shape;
        let pooled = shape.pooled();
        let aggregated = Part {
            values: buffers.aggregated.read(),
            stride: shape.aggregated(),
            offset: 0,
        };
        let last = shape.channels.len() - 1;
        let taps = (shape.kernels[last], shape.dilations[last]);
        let window = pass.window(std::slice::from_ref(&aggregated), taps, shape.aggregated());
        pass.convolve(&self.mfa, &window, |frame, channel, value| {
            buffers
                .features
                .set(frame * pooled + channel, value.max(0.0));
        });
        let features = buffers.features.read();
        self.moments(pass, features);
        self.attend(pass, features);
        self.weighted(pass, features);
        let embed = |channel: usize, value: f32| buffers.embedding.set(channel, value);
        pass.project_row(&self.out, buffers.pooled.read(), embed);
    }

    /// The mean and deviation of every channel of `features` over time, into `moments`.
    fn moments(&self, pass: &Pass, features: &[f32]) {
        let pooled = self.shape.pooled();
        let frames = pass.frames as f32;
        for channel in pass.ctx.split(pooled, 1) {
            let mean = column(features, channel, pooled).sum::<f32>() / frames;
            let squares = column(features, channel, pooled);
            let variance = squares.map(|value| (value - mean).powi(2)).sum::<f32>() / frames;
            pass.buffers.moments.set(channel, mean);
            let deviation = variance.max(VARIANCE_FLOOR).sqrt();
            pass.buffers.moments.set(pooled + channel, deviation);
        }
        pass.ctx.barrier();
    }

    /// The attention of every frame and channel: a ReLU and tanh conv over each frame's channels
    /// and the whole utterance's moments, then a conv back to the channels.
    fn attend(&self, pass: &Pass, features: &[f32]) {
        let buffers = pass.buffers;
        let attention = self.shape.attention;
        let pooled = self.shape.pooled();
        let moments = buffers.moments.read();
        let gather = |frame: usize, out: &mut [f32]| {
            let (frame_values, statistics) = out.split_at_mut(pooled);
            frame_values.copy_from_slice(&features[frame * pooled..][..pooled]);
            statistics.copy_from_slice(moments);
        };
        pass.product(&self.attention, gather, |frame, channel, value| {
            buffers
                .attended
                .set(frame * attention + channel, value.max(0.0).tanh());
        });
        let attended = Part {
            values: buffers.attended.read(),
            stride: attention,
            offset: 0,
        };
        let window = pass.window(std::slice::from_ref(&attended), (1, 1), attention);
        pass.convolve(&self.score, &window, |frame, channel, value| {
            buffers.scores.set(frame * pooled + channel, value);
        });
    }

    /// The mean and deviation of every channel of `features` weighted by the softmax over time
    /// of its scores, into `pooled`.
    fn weighted(&self, pass: &Pass, features: &[f32]) {
        let pooled = self.shape.pooled();
        let scores = pass.buffers.scores.read();
        for channel in pass.ctx.split(pooled, 1) {
            let peak = column(scores, channel, pooled).fold(f32::NEG_INFINITY, f32::max);
            let weights = || column(scores, channel, pooled).map(|score| (score - peak).exp());
            let total: f32 = weights().sum();
            let values = weights().zip(column(features, channel, pooled));
            let mean = values.map(|(weight, value)| weight * value).sum::<f32>() / total;
            let values = weights().zip(column(features, channel, pooled));
            let spread = values.map(|(weight, value)| weight * (value - mean).powi(2));
            let variance = spread.sum::<f32>() / total;
            pass.buffers.pooled.set(channel, mean);
            let deviation = variance.max(VARIANCE_FLOOR).sqrt();
            pass.buffers.pooled.set(pooled + channel, deviation);
        }
        pass.ctx.barrier();
    }
}
