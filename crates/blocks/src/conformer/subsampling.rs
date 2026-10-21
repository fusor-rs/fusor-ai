//! The FastConformer subsampler: `subsampling_stages` convolutions over time and frequency, each
//! a `subsampling_kernel`-wide square window at stride `subsampling_stride` with zero padding of
//! half the window, a ReLU after each stage. The first maps the mel features to `channels`; the
//! others are depthwise, each followed by a pointwise linear. A last linear maps each frame's
//! remaining frequencies of `channels` to `hidden`.
//!
//! A stage of `n` input rows (or frequencies) has `(n + 2 · ⌊kernel / 2⌋ − kernel) / stride + 1`
//! outputs. As in the reference, the features count one zero frame past the `mel_frames` real
//! ones, and each stage zeroes its output rows past its valid length (the same count of its
//! input's valid rows) before its pointwise linear and ReLU. So `forward` yields `Frames::all`
//! rows, of which the first `Frames::valid` are the encoder's frames; the rows after them see the
//! pointwise biases.
//!
//! `forward` runs inside a team run: each thread computes blocks of output frames (the buffers'
//! `block_frames` at a time) through every stage, from the rows each block needs of the stage
//! before it, then the last linear splits by output columns.
//!
//! Tensors under a prefix `p`: `p.conv` and `p.depthwise{i}` for `i` from 1, each a `weight`
//! (`[kernel², channels]`, taps row-major over time then frequency) and a `bias`
//! (`[channels]`); the linears `p.pointwise{i}` and `p.out` (its inputs frequency-major,
//! `f · channels + c`).

use std::ops::Range;

use bundle::{Load, Store};
use kernels::{COLUMN_ALIGN, Error, Linear, QuantizedRows, Tensor, matmul};
use team::{Ctx, PerThread, SharedBuf};

use super::{Shape, vector};

impl Shape {
    /// Mel frames per encoder frame.
    pub const fn subsampling_factor(&self) -> usize {
        self.subsampling_stride.pow(self.subsampling_stages as u32)
    }

    /// Outputs of a stage over `inputs` rows (or frequencies).
    fn stage_outputs(&self, inputs: usize) -> usize {
        let padding = self.subsampling_kernel / 2;
        (inputs + 2 * padding - self.subsampling_kernel) / self.subsampling_stride + 1
    }

    /// The input rows a stage reads for its output rows `rows`, within `available` input rows.
    fn reads(&self, rows: &Range<usize>, available: usize) -> Range<usize> {
        let padding = self.subsampling_kernel / 2;
        let first = (self.subsampling_stride * rows.start).saturating_sub(padding);
        let last = self.subsampling_stride * (rows.end - 1) + self.subsampling_kernel - padding;
        first..last.min(available)
    }
}

/// A stage's convolution: taps and bias.
struct Kernel {
    taps: Tensor,
    bias: Tensor,
}

/// A depthwise convolution and the pointwise linear after it.
struct Stage {
    depthwise: Kernel,
    pointwise: Linear,
}

pub struct Subsampling {
    shape: Shape,
    conv: Kernel,
    stages: Vec<Stage>,
    out: Linear,
}

/// The encoder frames of `mel_frames` mel frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frames {
    /// Rows of the subsampler's output.
    pub all: usize,
    /// The leading rows that are the encoder's frames.
    pub valid: usize,
}

impl Frames {
    pub fn of(shape: &Shape, mel_frames: usize) -> Self {
        let last = *lengths(shape, mel_frames)
            .last()
            .expect("at least one stage");
        Self {
            all: last.all,
            valid: last.valid,
        }
    }
}

/// Activations of one `forward` over up to `mel_frames` mel frames, each thread computing
/// `block_frames` output frames at a time (its working set).
pub struct Buffers {
    mel_frames: usize,
    block_frames: usize,
    /// The last stage's rows, `frames × frequencies · channels`.
    stacked: SharedBuf,
    /// The subsampled frames, `frames × hidden`.
    pub hidden: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    /// The rows of each stage a block reads, and computes.
    reads: Vec<Range<usize>>,
    stages: Vec<Vec<f32>>,
    pointwise: Vec<f32>,
    quant: QuantizedRows,
}

impl Buffers {
    pub fn new(shape: Shape, mel_frames: usize, block_frames: usize, threads: usize) -> Self {
        let frames = Frames::of(&shape, mel_frames).all;
        let width = stacked_width(&shape);
        Self {
            mel_frames,
            block_frames,
            stacked: SharedBuf::new(frames * width),
            hidden: SharedBuf::new(frames * shape.hidden),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    pub fn mel_frames(&self) -> usize {
        self.mel_frames
    }
}

/// A stage's output rows and the leading ones that are valid.
#[derive(Clone, Copy)]
struct Length {
    all: usize,
    valid: usize,
}

/// One call's stages: their output lengths, and the frequencies of their inputs (the mel bins
/// first) and of the last one's output.
struct Layout {
    lengths: Vec<Length>,
    widths: Vec<usize>,
}

/// Every stage's output length for `mel_frames` mel frames.
fn lengths(shape: &Shape, mel_frames: usize) -> Vec<Length> {
    let mut length = Length {
        all: mel_frames + 1,
        valid: mel_frames,
    };
    let stages = (0..shape.subsampling_stages).map(|_| {
        length = Length {
            all: shape.stage_outputs(length.all),
            valid: shape.stage_outputs(length.valid),
        };
        length
    });
    stages.collect()
}

/// The frequencies after every stage, the first the mel bins.
fn widths(shape: &Shape) -> Vec<usize> {
    let mut width = shape.mels;
    let mut widths = vec![width];
    for _ in 0..shape.subsampling_stages {
        width = shape.stage_outputs(width);
        widths.push(width);
    }
    widths
}

/// Values of one subsampled frame before the last linear.
fn stacked_width(shape: &Shape) -> usize {
    widths(shape).last().expect("the mel bins") * shape.channels
}

/// Rows of a stage's input: `values` holds rows `first..` of `width × depth` values each; rows
/// outside it are zeros. Depth 1 is the features (every channel reads the one value).
struct Plane<'values> {
    values: &'values [f32],
    first: usize,
    width: usize,
    depth: usize,
}

impl Plane<'_> {
    /// The values at `(row, column)`, or `None` for the zeros around the plane (`None`
    /// coordinates are the padding before it).
    fn at(&self, row: Option<usize>, column: Option<usize>) -> Option<&[f32]> {
        let (row, column) = (row?, column?);
        let rows = self.values.len() / (self.width * self.depth);
        if !(self.first..self.first + rows).contains(&row) || column >= self.width {
            return None;
        }
        let start = ((row - self.first) * self.width + column) * self.depth;
        Some(&self.values[start..start + self.depth])
    }
}

impl Subsampling {
    /// `hidden[..frames.all]` from `features`, `mel_frames × mels` row-major.
    ///
    /// # Panics
    /// When there are no features or more than the buffers hold.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, features: &[f32]) -> Frames {
        let mel_frames = features.len() / self.shape.mels;
        assert!(
            (1..=buffers.mel_frames).contains(&mel_frames),
            "the buffers hold the features"
        );
        let layout = Layout {
            lengths: lengths(&self.shape, mel_frames),
            widths: widths(&self.shape),
        };
        let frames = Frames::of(&self.shape, mel_frames);
        let width = stacked_width(&self.shape);
        for block in ctx.split(frames.all.div_ceil(buffers.block_frames), 1) {
            let first = block * buffers.block_frames;
            let rows = first..(first + buffers.block_frames).min(frames.all);
            let stacked = self.block(features, &layout, rows.clone(), buffers.scratch.get(ctx));
            let part = buffers.stacked.part(rows.start * width..rows.end * width);
            part.copy_from_slice(stacked);
        }
        ctx.barrier();
        let matrix = self.out.matrix();
        let quant = &mut buffers.scratch.get(ctx).quant;
        let stacked = &buffers.stacked.read()[..frames.all * matrix.cols];
        let x = quant.input(&matrix, stacked, frames.all);
        let bias = self.out.bias();
        let cols = ctx.split(matrix.rows, COLUMN_ALIGN);
        matmul(&matrix, &x, frames.all, cols, |row, col, value| {
            let biased = value + bias.map_or(0.0, |bias| bias[col]);
            buffers.hidden.set(row * matrix.rows + col, biased);
        });
        ctx.barrier();
        frames
    }

    /// The last stage's output rows `rows`, through the rows of each stage before it that they
    /// read.
    fn block<'scratch>(
        &self,
        features: &[f32],
        layout: &Layout,
        rows: Range<usize>,
        scratch: &'scratch mut Scratch,
    ) -> &'scratch [f32] {
        let Layout { lengths, widths } = layout;
        let Scratch {
            reads,
            stages,
            pointwise,
            quant,
        } = scratch;
        reads.clear();
        reads.resize(lengths.len(), rows);
        for stage in (0..lengths.len() - 1).rev() {
            reads[stage] = self.shape.reads(&reads[stage + 1], lengths[stage].all);
        }
        stages.resize_with(lengths.len(), Vec::new);
        let features = Plane {
            values: features,
            first: 0,
            width: self.shape.mels,
            depth: 1,
        };
        let valid = lengths[0].valid;
        self.conv.apply(
            &self.shape,
            &features,
            (reads[0].clone(), valid),
            &mut stages[0],
        );
        relu(&mut stages[0]);
        for (index, stage) in self.stages.iter().enumerate() {
            let (done, rest) = stages.split_at_mut(index + 1);
            let input = Plane {
                values: &done[index],
                first: reads[index].start,
                width: widths[index + 1],
                depth: self.shape.channels,
            };
            let span = (reads[index + 1].clone(), lengths[index + 1].valid);
            stage
                .depthwise
                .apply(&self.shape, &input, span, &mut rest[0]);
            mix(
                &stage.pointwise,
                self.shape.channels,
                &mut rest[0],
                (pointwise, quant),
            );
        }
        stages.last().expect("at least one stage")
    }
}

/// `relu(linear(values))` over every position of `channels` values, in place.
fn mix(
    linear: &Linear,
    channels: usize,
    values: &mut [f32],
    (out, quant): (&mut Vec<f32>, &mut QuantizedRows),
) {
    let positions = values.len() / channels;
    out.resize(values.len(), 0.0);
    linear.apply(values, positions, quant, out);
    values.copy_from_slice(out);
    relu(values);
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

impl Kernel {
    /// Output rows `rows` of this convolution of `input` into `out`, `rows × outputs(width) ×
    /// channels`; rows from `valid` on are zeros.
    fn apply(
        &self,
        shape: &Shape,
        input: &Plane,
        (rows, valid): (Range<usize>, usize),
        out: &mut Vec<f32>,
    ) {
        let channels = self.bias.values().expect("checked at load").len();
        let width = shape.stage_outputs(input.width);
        out.clear();
        out.resize(rows.len() * width * channels, 0.0);
        let outputs = out.chunks_exact_mut(width * channels);
        for (row, values) in rows.take_while(|&row| row < valid).zip(outputs) {
            for (column, sums) in values.chunks_exact_mut(channels).enumerate() {
                self.sum_at(shape, input, (row, column), sums);
            }
        }
    }

    /// The bias plus every tap's product at output `(row, column)`.
    fn sum_at(
        &self,
        shape: &Shape,
        input: &Plane,
        (row, column): (usize, usize),
        sums: &mut [f32],
    ) {
        let taps = self.taps.values().expect("checked at load");
        let (side, stride, padding) = (
            shape.subsampling_kernel,
            shape.subsampling_stride,
            shape.subsampling_kernel / 2,
        );
        let channels = sums.len();
        sums.copy_from_slice(self.bias.values().expect("checked at load"));
        for tap in 0..side * side {
            let source = (stride * row + tap / side).checked_sub(padding);
            let at = (stride * column + tap % side).checked_sub(padding);
            if let Some(x) = input.at(source, at) {
                accumulate(sums, &taps[tap * channels..(tap + 1) * channels], x);
            }
        }
    }
}

/// `sums += weights · x`, `x` one value for every channel or one per channel.
fn accumulate(sums: &mut [f32], weights: &[f32], x: &[f32]) {
    if let [value] = x {
        for (sum, weight) in sums.iter_mut().zip(weights) {
            *sum += weight * value;
        }
        return;
    }
    for ((sum, weight), value) in sums.iter_mut().zip(weights).zip(x) {
        *sum += weight * value;
    }
}

impl Load for Kernel {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let taps = shape.subsampling_kernel * shape.subsampling_kernel;
        Ok(Self {
            taps: vector(store, &format!("{prefix}.weight"), taps * shape.channels)?,
            bias: vector(store, &format!("{prefix}.bias"), shape.channels)?,
        })
    }
}

impl Load for Subsampling {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let stages = (1..shape.subsampling_stages)
            .map(|stage| {
                Ok(Stage {
                    depthwise: Kernel::load(store, &name(&format!("depthwise{stage}")), shape)?,
                    pointwise: Linear::load(store, &name(&format!("pointwise{stage}")), ())?,
                })
            })
            .collect::<Result<_, bundle::Error>>()?;
        let subsampling = Self {
            shape,
            conv: Kernel::load(store, &name("conv"), shape)?,
            stages,
            out: Linear::load(store, &name("out"), ())?,
        };
        let inputs = stacked_width(&shape);
        let found = subsampling.out.matrix().cols;
        if found != inputs {
            return Err(
                Error::Format(format!("{prefix}.out reads {found} values, not {inputs}")).into(),
            );
        }
        Ok(subsampling)
    }
}
