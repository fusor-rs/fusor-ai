//! The FastConformer subsampler (and LFM2-Audio's): `stages` convolutions over time and
//! frequency, each a `kernel`-wide square window at stride `stride` with zero padding of half the
//! window, the shape's activation after each stage. The first maps the mel features to
//! `channels`; the others are depthwise, each followed by a pointwise linear. A last linear maps
//! each frame's remaining frequencies of `channels` to `hidden`.
//!
//! A stage of `n` input rows (or frequencies) has `(n + 2 · ⌊kernel / 2⌋ − kernel) / stride + 1`
//! outputs. As in the reference, the features count `padding_frames` zero frames past the
//! `mel_frames` real ones, and each stage zeroes its output rows past its valid length (the same
//! count of its input's valid rows) before its pointwise linear and activation. So `forward`
//! yields `Frames::all` rows, of which the first `Frames::valid` are the encoder's frames; the
//! rows after them see the pointwise biases.
//!
//! `forward` runs inside a team run: each thread computes blocks of output frames (the buffers'
//! `block_frames` at a time) through every stage, from the rows each block needs of the stage
//! before it, then the last linear splits by output columns.
//!
//! Tensors under a prefix `p`: `p.conv` and `p.depthwise{i}` for `i` from 1, each a `weight`
//! (`[kernel², channels]`, taps row-major over time then frequency) and a `bias`
//! (`[channels]`); the linears `p.pointwise{i}` and `p.out` (its inputs frequency-major,
//! `f · channels + c`).

mod buffers;
mod kernel;
mod shape;

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::{Linear, QuantizedRows, SharedRows, project};
use team::Ctx;

pub use buffers::Buffers;
use buffers::Scratch;
use kernel::{Kernel, Plane};
use shape::lengths;
pub use shape::{Activation, Frames, Shape};

/// A depthwise convolution and the pointwise linear after it.
struct Stage {
    depthwise: Kernel,
    pointwise: Linear,
}

pub struct Subsampling {
    shape: Shape,
    /// The frequencies after every stage, the first the mel bins.
    widths: Vec<usize>,
    conv: Kernel,
    stages: Vec<Stage>,
    out: Linear,
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
        let lengths = lengths(&self.shape, mel_frames);
        let frames = *lengths.last().expect("a subsampler has stages");
        let width = self.out.matrix().cols;
        for block in ctx.split(frames.all.div_ceil(buffers.block_frames), 1) {
            let first = block * buffers.block_frames;
            let rows = first..(first + buffers.block_frames).min(frames.all);
            let stacked = self.block(features, &lengths, rows.clone(), buffers.scratch.get(ctx));
            let part = buffers.stacked.part(rows.start * width..rows.end * width);
            part.copy_from_slice(stacked);
        }
        ctx.barrier();
        let stacked = SharedRows {
            values: &buffers.stacked,
            rows: frames.all,
        };
        let quant = &mut buffers.scratch.get(ctx).quant;
        project(ctx, &self.out, stacked, quant, |at, value| {
            buffers.hidden.set(at, value);
        });
        ctx.barrier();
        frames
    }

    /// The last stage's output rows `rows`, through the rows of each stage before it that they
    /// read.
    fn block<'scratch>(
        &self,
        features: &[f32],
        lengths: &[Frames],
        rows: Range<usize>,
        scratch: &'scratch mut Scratch,
    ) -> &'scratch [f32] {
        let widths = &self.widths;
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
        self.shape.activation.apply(&mut stages[0]);
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
                self.shape.activation,
                &mut rest[0],
                pointwise,
                quant,
            );
        }
        stages.last().expect("a subsampler has stages")
    }
}

/// `activation(linear(values))` over every position (a row of the linear's inputs), in place.
fn mix(
    linear: &Linear,
    activation: Activation,
    values: &mut [f32],
    out: &mut Vec<f32>,
    quant: &mut QuantizedRows,
) {
    out.resize(values.len(), 0.0);
    linear.apply(values, quant, out);
    values.copy_from_slice(out);
    activation.apply(values);
}

impl Load for Subsampling {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        let channels = [shape.channels; 2];
        let stages = (1..shape.stages)
            .map(|stage| {
                Ok(Stage {
                    depthwise: scope.load(&format!("depthwise{stage}"), shape)?,
                    pointwise: scope.load(&format!("pointwise{stage}"), channels)?,
                })
            })
            .collect::<Result<_, bundle::Error>>()?;
        Ok(Self {
            shape,
            widths: shape.widths(),
            conv: scope.load("conv", shape)?,
            stages,
            out: scope.load("out", [shape.hidden, shape.stacked_width()])?,
        })
    }
}
