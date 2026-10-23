//! The blocks of the line recognizers and the math encoder, over a batch of padded sequences
//! (`rows = sequences × steps`, `channels` each, only a sequence's first `length` steps valid):
//! `x = (x + project(e · se(e))) · valid`, where `e = hardswish(expand(depthwise(x)))` and
//! `se(e) = σ(fc2(relu(fc1(mean e))))` per sequence, the mean over its valid steps.
//!
//! Tensors under a block's prefix `p`: `p.dw` (`[channels, kernel]`), `p.expand`, `p.se.fc1`,
//! `p.se.fc2` and `p.project`.

use bundle::{Load, Store};
use kernels::Linear;
use kernels::conv::{Depthwise2d, DepthwiseShape, Grid, project_rows};
use kernels::ops::{hard_swish, sigmoid};
use team::SharedBuf;

use crate::Error;
use crate::plane::{Layers, Plane};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockShape {
    pub(crate) channels: usize,
    pub(crate) expanded: usize,
    /// The squeeze-excitation's width.
    pub(crate) reduced: usize,
    pub(crate) kernel: usize,
}

pub(crate) struct Block {
    depthwise: Depthwise2d,
    expand: Linear,
    reduce: Linear,
    restore: Linear,
    project: Linear,
}

/// A batch of padded sequences: `grid.images` sequences of `grid.width` steps, each valid up
/// to its length.
#[derive(Clone, Copy)]
pub(crate) struct Sequences<'job> {
    pub(crate) grid: Grid,
    pub(crate) lengths: &'job [usize],
}

impl Sequences<'_> {
    pub(crate) fn rows(&self) -> usize {
        self.grid.pixels()
    }

    pub(crate) fn valid(&self, row: usize) -> bool {
        row % self.grid.width < self.lengths[row / self.grid.width]
    }
}

/// The inner activations of a block.
pub(crate) struct BlockBuffers {
    depthwise: SharedBuf,
    expanded: SharedBuf,
    means: SharedBuf,
    reduced: SharedBuf,
    gates: SharedBuf,
}

impl BlockBuffers {
    pub(crate) fn new(shape: BlockShape, sequences: usize, rows: usize) -> Self {
        Self {
            depthwise: SharedBuf::new(rows * shape.channels),
            expanded: SharedBuf::new(rows * shape.expanded),
            means: SharedBuf::new(sequences * shape.expanded),
            reduced: SharedBuf::new(sequences * shape.reduced),
            gates: SharedBuf::new(sequences * shape.expanded),
        }
    }
}

impl Block {
    /// The block over `x` (`sequences.rows() × channels`, in and out).
    pub(crate) fn forward(
        &self,
        layers: &Layers,
        x: &SharedBuf,
        sequences: Sequences,
        buffers: &BlockBuffers,
    ) {
        let (ctx, patches) = (layers.ctx, layers.patches.get(layers.ctx));
        let rows = sequences.rows();
        let channels = self.depthwise.channels();
        let input = Plane {
            values: x,
            grid: sequences.grid,
            channels,
        };
        let mixed = layers.depthwise(&self.depthwise, input, &buffers.depthwise, |_, _, value| {
            value
        });
        let expanded = self.expand.matrix().rows;
        project_rows(
            ctx,
            &self.expand,
            (mixed.values.read(), rows),
            patches,
            |row, channel, value| {
                buffers
                    .expanded
                    .set(row * expanded + channel, hard_swish(value));
            },
        );
        ctx.barrier();
        self.squeeze(layers, sequences, buffers);
        let gated = buffers.expanded.read();
        project_rows(
            ctx,
            &self.project,
            (gated, rows),
            patches,
            |row, channel, value| {
                let at = row * channels + channel;
                let kept = if sequences.valid(row) {
                    x.get(at) + value
                } else {
                    0.0
                };
                x.set(at, kept);
            },
        );
        ctx.barrier();
    }

    /// Each sequence's gates from the mean of its valid expanded steps, multiplied into them.
    fn squeeze(&self, layers: &Layers, sequences: Sequences, buffers: &BlockBuffers) {
        let (ctx, patches) = (layers.ctx, layers.patches.get(layers.ctx));
        let expanded = self.expand.matrix().rows;
        let count = sequences.grid.images;
        let steps = sequences.grid.width;
        for at in ctx.split(count * expanded, 1) {
            let (sequence, channel) = (at / expanded, at % expanded);
            let length = sequences.lengths[sequence];
            let rows = sequence * steps..sequence * steps + length;
            let sum: f32 = rows
                .map(|row| buffers.expanded.get(row * expanded + channel))
                .sum();
            buffers.means.set(at, sum / length.max(1) as f32);
        }
        ctx.barrier();
        let reduced = self.reduce.matrix().rows;
        project_rows(
            ctx,
            &self.reduce,
            (buffers.means.read(), count),
            patches,
            |row, channel, value| {
                buffers.reduced.set(row * reduced + channel, value.max(0.0));
            },
        );
        ctx.barrier();
        project_rows(
            ctx,
            &self.restore,
            (buffers.reduced.read(), count),
            patches,
            |row, channel, value| {
                buffers.gates.set(row * expanded + channel, sigmoid(value));
            },
        );
        ctx.barrier();
        for row in ctx.split(sequences.rows(), 1) {
            let gates = &buffers.gates.read()[row / steps * expanded..][..expanded];
            let values = buffers.expanded.part(row * expanded..(row + 1) * expanded);
            for (value, gate) in values.iter_mut().zip(gates) {
                *value *= gate;
            }
        }
        ctx.barrier();
    }
}

impl Load for Block {
    type Shape = BlockShape;

    fn load(store: &mut Store, prefix: &str, shape: BlockShape) -> Result<Self, Error> {
        let BlockShape {
            channels,
            expanded,
            reduced,
            kernel,
        } = shape;
        let window = DepthwiseShape {
            channels,
            window: [1, kernel],
            stride: 1,
        };
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            depthwise: Depthwise2d::load(store, &name("dw"), window)?,
            expand: Linear::load(store, &name("expand"), [expanded, channels])?,
            reduce: Linear::load(store, &name("se.fc1"), [reduced, expanded])?,
            restore: Linear::load(store, &name("se.fc2"), [expanded, reduced])?,
            project: Linear::load(store, &name("project"), [channels, expanded])?,
        })
    }
}
