//! A CTC line recognizer: each 32 × 4 patch of the line band a step of 128 channels (the stem),
//! batch-normed, HardSwish, ten blocks (`sequence`) and a head over the vocabulary's classes,
//! reduced per step to what decoding needs: the best class, its probability (the reciprocal of
//! the softmax's sum), the blank's log-probability and the eight likeliest classes.
//!
//! The base recognizer reads a tall window, the line band with the page above and below it, and
//! adds a context to each step: 4 × 4 patches of the window to 32 channels (HardSwish), a 3 × 3
//! depthwise and a pointwise convolution to 64 (HardSwish), four groups of 16 channels each
//! averaged over the window's rows by a softmax of its own score plus a learned prior per row,
//! projected to the stem's width. The group recognizers read the band alone.
//!
//! Tensors under a recognizer's prefix `p` (`rec.rec` and the groups' `rec_<group>.rec`):
//! `p.stem`, `p.stem_bn.{scale, shift}`, `p.blocks.{i}`, `p.head`, and the base's
//! `p.context.{patch, dw, pw, score, row_prior, proj}`.

use bundle::{Load, Store, load_each};
use kernels::conv::{Depthwise2d, DepthwiseShape, Grid, project_rows};
use kernels::numpy::argmax;
use kernels::ops::{hard_swish, softmax};
use kernels::{Linear, Vector};
use team::{SharedBuf, Team};

use crate::Error;
use crate::plane::{Layers, Plane, Scratch};
use crate::sequence::{Block, BlockBuffers, BlockShape, Sequences};

/// A line band's height, and the window of the base recognizer in bands.
pub const LINE_HEIGHT: usize = 32;
/// Input columns per step.
pub const STEP: usize = 4;
/// Classes kept per step for the beam search.
pub const BEST: usize = 8;
const CHANNELS: usize = 128;
const BLOCKS: usize = 10;
const BLOCK: BlockShape = BlockShape {
    channels: CHANNELS,
    expanded: 256,
    reduced: 64,
    kernel: 9,
};
const PATCH: usize = LINE_HEIGHT * STEP;
const CONTEXT_PATCH: usize = 4;
const CONTEXT_NARROW: usize = 32;
const CONTEXT_WIDE: usize = 64;
const CONTEXT_KERNEL: usize = 3;
const CONTEXT_GROUPS: usize = 4;
/// Head rows reduced at a time: the logits of a chunk are `HEAD_ROWS × classes`.
const HEAD_ROWS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecognizerShape {
    pub classes: usize,
    /// The window's height in bands: 1, or 3 with the context.
    pub window: usize,
}

pub struct Recognizer {
    window: usize,
    stem: Linear,
    context: Option<Context>,
    scale: Vector,
    shift: Vector,
    blocks: Vec<Block>,
    head: Linear,
}

struct Context {
    patch: Linear,
    depthwise: Depthwise2d,
    pointwise: Linear,
    score: Linear,
    /// Per group, a prior per window row.
    prior: Vector,
    project: Linear,
}

/// Line crops padded to one width, each `height × width`, and their own widths.
#[derive(Clone, Copy)]
pub struct Batch<'job> {
    pub pixels: &'job [f32],
    pub widths: &'job [usize],
    pub height: usize,
    pub width: usize,
}

impl Batch<'_> {
    pub fn lines(&self) -> usize {
        self.widths.len()
    }
}

/// What decoding needs of each step of each line: `lines × steps` rows, the best `BEST`
/// classes per row by their log-probabilities, best first.
pub struct Outputs {
    pub steps: usize,
    pub lengths: Vec<usize>,
    pub ids: Vec<u32>,
    pub confidences: Vec<f32>,
    pub blanks: Vec<f32>,
    pub top_values: Vec<f32>,
    pub top_ids: Vec<u32>,
}

/// A chunk of the head's logits, and their reductions, one row per step.
struct Reduced {
    logits: SharedBuf,
    ids: SharedBuf<u32>,
    confidences: SharedBuf,
    blanks: SharedBuf,
    top_values: SharedBuf,
    top_ids: SharedBuf<u32>,
}

impl Recognizer {
    pub fn forward(&self, team: &Team, scratch: &Scratch, batch: Batch) -> Outputs {
        let steps = batch.width / STEP;
        let grid = Grid {
            images: batch.lines(),
            height: 1,
            width: steps,
        };
        let rows = grid.pixels();
        let lengths: Vec<usize> = batch
            .widths
            .iter()
            .map(|width| width.div_ceil(STEP).min(steps))
            .collect();
        let sequences = Sequences {
            grid,
            lengths: &lengths,
        };
        let gathered = SharedBuf::new(rows * PATCH);
        let x = SharedBuf::new(rows * CHANNELS);
        let blocks = BlockBuffers::new(BLOCK, batch.lines(), rows);
        let context = self
            .context
            .as_ref()
            .map(|_| ContextBuffers::new(batch, steps));
        let reduced = Reduced::new(rows, self.head.matrix().rows);
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            self.stem(&layers, batch, &gathered, &x);
            if let (Some(network), Some(buffers)) = (&self.context, &context) {
                network.forward(&layers, batch, buffers, &x);
            }
            self.normalize(&layers, sequences, &x);
            for block in &self.blocks {
                block.forward(&layers, &x, sequences, &blocks);
            }
            self.head(&layers, &x, rows, &reduced);
        });
        reduced.outputs(steps, lengths)
    }

    /// `x` = the stem's projection of each step's patch of the line band.
    fn stem(&self, layers: &Layers, batch: Batch, gathered: &SharedBuf, x: &SharedBuf) {
        let steps = batch.width / STEP;
        let top = (self.window - 1) / 2 * LINE_HEIGHT;
        for row in layers.ctx.split(batch.lines() * steps, 1) {
            let (line, step) = (row / steps, row % steps);
            let patch = gathered.part(row * PATCH..(row + 1) * PATCH);
            for (index, value) in patch.iter_mut().enumerate() {
                let (y, column) = (top + index / STEP, step * STEP + index % STEP);
                *value = batch.pixels[(line * batch.height + y) * batch.width + column];
            }
        }
        layers.ctx.barrier();
        let patches = layers.patches.get(layers.ctx);
        let rows = batch.lines() * steps;
        project_rows(
            layers.ctx,
            &self.stem,
            (gathered.read(), rows),
            patches,
            |row, channel, value| {
                x.set(row * CHANNELS + channel, value);
            },
        );
        layers.ctx.barrier();
    }

    /// The batch norm, the HardSwish and the mask, in place.
    fn normalize(&self, layers: &Layers, sequences: Sequences, x: &SharedBuf) {
        let (scale, shift) = (self.scale.values(), self.shift.values());
        for row in layers.ctx.split(sequences.rows(), 1) {
            let valid = sequences.valid(row);
            let values = x.part(row * CHANNELS..(row + 1) * CHANNELS);
            for ((value, scale), shift) in values.iter_mut().zip(scale).zip(shift) {
                *value = if valid {
                    hard_swish(*value * scale + shift)
                } else {
                    0.0
                };
            }
        }
        layers.ctx.barrier();
    }

    /// The head over `x`, `HEAD_ROWS` rows at a time, reduced per row.
    fn head(&self, layers: &Layers, x: &SharedBuf, rows: usize, reduced: &Reduced) {
        let logits = &reduced.logits;
        let classes = self.head.matrix().rows;
        let patches = layers.patches.get(layers.ctx);
        for first in (0..rows).step_by(HEAD_ROWS) {
            let count = HEAD_ROWS.min(rows - first);
            let input = &x.read()[first * CHANNELS..];
            project_rows(
                layers.ctx,
                &self.head,
                (input, count),
                patches,
                |row, class, value| {
                    logits.set(row * classes + class, value);
                },
            );
            layers.ctx.barrier();
            for row in layers.ctx.split(count, 1) {
                reduced.reduce(
                    first + row,
                    &logits.read()[row * classes..(row + 1) * classes],
                );
            }
            layers.ctx.barrier();
        }
    }
}

impl Reduced {
    fn new(rows: usize, classes: usize) -> Self {
        Self {
            logits: SharedBuf::new(HEAD_ROWS.min(rows) * classes),
            ids: SharedBuf::new(rows),
            confidences: SharedBuf::new(rows),
            blanks: SharedBuf::new(rows),
            top_values: SharedBuf::new(rows * BEST),
            top_ids: SharedBuf::new(rows * BEST),
        }
    }

    /// Row `row`'s best class, its probability, the blank's log-probability and the `BEST`
    /// likeliest classes, from its logits.
    fn reduce(&self, row: usize, logits: &[f32]) {
        let best = argmax(logits);
        let max = logits[best];
        let sum: f32 = logits.iter().map(|&value| (value - max).exp()).sum();
        let log_sum = sum.ln();
        self.ids.set(row, best as u32);
        self.confidences.set(row, 1.0 / sum);
        self.blanks.set(row, logits[0] - max - log_sum);
        let mut top = [(f32::NEG_INFINITY, 0u32); BEST];
        for (class, &value) in logits.iter().enumerate() {
            if value <= top[BEST - 1].0 {
                continue;
            }
            let at = top.partition_point(|&(kept, _)| kept >= value);
            top.copy_within(at..BEST - 1, at + 1);
            top[at] = (value, class as u32);
        }
        for (slot, (value, class)) in top.into_iter().enumerate() {
            self.top_values
                .set(row * BEST + slot, value - max - log_sum);
            self.top_ids.set(row * BEST + slot, class);
        }
    }

    fn outputs(self, steps: usize, lengths: Vec<usize>) -> Outputs {
        Outputs {
            steps,
            lengths,
            ids: self.ids.read().to_vec(),
            confidences: self.confidences.read().to_vec(),
            blanks: self.blanks.read().to_vec(),
            top_values: self.top_values.read().to_vec(),
            top_ids: self.top_ids.read().to_vec(),
        }
    }
}

/// The context's activations: `lines × rows × steps` window cells, `rows` the window's patch
/// rows.
struct ContextBuffers {
    rows: usize,
    gathered: SharedBuf,
    narrow: SharedBuf,
    mixed: SharedBuf,
    wide: SharedBuf,
    scores: SharedBuf,
    summed: SharedBuf,
}

impl ContextBuffers {
    fn new(batch: Batch, steps: usize) -> Self {
        let rows = batch.height / CONTEXT_PATCH;
        let cells = batch.lines() * rows * steps;
        Self {
            rows,
            gathered: SharedBuf::new(cells * CONTEXT_PATCH * CONTEXT_PATCH),
            narrow: SharedBuf::new(cells * CONTEXT_NARROW),
            mixed: SharedBuf::new(cells * CONTEXT_NARROW),
            wide: SharedBuf::new(cells * CONTEXT_WIDE),
            scores: SharedBuf::new(cells * CONTEXT_GROUPS),
            summed: SharedBuf::new(batch.lines() * steps * CONTEXT_WIDE),
        }
    }
}

impl Context {
    /// Adds each step's context to `x`.
    fn forward(&self, layers: &Layers, batch: Batch, buffers: &ContextBuffers, x: &SharedBuf) {
        let (ctx, patches) = (layers.ctx, layers.patches.get(layers.ctx));
        let steps = batch.width / STEP;
        let grid = Grid {
            images: batch.lines(),
            height: buffers.rows,
            width: steps,
        };
        self.cells(layers, batch, grid, buffers);
        let prior = self.prior.values();
        let cells = (buffers.wide.read(), grid.pixels());
        project_rows(ctx, &self.score, cells, patches, |cell, group, value| {
            let row = cell / steps % buffers.rows;
            let score = value + prior[group * buffers.rows + row];
            buffers.scores.set(cell * CONTEXT_GROUPS + group, score);
        });
        ctx.barrier();
        pool_rows(layers, grid, buffers);
        let rows = (buffers.summed.read(), batch.lines() * steps);
        project_rows(ctx, &self.project, rows, patches, |row, channel, value| {
            let at = row * CHANNELS + channel;
            x.set(at, x.get(at) + value);
        });
        ctx.barrier();
    }

    /// `buffers.wide`: each window cell's features, from its patch through the patch
    /// convolution, the depthwise and the pointwise one.
    fn cells(&self, layers: &Layers, batch: Batch, grid: Grid, buffers: &ContextBuffers) {
        let (ctx, patches) = (layers.ctx, layers.patches.get(layers.ctx));
        let cells = grid.pixels();
        gather_patches(layers, batch, grid, &buffers.gathered);
        let gathered = (buffers.gathered.read(), cells);
        project_rows(
            ctx,
            &self.patch,
            gathered,
            patches,
            |cell, channel, value| {
                buffers
                    .narrow
                    .set(cell * CONTEXT_NARROW + channel, hard_swish(value));
            },
        );
        ctx.barrier();
        let narrow = Plane {
            values: &buffers.narrow,
            grid,
            channels: CONTEXT_NARROW,
        };
        let mixed = layers.depthwise(&self.depthwise, narrow, &buffers.mixed, |_, _, value| value);
        project_rows(
            ctx,
            &self.pointwise,
            (mixed.values.read(), cells),
            patches,
            |cell, channel, value| {
                buffers
                    .wide
                    .set(cell * CONTEXT_WIDE + channel, hard_swish(value));
            },
        );
        ctx.barrier();
    }
}

/// Each window cell's `CONTEXT_PATCH × CONTEXT_PATCH` pixels, row-major.
fn gather_patches(layers: &Layers, batch: Batch, grid: Grid, into: &SharedBuf) {
    let size = CONTEXT_PATCH * CONTEXT_PATCH;
    for cell in layers.ctx.split(grid.pixels(), 1) {
        let (line, row, step) = (
            cell / grid.area(),
            cell / grid.width % grid.height,
            cell % grid.width,
        );
        let patch = into.part(cell * size..(cell + 1) * size);
        for (index, value) in patch.iter_mut().enumerate() {
            let y = row * CONTEXT_PATCH + index / CONTEXT_PATCH;
            let column = step * CONTEXT_PATCH + index % CONTEXT_PATCH;
            *value = batch.pixels[(line * batch.height + y) * batch.width + column];
        }
    }
    layers.ctx.barrier();
}

/// Per step, each group's channels averaged over the window's rows by the softmax of the
/// group's scores.
fn pool_rows(layers: &Layers, grid: Grid, buffers: &ContextBuffers) {
    let width = CONTEXT_WIDE / CONTEXT_GROUPS;
    let mut weights = vec![0.0; grid.height];
    for row in layers.ctx.split(grid.images * grid.width, 1) {
        let (line, step) = (row / grid.width, row % grid.width);
        let cell = |window_row: usize| (line * grid.height + window_row) * grid.width + step;
        let summed = buffers
            .summed
            .part(row * CONTEXT_WIDE..(row + 1) * CONTEXT_WIDE);
        for group in 0..CONTEXT_GROUPS {
            for (window_row, weight) in weights.iter_mut().enumerate() {
                *weight = buffers
                    .scores
                    .get(cell(window_row) * CONTEXT_GROUPS + group);
            }
            softmax(&mut weights);
            for (offset, out) in summed[group * width..(group + 1) * width]
                .iter_mut()
                .enumerate()
            {
                let channel = group * width + offset;
                *out = weights
                    .iter()
                    .enumerate()
                    .map(|(window_row, weight)| {
                        buffers.wide.get(cell(window_row) * CONTEXT_WIDE + channel) * weight
                    })
                    .sum();
            }
        }
    }
    layers.ctx.barrier();
}

impl Load for Recognizer {
    type Shape = RecognizerShape;

    fn load(store: &mut Store, prefix: &str, shape: RecognizerShape) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let blocks = load_each(store, &name("blocks"), BLOCKS, |store, block| {
            Block::load(store, block, BLOCK)
        })?;
        let context = if shape.window > 1 {
            Some(Context::load(
                store,
                &name("context"),
                shape.window * LINE_HEIGHT / CONTEXT_PATCH,
            )?)
        } else {
            None
        };
        Ok(Self {
            window: shape.window,
            stem: Linear::load(store, &name("stem"), [CHANNELS, PATCH])?,
            context,
            scale: Vector::load(store, &name("stem_bn.scale"), CHANNELS)?,
            shift: Vector::load(store, &name("stem_bn.shift"), CHANNELS)?,
            blocks,
            head: Linear::load(store, &name("head"), [shape.classes, CHANNELS])?,
        })
    }
}

impl Load for Context {
    /// The window's patch rows.
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, rows: usize) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let window = DepthwiseShape {
            channels: CONTEXT_NARROW,
            window: [CONTEXT_KERNEL, CONTEXT_KERNEL],
            stride: 1,
        };
        let patch = CONTEXT_PATCH * CONTEXT_PATCH;
        Ok(Self {
            patch: Linear::load(store, &name("patch"), [CONTEXT_NARROW, patch])?,
            depthwise: Depthwise2d::load(store, &name("dw"), window)?,
            pointwise: Linear::load(store, &name("pw"), [CONTEXT_WIDE, CONTEXT_NARROW])?,
            score: Linear::load(store, &name("score"), [CONTEXT_GROUPS, CONTEXT_WIDE])?,
            prior: Vector::load(store, &name("row_prior"), CONTEXT_GROUPS * rows)?,
            project: Linear::load(store, &name("proj"), [CHANNELS, CONTEXT_WIDE])?,
        })
    }
}
