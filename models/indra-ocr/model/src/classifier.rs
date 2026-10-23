//! The line classifiers: which recognizer reads a line (`script`) and whether it holds math
//! (`mathcls`). A line crop averaged 2 × 2, four 3 × 3 convolutions (ReLU) each followed by a
//! 2 × 2 max pool, the mean over the steps its width covers (`width / 32`, at least one), and a
//! linear head.
//!
//! Tensors under `script.m.` and `mathcls.m.`: the convolutions `body.{1, 5, 9, 13}`, `head`.

use bundle::{Load, Store};
use kernels::conv::{self, Conv2d, Grid};
use kernels::{Linear, QuantizedRows};
use team::{SharedBuf, Team};

use crate::Error;
use crate::plane::{Layers, Plane, Scratch};
use crate::recognizer::Batch;

/// Each convolution's inputs, outputs and name under `m.body`.
const CONVS: [(usize, usize, usize); 4] = [(1, 16, 1), (16, 32, 5), (32, 48, 9), (48, 64, 13)];
const KERNEL: usize = 3;
const POOL: usize = 2;
/// Input columns per output step: the average pool's and every max pool's.
const REDUCTION: usize = 32;

pub struct Classifier {
    convs: Vec<Conv2d>,
    head: Linear,
}

#[derive(Clone, Copy)]
enum Pool {
    Average,
    Max,
}

impl Classifier {
    /// The head's logits for each line of `batch`, `lines × classes`.
    pub fn forward(&self, team: &Team, scratch: &Scratch, batch: Batch) -> Vec<f32> {
        let grid = Grid {
            images: batch.lines(),
            height: batch.height,
            width: batch.width,
        };
        let mut input = SharedBuf::new(batch.pixels.len());
        input.get_mut().copy_from_slice(batch.pixels);
        let first = pooled(grid);
        let buffers = [0, 1].map(|_| SharedBuf::new(first.pixels() * CONVS[3].1));
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            let mut plane = Plane {
                values: &input,
                grid,
                channels: 1,
            };
            plane = pool(&layers, plane, &buffers[0], Pool::Average);
            for conv in &self.convs {
                let convolved = layers.conv(conv, plane, &buffers[1], |_, _, value| value.max(0.0));
                plane = pool(&layers, convolved, &buffers[0], Pool::Max);
            }
        });
        let last = self.convs.iter().fold(first, |grid, _| pooled(grid));
        let channels = self.head.matrix().cols;
        let means = masked_means(
            &buffers[0].read()[..last.pixels() * channels],
            last,
            batch.widths,
        );
        let mut logits = vec![0.0; batch.lines() * self.head.matrix().rows];
        self.head
            .apply(&means, &mut QuantizedRows::default(), &mut logits);
        logits
    }
}

/// The grid after a pool.
fn pooled(grid: Grid) -> Grid {
    Grid {
        height: grid.height / POOL,
        width: grid.width / POOL,
        ..grid
    }
}

/// `input` pooled over `POOL × POOL` windows at stride `POOL`, edges past a whole window
/// dropped.
fn pool<'out>(layers: &Layers, input: Plane, into: &'out SharedBuf, kind: Pool) -> Plane<'out> {
    let grid = pooled(input.grid);
    let channels = input.channels;
    for pixel in layers.ctx.split(grid.pixels(), 1) {
        let (image, row, column) = (
            pixel / grid.area(),
            pixel / grid.width % grid.height,
            pixel % grid.width,
        );
        let source = |dy: usize, dx: usize| {
            (image * input.grid.height + row * POOL + dy) * input.grid.width + column * POOL + dx
        };
        for channel in 0..channels {
            let [top_left, top_right, bottom_left, bottom_right] =
                [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(dy, dx)| input.at(source(dy, dx), channel));
            let value = match kind {
                Pool::Average => {
                    (top_left + top_right + bottom_left + bottom_right) / (POOL * POOL) as f32
                }
                Pool::Max => top_left.max(top_right).max(bottom_left).max(bottom_right),
            };
            into.set(pixel * channels + channel, value);
        }
    }
    layers.ctx.barrier();
    Plane {
        values: into,
        grid,
        channels,
    }
}

/// Each line's mean over the steps its width covers, `lines × channels`.
fn masked_means(values: &[f32], grid: Grid, widths: &[usize]) -> Vec<f32> {
    let channels = values.len() / grid.pixels();
    let mut means = vec![0.0; grid.images * channels];
    for (line, (mean, &width)) in means.chunks_exact_mut(channels).zip(widths).enumerate() {
        let steps = (width / REDUCTION).max(1).min(grid.width);
        for step in 0..steps {
            let row = &values[(line * grid.width + step) * channels..][..channels];
            for (sum, value) in mean.iter_mut().zip(row) {
                *sum += value;
            }
        }
        mean.iter_mut().for_each(|sum| *sum /= steps as f32);
    }
    means
}

impl Load for Classifier {
    /// The head's classes.
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, classes: usize) -> Result<Self, Error> {
        let convs = CONVS.map(|(inputs, outputs, index)| {
            let shape = conv::Shape::same(inputs, outputs, KERNEL);
            Conv2d::load(store, &format!("{prefix}.body.{index}"), shape)
        });
        Ok(Self {
            convs: convs.into_iter().collect::<Result<_, _>>()?,
            head: Linear::load(store, &format!("{prefix}.head"), [classes, CONVS[3].1])?,
        })
    }
}
