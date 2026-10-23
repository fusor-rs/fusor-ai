//! The math encoder: three 3 × 3 convolutions at stride 2 (exact GELU), each step's remaining
//! rows folded into its channels (a batch norm, GELU), its valid steps (`⌈width/8⌉`) kept, six
//! blocks (`sequence`), two pre-norm attention layers whose scores fall by `2^−(h+1)` per step of
//! distance in head `h` (ALiBi), over the valid steps, and sinusoidal positions added.
//!
//! Tensors under `math_enc.enc.`: `front.{0, 3, 6}`, `fold`, `fold_bn.{scale, shift}`,
//! `blocks.{i}` and `attn.{i}.{ln1, qkv, out, ln2, ff1, ff2}`.

use blocks::norm::LayerNorm;
use bundle::{Load, Store, load_each};
use kernels::conv::{self, Conv2d, Grid, project_rows};
use kernels::ops::{gelu_erf, softmax};
use kernels::{Linear, Vector, dot_f32};
use team::{SharedBuf, Team};

use crate::Error;
use crate::plane::{Layers, Plane, Scratch};
use crate::recognizer::Batch;
use crate::sequence::{Block, BlockBuffers, BlockShape, Sequences};

/// The crop's height.
pub const HEIGHT: usize = 64;
pub(crate) const WIDTH: usize = 192;
/// The front's channels.
const FRONT: [usize; 3] = [32, 64, 96];
const FRONT_KERNEL: usize = 3;
/// Input columns per step.
const STEP: usize = 8;
const BLOCK: BlockShape = BlockShape {
    channels: WIDTH,
    expanded: 384,
    reduced: 96,
    kernel: 9,
};
const BLOCKS: usize = 6;
const LAYERS: usize = 2;
const HEADS: usize = 8;
const HEAD_DIM: usize = WIDTH / HEADS;
const FEED_FORWARD: usize = 768;
const EPS: f32 = 1e-5;
/// The sinusoidal positions' base.
const POSITION_BASE: f32 = 10_000.0;

pub struct Encoder {
    front: Vec<Conv2d>,
    fold: Linear,
    scale: Vector,
    shift: Vector,
    blocks: Vec<Block>,
    layers: Vec<Attention>,
}

struct Attention {
    norm1: LayerNorm,
    qkv: Linear,
    out: Linear,
    norm2: LayerNorm,
    up: Linear,
    down: Linear,
}

/// The encoder's output: `lines × steps` rows of `WIDTH`, each line valid up to its length.
pub struct Memory {
    pub steps: usize,
    pub lengths: Vec<usize>,
    pub values: Vec<f32>,
}

/// The attention layers' activations.
struct LayerBuffers {
    normed: SharedBuf,
    qkv: SharedBuf,
    attended: SharedBuf,
    inner: SharedBuf,
}

/// The encoder's activations over a batch.
struct EncoderBuffers {
    input: SharedBuf,
    fronts: Vec<SharedBuf>,
    gathered: SharedBuf,
    x: SharedBuf,
    blocks: BlockBuffers,
    layers: LayerBuffers,
}

impl EncoderBuffers {
    /// Buffers for `batch`, whose front's last output is `last`.
    fn new(encoder: &Encoder, batch: Batch, last: Grid) -> Self {
        let mut input = SharedBuf::new(batch.pixels.len());
        input.get_mut().copy_from_slice(batch.pixels);
        let mut grid = Grid {
            images: batch.lines(),
            height: batch.height,
            width: batch.width,
        };
        let fronts = encoder
            .front
            .iter()
            .zip(FRONT)
            .map(|(conv, channels)| {
                grid = conv.output_grid(grid);
                SharedBuf::new(grid.pixels() * channels)
            })
            .collect();
        let rows = batch.lines() * last.width;
        Self {
            input,
            fronts,
            gathered: SharedBuf::new(rows * last.height * FRONT[2]),
            x: SharedBuf::new(rows * WIDTH),
            blocks: BlockBuffers::new(BLOCK, batch.lines(), rows),
            layers: LayerBuffers {
                normed: SharedBuf::new(rows * WIDTH),
                qkv: SharedBuf::new(rows * 3 * WIDTH),
                attended: SharedBuf::new(rows * WIDTH),
                inner: SharedBuf::new(rows * FEED_FORWARD),
            },
        }
    }
}

impl Encoder {
    pub fn forward(&self, team: &Team, scratch: &Scratch, batch: Batch) -> Memory {
        let grid = Grid {
            images: batch.lines(),
            height: batch.height,
            width: batch.width,
        };
        let last = self
            .front
            .iter()
            .fold(grid, |grid, conv| conv.output_grid(grid));
        let steps = last.width;
        let lengths: Vec<usize> = batch
            .widths
            .iter()
            .map(|width| width.div_ceil(STEP).clamp(1, steps))
            .collect();
        let sequences = Sequences {
            grid: Grid {
                images: batch.lines(),
                height: 1,
                width: steps,
            },
            lengths: &lengths,
        };
        let buffers = EncoderBuffers::new(self, batch, last);
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            let mut plane = Plane {
                values: &buffers.input,
                grid,
                channels: 1,
            };
            for (conv, buffer) in self.front.iter().zip(&buffers.fronts) {
                plane = layers.conv(conv, plane, buffer, |_, _, value| gelu_erf(value));
            }
            self.fold(&layers, plane, sequences, &buffers);
            for block in &self.blocks {
                block.forward(&layers, &buffers.x, sequences, &buffers.blocks);
            }
            for layer in &self.layers {
                layer.forward(&layers, &buffers.x, sequences, &buffers.layers);
            }
        });
        let mut values = buffers.x.read().to_vec();
        add_positions(&mut values, steps);
        Memory {
            steps,
            lengths,
            values,
        }
    }

    /// Each step's column of the front's output (its rows' channels, row by row) projected,
    /// batch-normed and through GELU, zero past the line's length.
    fn fold(&self, layers: &Layers, front: Plane, sequences: Sequences, buffers: &EncoderBuffers) {
        let (gathered, x) = (&buffers.gathered, &buffers.x);
        let column = front.grid.height * front.channels;
        let steps = front.grid.width;
        for row in layers.ctx.split(sequences.rows(), 1) {
            let (line, step) = (row / steps, row % steps);
            let out = gathered.part(row * column..(row + 1) * column);
            for (index, value) in out.iter_mut().enumerate() {
                let pixel = (line * front.grid.height + index / front.channels) * steps + step;
                *value = front.at(pixel, index % front.channels);
            }
        }
        layers.ctx.barrier();
        let (scale, shift) = (self.scale.values(), self.shift.values());
        let patches = layers.patches.get(layers.ctx);
        project_rows(
            layers.ctx,
            &self.fold,
            (gathered.read(), sequences.rows()),
            patches,
            |row, channel, value| {
                let kept = if sequences.valid(row) {
                    gelu_erf(value * scale[channel] + shift[channel])
                } else {
                    0.0
                };
                x.set(row * WIDTH + channel, kept);
            },
        );
        layers.ctx.barrier();
    }
}

impl Attention {
    /// `x += out(attention(ln1 x))`, `x += ff(ln2 x)`, then zero past each line's length.
    fn forward(
        &self,
        layers: &Layers,
        x: &SharedBuf,
        sequences: Sequences,
        buffers: &LayerBuffers,
    ) {
        let (ctx, patches) = (layers.ctx, layers.patches.get(layers.ctx));
        let rows = sequences.rows();
        normalize(layers, &self.norm1, x, &buffers.normed, rows);
        project_rows(
            ctx,
            &self.qkv,
            (buffers.normed.read(), rows),
            patches,
            |row, column, value| {
                buffers.qkv.set(row * 3 * WIDTH + column, value);
            },
        );
        ctx.barrier();
        attend(layers, sequences, &buffers.qkv, &buffers.attended);
        project_rows(
            ctx,
            &self.out,
            (buffers.attended.read(), rows),
            patches,
            |row, channel, value| {
                let at = row * WIDTH + channel;
                x.set(at, x.get(at) + value);
            },
        );
        ctx.barrier();
        normalize(layers, &self.norm2, x, &buffers.normed, rows);
        project_rows(
            ctx,
            &self.up,
            (buffers.normed.read(), rows),
            patches,
            |row, column, value| {
                buffers
                    .inner
                    .set(row * FEED_FORWARD + column, gelu_erf(value));
            },
        );
        ctx.barrier();
        project_rows(
            ctx,
            &self.down,
            (buffers.inner.read(), rows),
            patches,
            |row, channel, value| {
                let at = row * WIDTH + channel;
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
}

/// `into` = the LayerNorm of each of `x`'s first `rows` rows.
pub(crate) fn normalize(
    layers: &Layers,
    norm: &LayerNorm,
    x: &SharedBuf,
    into: &SharedBuf,
    rows: usize,
) {
    for row in layers.ctx.split(rows, 1) {
        let input = &x.read()[row * WIDTH..(row + 1) * WIDTH];
        norm.apply(input, EPS, into.part(row * WIDTH..(row + 1) * WIDTH));
    }
    layers.ctx.barrier();
}

/// Each valid row's attention over its line's valid steps, head by head: `q · k / √d − slope ·
/// distance`, the slope of head `h` `2^−(h+1)`. Rows past a line's length stay zero.
fn attend(layers: &Layers, sequences: Sequences, qkv: &SharedBuf, into: &SharedBuf) {
    let steps = sequences.grid.width;
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let qkv = qkv.read();
    let mut scores = Vec::with_capacity(steps);
    for unit in layers.ctx.split(sequences.rows() * HEADS, 1) {
        let (row, head) = (unit / HEADS, unit % HEADS);
        let out = into.part(row * WIDTH + head * HEAD_DIM..row * WIDTH + (head + 1) * HEAD_DIM);
        out.fill(0.0);
        if !sequences.valid(row) {
            continue;
        }
        let (line, step) = (row / steps, row % steps);
        let slope = 0.5f32.powi(head as i32 + 1);
        let head_of = |row: usize, part: usize| {
            &qkv[row * 3 * WIDTH + part * WIDTH + head * HEAD_DIM..][..HEAD_DIM]
        };
        let query = head_of(row, 0);
        scores.clear();
        for key in 0..sequences.lengths[line] {
            let distance = step.abs_diff(key) as f32;
            scores.push(dot_f32(query, head_of(line * steps + key, 1)) * scale - slope * distance);
        }
        softmax(&mut scores);
        for (key, weight) in scores.iter().enumerate() {
            for (value, component) in out.iter_mut().zip(head_of(line * steps + key, 2)) {
                *value += weight * component;
            }
        }
    }
    layers.ctx.barrier();
}

/// Adds step `t`'s sinusoidal position to every line's row `t`: channel `2i` `sin(t / base^(2i /
/// WIDTH))`, channel `2i + 1` its cosine.
fn add_positions(values: &mut [f32], steps: usize) {
    for (row, values) in values.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        let position = (row % steps) as f32;
        for (pair, [sine, cosine]) in values.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let angle = position / POSITION_BASE.powf((2 * pair) as f32 / WIDTH as f32);
            *sine += angle.sin();
            *cosine += angle.cos();
        }
    }
}

impl Load for Encoder {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let inputs = [1, FRONT[0], FRONT[1]];
        let front =
            [0, 3, 6]
                .into_iter()
                .zip(inputs)
                .zip(FRONT)
                .map(|((index, inputs), outputs)| {
                    Conv2d::load(
                        store,
                        &name(&format!("front.{index}")),
                        conv::Shape::halving(inputs, outputs, FRONT_KERNEL),
                    )
                });
        let front = front.collect::<Result<_, _>>()?;
        let fold_inputs = HEIGHT / STEP * FRONT[2];
        let blocks = load_each(store, &name("blocks"), BLOCKS, |store, block| {
            Block::load(store, block, BLOCK)
        })?;
        let layers = load_each(store, &name("attn"), LAYERS, |store, layer| {
            Attention::load(store, layer, ())
        })?;
        Ok(Self {
            front,
            fold: Linear::load(store, &name("fold"), [WIDTH, fold_inputs])?,
            scale: Vector::load(store, &name("fold_bn.scale"), WIDTH)?,
            shift: Vector::load(store, &name("fold_bn.shift"), WIDTH)?,
            blocks,
            layers,
        })
    }
}

impl Load for Attention {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm1: LayerNorm::load(store, &name("ln1"), WIDTH)?,
            qkv: Linear::load(store, &name("qkv"), [3 * WIDTH, WIDTH])?,
            out: Linear::load(store, &name("out"), [WIDTH, WIDTH])?,
            norm2: LayerNorm::load(store, &name("ln2"), WIDTH)?,
            up: Linear::load(store, &name("ff1"), [FEED_FORWARD, WIDTH])?,
            down: Linear::load(store, &name("ff2"), [WIDTH, FEED_FORWARD])?,
        })
    }
}
