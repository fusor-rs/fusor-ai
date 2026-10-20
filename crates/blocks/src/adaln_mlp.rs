//! The adaLN residual MLP of MAR-style flow heads (MAR, Pocket TTS, Phonon): a condition vector
//! modulates every block of an MLP over a small latent.
//!
//! `y = cond · c + b`; one `modulation` matrix maps `SiLU(y)` to every block's LayerNorm weight,
//! LayerNorm bias and gate, then the final LayerNorm's weight and bias (each `width` long, in that
//! order): the compiler folds the checkpoint's norm affines, `1 + scale` and shifts into it. Then
//! `x = input · noise + b`; per block `x += gate ⊙ down(SiLU(up(norm(x))))`; the output is
//! `final(norm(x))`.
//!
//! `forward` runs inside a team run, every phase split by columns: the condition and input
//! projections, the modulation, then two phases per block and one for the output.
//!
//! Tensors under a prefix `p`: `p.cond`, `p.input`, `p.modulation`, `p.blocks.{i}.{up, down}`
//! and `p.final`, each a linear layer with a bias.

use std::ops::Range;

use bundle::{Load, Store};
use kernels::ops::{layer_norm, silu};
use kernels::{COLUMN_ALIGN, Error, Linear, NARROW_COLUMN_ALIGN, QuantizedRows, matmul};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub width: usize,
    pub depth: usize,
    pub norm_eps: f32,
}

pub struct Block {
    pub up: Linear,
    pub down: Linear,
}

pub struct AdalnMlp {
    shape: Shape,
    cond: Linear,
    input: Linear,
    modulation: Linear,
    blocks: Vec<Block>,
    last: Linear,
}

/// Activations of one row.
pub struct Buffers {
    condition: SharedBuf,
    modulation: SharedBuf,
    x: SharedBuf,
    hidden: SharedBuf,
    /// The head's output, written by `forward`.
    pub output: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    normed: Vec<f32>,
    quant: QuantizedRows,
}

impl Buffers {
    pub fn new(head: &AdalnMlp, threads: usize) -> Self {
        let width = head.shape.width;
        Self {
            condition: SharedBuf::new(width),
            modulation: SharedBuf::new(head.modulation.matrix().rows),
            x: SharedBuf::new(width),
            hidden: SharedBuf::new(width),
            output: SharedBuf::new(head.last.matrix().rows),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

impl AdalnMlp {
    pub fn new(shape: Shape, parts: [Linear; 4], blocks: Vec<Block>) -> Result<Self, Error> {
        let [cond, input, modulation, last] = parts;
        let modulated = (3 * shape.depth + 2) * shape.width;
        let linears = [&cond, &input, &modulation, &last].into_iter();
        let biased = linears
            .chain(blocks.iter().flat_map(|block| [&block.up, &block.down]))
            .all(|linear| linear.bias().is_some());
        if blocks.len() != shape.depth || modulation.matrix().rows != modulated || !biased {
            return Err(Error::Format(format!(
                "{} blocks and {} modulations for a {}-block head of width {}, all with biases",
                blocks.len(),
                modulation.matrix().rows,
                shape.depth,
                shape.width
            )));
        }
        Ok(Self {
            shape,
            cond,
            input,
            modulation,
            blocks,
            last,
        })
    }

    /// The head for one `condition` and `noise`; the result lands in `buffers.output`, readable
    /// after the caller's next barrier.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, condition: &[f32], noise: &[f32]) {
        let cols = ctx.split(self.shape.width, COLUMN_ALIGN);
        let scratch = buffers.scratch.get(ctx);
        let quant = &mut scratch.quant;
        project(
            &self.cond,
            condition,
            cols.clone(),
            &buffers.condition,
            quant,
        );
        project(&self.input, noise, cols.clone(), &buffers.x, quant);
        ctx.barrier();
        scratch.normed.clear();
        let activated = buffers.condition.read().iter().map(|value| silu(*value));
        scratch.normed.extend(activated);
        let rows = ctx.split(self.modulation.matrix().rows, COLUMN_ALIGN);
        project(
            &self.modulation,
            &scratch.normed,
            rows,
            &buffers.modulation,
            quant,
        );
        ctx.barrier();
        for (index, block) in self.blocks.iter().enumerate() {
            self.normalize(buffers, index, &mut scratch.normed);
            project(
                &block.up,
                &scratch.normed,
                cols.clone(),
                &buffers.hidden,
                quant,
            );
            for value in buffers.hidden.part(cols.clone()) {
                *value = silu(*value);
            }
            ctx.barrier();
            self.gated_residual(buffers, index, cols.clone(), quant);
            ctx.barrier();
        }
        self.normalize(buffers, self.shape.depth, &mut scratch.normed);
        let outputs = ctx.split(self.last.matrix().rows, NARROW_COLUMN_ALIGN);
        project(&self.last, &scratch.normed, outputs, &buffers.output, quant);
    }

    /// `x[cols] += gate ⊙ down(hidden)` for block `index`.
    fn gated_residual(
        &self,
        buffers: &Buffers,
        index: usize,
        cols: Range<usize>,
        quant: &mut QuantizedRows,
    ) {
        let width = self.shape.width;
        let gate = &buffers.modulation.read()[(3 * index + 2) * width..];
        let down = &self.blocks[index].down;
        let input = quant.input(&down.matrix(), buffers.hidden.read(), 1);
        let bias = down.bias().expect("checked in AdalnMlp::new");
        matmul(&down.matrix(), &input, 1, cols, |_, col, value| {
            let update = gate[col] * (value + bias[col]);
            buffers.x.set(col, buffers.x.get(col) + update);
        });
    }

    /// `x` under the LayerNorm weight and bias of block `index` (the final norm for
    /// `index == depth`).
    fn normalize(&self, buffers: &Buffers, index: usize, out: &mut Vec<f32>) {
        let width = self.shape.width;
        let modulation = &buffers.modulation.read()[3 * width * index..];
        let (weight, bias) = (&modulation[..width], &modulation[width..2 * width]);
        out.resize(width, 0.0);
        layer_norm(buffers.x.read(), weight, bias, self.shape.norm_eps, out);
    }
}

/// `out[cols] = linear(x)[cols]` for one row.
fn project(
    linear: &Linear,
    x: &[f32],
    cols: Range<usize>,
    out: &SharedBuf,
    quant: &mut QuantizedRows,
) {
    let matrix = linear.matrix();
    let input = quant.input(&matrix, x, 1);
    let bias = linear.bias().expect("checked in AdalnMlp::new");
    matmul(&matrix, &input, 1, cols, |_, col, value| {
        out.set(col, value + bias[col]);
    });
}

impl Load for Block {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        Ok(Self {
            up: Linear::load(store, &format!("{prefix}.up"), ())?,
            down: Linear::load(store, &format!("{prefix}.down"), ())?,
        })
    }
}

impl Load for AdalnMlp {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let mut linear = |part: &str| Linear::load(store, &format!("{prefix}.{part}"), ());
        let parts = [
            linear("cond")?,
            linear("input")?,
            linear("modulation")?,
            linear("final")?,
        ];
        let blocks = (0..shape.depth)
            .map(|i| Block::load(store, &format!("{prefix}.blocks.{i}"), ()))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(shape, parts, blocks)?)
    }
}
