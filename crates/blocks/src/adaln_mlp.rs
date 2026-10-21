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
//! and `p.final`, each a linear layer.

use std::ops::Range;

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::{layer_norm, silu};
use kernels::{COLUMN_ALIGN, Linear, NARROW_COLUMN_ALIGN, QuantizedRows};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// The condition vector's width.
    pub condition: usize,
    /// The latent's width: the noise in, the output out.
    pub latent: usize,
    pub width: usize,
    pub depth: usize,
    pub norm_eps: f32,
}

/// What the modulation gives a block, `width` values each.
#[derive(Clone, Copy)]
enum Modulated {
    NormWeight,
    NormBias,
    Gate,
}

/// The modulations of each block, in order; the final norm has the first two.
const BLOCK_MODULATIONS: [Modulated; 3] =
    [Modulated::NormWeight, Modulated::NormBias, Modulated::Gate];
const FINAL_MODULATIONS: [Modulated; 2] = [Modulated::NormWeight, Modulated::NormBias];

impl Shape {
    /// Rows of the modulation matrix.
    fn modulations(&self) -> usize {
        let per_block = BLOCK_MODULATIONS.len();
        (per_block * self.depth + FINAL_MODULATIONS.len()) * self.width
    }
}

struct Block {
    up: Linear,
    down: Linear,
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
    activated: Vec<f32>,
    quant: QuantizedRows,
}

impl Buffers {
    pub fn new(head: &AdalnMlp, threads: usize) -> Self {
        let Shape { width, latent, .. } = head.shape;
        Self {
            condition: SharedBuf::new(width),
            modulation: SharedBuf::new(head.shape.modulations()),
            x: SharedBuf::new(width),
            hidden: SharedBuf::new(width),
            output: SharedBuf::new(latent),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

impl AdalnMlp {
    /// The head for one `condition` and `noise`; the result lands in `buffers.output`, readable
    /// after the caller's next barrier.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, condition: &[f32], noise: &[f32]) {
        let cols = ctx.split(self.shape.width, COLUMN_ALIGN);
        let Scratch {
            normed,
            activated,
            quant,
        } = buffers.scratch.get(ctx);
        project(
            &self.cond,
            condition,
            cols.clone(),
            &buffers.condition,
            quant,
        );
        project(&self.input, noise, cols.clone(), &buffers.x, quant);
        ctx.barrier();
        activated.clear();
        activated.extend(buffers.condition.read().iter().map(|value| silu(*value)));
        let modulation_cols = ctx.split(self.shape.modulations(), COLUMN_ALIGN);
        let modulation = &buffers.modulation;
        project(
            &self.modulation,
            activated,
            modulation_cols,
            modulation,
            quant,
        );
        ctx.barrier();
        for (index, block) in self.blocks.iter().enumerate() {
            self.normalize(buffers, index, normed);
            project(&block.up, normed, cols.clone(), &buffers.hidden, quant);
            for value in buffers.hidden.part(cols.clone()) {
                *value = silu(*value);
            }
            ctx.barrier();
            self.gated_residual(buffers, index, cols.clone(), quant);
            ctx.barrier();
        }
        self.normalize(buffers, self.shape.depth, normed);
        let outputs = ctx.split(self.shape.latent, NARROW_COLUMN_ALIGN);
        project(&self.last, normed, outputs, &buffers.output, quant);
    }

    /// `x[cols] += gate ⊙ down(hidden)` for block `index`.
    fn gated_residual(
        &self,
        buffers: &Buffers,
        index: usize,
        cols: Range<usize>,
        quant: &mut QuantizedRows,
    ) {
        let gate = self.modulated(buffers, index, Modulated::Gate);
        let down = &self.blocks[index].down;
        let input = quant.input(&down.matrix(), buffers.hidden.read());
        down.project(&input, cols, |_, col, value| {
            buffers.x.set(col, buffers.x.get(col) + gate[col] * value);
        });
    }

    /// `x` under the LayerNorm weight and bias of block `index` (the final norm for
    /// `index == depth`).
    fn normalize(&self, buffers: &Buffers, index: usize, out: &mut Vec<f32>) {
        let weight = self.modulated(buffers, index, Modulated::NormWeight);
        let bias = self.modulated(buffers, index, Modulated::NormBias);
        out.resize(self.shape.width, 0.0);
        layer_norm(buffers.x.read(), weight, bias, self.shape.norm_eps, out);
    }

    /// What the modulation gives block `index` (the final norm for `index == depth`).
    fn modulated<'buf>(
        &self,
        buffers: &'buf Buffers,
        index: usize,
        part: Modulated,
    ) -> &'buf [f32] {
        let width = self.shape.width;
        let start = (BLOCK_MODULATIONS.len() * index + part as usize) * width;
        &buffers.modulation.read()[start..start + width]
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
    let input = quant.input(&linear.matrix(), x);
    linear.project(&input, cols, |_, col, value| out.set(col, value));
}

impl Load for AdalnMlp {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            condition,
            latent,
            width,
            ..
        } = shape;
        let blocks = load_each(
            store,
            &format!("{prefix}.blocks"),
            shape.depth,
            |store, block| {
                let mut scope = Scope::new(store, block);
                Ok(Block {
                    up: scope.load("up", [width, width])?,
                    down: scope.load("down", [width, width])?,
                })
            },
        )?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            cond: scope.load("cond", [width, condition])?,
            input: scope.load("input", [width, latent])?,
            modulation: scope.load("modulation", [shape.modulations(), width])?,
            last: scope.load("final", [latent, width])?,
            blocks,
            shape,
        })
    }
}
