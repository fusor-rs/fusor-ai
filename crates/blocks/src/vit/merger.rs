//! A vision-language model's patch merger over the vision tower's output (`tower`): every
//! `merge × merge` block of patches, each LayerNorm-ed, side by side through `down(gelu(up(·)))`
//! (exact GELU) into one language-model token, the tokens row-major over the blocks.
//!
//! Tensors under a prefix `p`: `p.norm` (`weight` and `bias`, a patch wide), `p.up` and `p.down`
//! (linears with biases).

use bundle::{Load, Scope, Store};
use kernels::ops::gelu_erf;
use kernels::{Linear, SharedRows, project};
use team::{Ctx, SharedBuf};

use super::{Grid, tower};
use crate::norm::LayerNorm;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Width of an encoded patch.
    pub patch: usize,
    /// Patches merged along each axis.
    pub merge: usize,
    /// Width of a token: the language model's.
    pub hidden: usize,
    pub norm_eps: f32,
}

impl Shape {
    /// Width of a merged block, its patches side by side.
    fn merged(&self) -> usize {
        self.patch * self.merge * self.merge
    }
}

pub struct Merger {
    shape: Shape,
    norm: LayerNorm,
    up: Linear,
    down: Linear,
}

/// Activations of one image of up to `patches` patches.
pub struct Buffers {
    /// Blocks of normed patches side by side, `tokens × merged`.
    merged: SharedBuf,
    /// The MLP's inner activations, `tokens × merged`.
    inner: SharedBuf,
    /// The image's tokens, `tokens × hidden`.
    pub tokens: SharedBuf,
}

impl Buffers {
    pub fn new(shape: &Shape, patches: usize) -> Self {
        let tokens = patches / (shape.merge * shape.merge);
        Self {
            merged: SharedBuf::new(tokens * shape.merged()),
            inner: SharedBuf::new(tokens * shape.merged()),
            tokens: SharedBuf::new(tokens * shape.hidden),
        }
    }
}

impl Merger {
    /// The tower's output (`grid.patches()` rows) into `buffers.tokens`.
    pub fn forward(&self, ctx: &Ctx, tower: &tower::Buffers, buffers: &Buffers, grid: Grid) {
        let Shape { patch, merge, .. } = self.shape;
        let (merged, tokens) = (self.shape.merged(), grid.patches() / (merge * merge));
        let hidden = tower.encoder.hidden.read();
        for token in ctx.split(tokens, 1) {
            let block = buffers.merged.part(token * merged..(token + 1) * merged);
            for (index, slot) in grid
                .merged_patches(token, merge)
                .zip(block.chunks_exact_mut(patch))
            {
                let row = &hidden[index * patch..(index + 1) * patch];
                self.norm.apply(row, self.shape.norm_eps, slot);
            }
        }
        ctx.barrier();
        let quant = tower.quant.get(ctx);
        let rows = |values| SharedRows {
            values,
            rows: tokens,
        };
        project(ctx, &self.up, rows(&buffers.merged), quant, |at, value| {
            buffers.inner.set(at, gelu_erf(value));
        });
        ctx.barrier();
        project(ctx, &self.down, rows(&buffers.inner), quant, |at, value| {
            buffers.tokens.set(at, value);
        });
        ctx.barrier();
    }
}

impl Load for Merger {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        let merged = shape.merged();
        Ok(Self {
            shape,
            norm: scope.load("norm", shape.patch)?,
            up: scope.load("up", [merged, merged])?,
            down: scope.load("down", [shape.hidden, merged])?,
        })
    }
}
