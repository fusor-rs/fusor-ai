//! The projector over the vision tower's output (`blocks::vit::tower`): every 2 × 2 block of
//! patches, normed and side by side, through the projector's MLP into one language-model token;
//! and the projector's activations.

use blocks::vit::{Grid, tower};
use kernels::ops::gelu_erf;
use kernels::{SharedRows, project};
use team::{Ctx, SharedBuf};

use crate::image::PREPROCESSING;
use crate::model::{HIDDEN, MERGE, MERGED, PROJECTOR_NORM_EPS, Projector, VISION_SHAPE};

/// The most patches an image has.
pub const MAX_PATCHES: usize = PREPROCESSING.max_patches();
const WIDTH: usize = VISION_SHAPE.hidden;
const MERGED_PATCHES: usize = MERGE * MERGE;

pub struct Buffers {
    /// Merged patches side by side, normed, `tokens × MERGED`.
    merged: SharedBuf,
    /// The projector's inner activations, `tokens × MERGED`.
    inner: SharedBuf,
    /// The image's tokens, `tokens × HIDDEN`.
    pub(crate) tokens: SharedBuf,
}

impl Buffers {
    pub(crate) fn new() -> Self {
        let tokens = MAX_PATCHES / MERGED_PATCHES;
        Self {
            merged: SharedBuf::new(tokens * MERGED),
            inner: SharedBuf::new(tokens * MERGED),
            tokens: SharedBuf::new(tokens * HIDDEN),
        }
    }
}

impl Projector {
    /// The tower's output (`grid.patches()` rows) into `tokens` (one per 2 × 2 patches,
    /// row-major).
    pub(crate) fn forward(&self, ctx: &Ctx, tower: &tower::Buffers, buffers: &Buffers, grid: Grid) {
        let tokens = grid.patches() / MERGED_PATCHES;
        let hidden = tower.encoder.hidden.read();
        for token in ctx.split(tokens, 1) {
            let merged = buffers.merged.part(token * MERGED..(token + 1) * MERGED);
            let slots = merged.as_chunks_mut::<WIDTH>().0;
            for (patch, slot) in grid.merged_patches(token, MERGE).zip(slots) {
                let row = &hidden[patch * WIDTH..(patch + 1) * WIDTH];
                self.norm
                    .apply(row, PROJECTOR_NORM_EPS, slot.as_mut_slice());
            }
        }
        ctx.barrier();
        let quant = tower.quant.get(ctx);
        let merged = SharedRows {
            values: &buffers.merged,
            rows: tokens,
        };
        project(ctx, &self.up, merged, quant, |at, value| {
            buffers.inner.set(at, gelu_erf(value));
        });
        ctx.barrier();
        let inner = SharedRows {
            values: &buffers.inner,
            rows: tokens,
        };
        project(ctx, &self.down, inner, quant, |at, value| {
            buffers.tokens.set(at, value);
        });
        ctx.barrier();
    }
}
