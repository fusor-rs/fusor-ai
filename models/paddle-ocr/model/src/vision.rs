//! The vision tower and the projector over one image's patches: each patch's embedding plus the
//! learned position grid resampled to the image's grid, the encoder and its final norm; then
//! every 2 × 2 block of patches, normed and side by side, through the projector's MLP into one
//! language-model token.

use blocks::vit::{self, Grid};
use kernels::image::{Size, resize_bilinear};
use kernels::ops::gelu_erf;
use kernels::{COLUMN_ALIGN, QuantizedRows, SharedRows, project};
use team::{Ctx, PerThread, SharedBuf};

use crate::image::MAX_PIXELS;
use crate::model::{
    HIDDEN, MERGE, MERGED, PATCH, POSITION_GRID, PROJECTOR_NORM_EPS, Projector, VISION_SHAPE,
    Vision,
};

/// The most patches an image has.
pub const MAX_PATCHES: usize = MAX_PIXELS / (PATCH * PATCH);
const WIDTH: usize = VISION_SHAPE.hidden;
const MERGED_PATCHES: usize = MERGE * MERGE;

pub(crate) struct Buffers {
    pub(crate) encoder: vit::Buffers,
    /// The encoder's normed output, `patches × WIDTH`.
    pub(crate) seen: SharedBuf,
    /// Merged patches side by side, normed, `tokens × MERGED`.
    merged: SharedBuf,
    /// The projector's inner activations, `tokens × MERGED`.
    inner: SharedBuf,
    /// The image's tokens, `tokens × HIDDEN`.
    pub(crate) tokens: SharedBuf,
    quant: PerThread<QuantizedRows>,
}

impl Buffers {
    pub(crate) fn new(threads: usize) -> Self {
        let tokens = MAX_PATCHES / MERGED_PATCHES;
        Self {
            encoder: VISION_SHAPE.buffers(MAX_PATCHES, threads),
            seen: SharedBuf::new(MAX_PATCHES * WIDTH),
            merged: SharedBuf::new(tokens * MERGED),
            inner: SharedBuf::new(tokens * MERGED),
            tokens: SharedBuf::new(tokens * HIDDEN),
            quant: PerThread::new(threads, QuantizedRows::default),
        }
    }
}

impl Vision {
    /// The position grid resampled to `grid`, `grid.patches() × WIDTH`.
    pub(crate) fn positions(&self, grid: Grid) -> Vec<f32> {
        let side = Size {
            width: POSITION_GRID,
            height: POSITION_GRID,
        };
        let to = Size {
            width: grid.columns,
            height: grid.rows,
        };
        resize_bilinear(&self.positions, side, to, WIDTH)
    }

    /// `encoder.hidden` = each patch's embedding plus its position, for this thread's columns.
    pub(crate) fn embed(&self, ctx: &Ctx, buffers: &Buffers, patches: &[f32], positions: &[f32]) {
        let input = buffers.quant.get(ctx).input(&self.patch.matrix(), patches);
        let cols = ctx.split(WIDTH, COLUMN_ALIGN);
        let hidden = &buffers.encoder.hidden;
        self.patch.project(&input, cols, |row, col, value| {
            let at = row * WIDTH + col;
            hidden.set(at, value + positions[at]);
        });
    }

    /// The encoder over `encoder.hidden`, then its final norm into `seen`.
    pub(crate) fn encode(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        self.encoder.forward(ctx, &buffers.encoder, grid);
        let rows = ctx.split(grid.patches(), 1);
        let range = rows.start * WIDTH..rows.end * WIDTH;
        let hidden = &buffers.encoder.hidden.read()[range.clone()];
        self.norm
            .apply(hidden, VISION_SHAPE.norm_eps, buffers.seen.part(range));
        ctx.barrier();
    }
}

impl Projector {
    /// `seen` (`grid.patches()` rows) into `tokens` (one per 2 × 2 patches, row-major).
    pub(crate) fn forward(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        let tokens = grid.patches() / MERGED_PATCHES;
        let seen = buffers.seen.read();
        for token in ctx.split(tokens, 1) {
            let merged = buffers.merged.part(token * MERGED..(token + 1) * MERGED);
            let slots = merged.as_chunks_mut::<WIDTH>().0;
            for (patch, slot) in grid.merged_patches(token, MERGE).zip(slots) {
                let row = &seen[patch * WIDTH..(patch + 1) * WIDTH];
                self.norm
                    .apply(row, PROJECTOR_NORM_EPS, slot.as_mut_slice());
            }
        }
        ctx.barrier();
        let quant = buffers.quant.get(ctx);
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
