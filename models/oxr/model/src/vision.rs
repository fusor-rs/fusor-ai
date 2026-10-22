//! The vision tower and the visual vocabulary over one image's patches: each patch's embedding
//! plus the learned position grid resampled to the image's grid, and the encoder; then every
//! 2 × 2 block of encoded patches, side by side, as a distribution over the visual words (their
//! logits, LayerNorm and softmax), and its language-model token, the mixture of the words'
//! embeddings that distribution weighs.

use std::ops::Range;

use blocks::vit::{self, Grid};
use kernels::image::{Size, resize_bicubic};
use kernels::ops::softmax;
use kernels::{
    BLOCK, COLUMN_ALIGN, Input, MatrixStorage, QUAD, Quantized, QuantizedRows, quantize_rows,
};
use team::{Ctx, PerThread, SharedBuf};

use crate::image::MAX_PIXELS;
use crate::model::{
    HIDDEN, MERGE, MERGED, PATCH, POSITION_GRID, VISION_SHAPE, VISUAL_NORM_EPS, VISUAL_VOCABULARY,
    VISUAL_WORDS, Vision, Visual,
};

/// The most patches an image has.
pub const MAX_PATCHES: usize = MAX_PIXELS / (PATCH * PATCH);
/// The most tokens an image has.
pub const MAX_IMAGE_TOKENS: usize = MAX_PATCHES / MERGED_PATCHES;
const WIDTH: usize = VISION_SHAPE.hidden;
const MERGED_PATCHES: usize = MERGE * MERGE;
/// Tokens whose visual-word distributions are held at once (`VISUAL_VOCABULARY` values each).
const CHUNK: usize = 64;

pub(crate) struct Buffers {
    pub(crate) encoder: vit::Buffers,
    /// Merged patches side by side, `tokens × MERGED`.
    merged: SharedBuf,
    /// A chunk's logits over the visual vocabulary, `CHUNK × VISUAL_VOCABULARY`.
    logits: SharedBuf,
    /// A chunk's distributions over the visual vocabulary, `CHUNK × VISUAL_VOCABULARY`; the
    /// indicators' columns stay zero.
    distributions: SharedBuf,
    /// The distributions quantized as `mix`'s input, when its weights are quantized.
    quants: SharedBuf<i8>,
    scales: SharedBuf,
    corrections: SharedBuf<i32>,
    /// The image's tokens, `tokens × HIDDEN`.
    pub(crate) tokens: SharedBuf,
    quant: PerThread<QuantizedRows>,
}

impl Buffers {
    pub(crate) fn new(threads: usize) -> Self {
        let chunk = CHUNK * VISUAL_VOCABULARY;
        Self {
            encoder: VISION_SHAPE.buffers(MAX_PATCHES, threads),
            merged: SharedBuf::new(MAX_IMAGE_TOKENS * MERGED),
            logits: SharedBuf::new(chunk),
            distributions: SharedBuf::new(chunk),
            quants: SharedBuf::new(chunk),
            scales: SharedBuf::new(chunk / BLOCK),
            corrections: SharedBuf::new(chunk / BLOCK),
            tokens: SharedBuf::new(MAX_IMAGE_TOKENS * HIDDEN),
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
        resize_bicubic(&self.positions, side, to, WIDTH)
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

    /// The encoder over `encoder.hidden`, in place.
    pub(crate) fn encode(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        self.encoder.forward(ctx, &buffers.encoder, grid);
    }
}

impl Visual {
    /// `encoder.hidden` (`grid.patches()` rows) into `tokens`, one per 2 × 2 patches, row-major.
    pub(crate) fn forward(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        let tokens = grid.patches() / MERGED_PATCHES;
        let hidden = buffers.encoder.hidden.read();
        for token in ctx.split(tokens, 1) {
            let merged = buffers.merged.part(token * MERGED..(token + 1) * MERGED);
            let slots = merged.as_chunks_mut::<WIDTH>().0;
            for (patch, slot) in grid.merged_patches(token, MERGE).zip(slots) {
                slot.copy_from_slice(&hidden[patch * WIDTH..(patch + 1) * WIDTH]);
            }
        }
        ctx.barrier();
        for first in (0..tokens).step_by(CHUNK) {
            self.chunk(ctx, buffers, first..tokens.min(first + CHUNK));
        }
    }

    /// The tokens of `chunk`: the head's logits, the distributions, then their mixtures.
    fn chunk(&self, ctx: &Ctx, buffers: &Buffers, chunk: Range<usize>) {
        let merged = &buffers.merged.read()[chunk.start * MERGED..chunk.end * MERGED];
        let input = buffers.quant.get(ctx).input(&self.head.matrix(), merged);
        let cols = ctx.split(VISUAL_VOCABULARY, COLUMN_ALIGN);
        self.head.project_quads(&input, cols, |row, first, values| {
            let at = row * VISUAL_VOCABULARY + first;
            buffers.logits.part(at..at + QUAD).copy_from_slice(&values);
        });
        ctx.barrier();
        for row in ctx.split(chunk.len(), 1) {
            self.distribution(buffers, row);
        }
        ctx.barrier();
        let input = self.mix_input(buffers, chunk.len());
        let cols = ctx.split(HIDDEN, COLUMN_ALIGN);
        self.mix.project_quads(&input, cols, |row, first, values| {
            let at = (chunk.start + row) * HIDDEN + first;
            buffers.tokens.part(at..at + QUAD).copy_from_slice(&values);
        });
        ctx.barrier();
    }

    /// Row `row` of the chunk's logits to its distribution over the visual words, and, for
    /// quantized `mix` weights, to that distribution quantized as `mix`'s input.
    fn distribution(&self, buffers: &Buffers, row: usize) {
        let values = row * VISUAL_VOCABULARY..(row + 1) * VISUAL_VOCABULARY;
        let words = values.start..values.start + VISUAL_WORDS;
        let distribution = buffers.distributions.part(words.clone());
        self.norm
            .apply(&buffers.logits.read()[words], VISUAL_NORM_EPS, distribution);
        softmax(distribution);
        let MatrixStorage::Quant { format, .. } = self.mix.matrix().storage else {
            return;
        };
        let groups = values.start / format.group()..values.end / format.group();
        quantize_rows(
            &buffers.distributions.read()[values.clone()],
            format,
            buffers.quants.part(values),
            buffers.scales.part(groups.clone()),
            buffers.corrections.part(groups),
        );
    }

    /// The first `rows` distributions as `mix`'s input, once every thread's rows are in.
    fn mix_input<'buffers>(&self, buffers: &'buffers Buffers, rows: usize) -> Input<'buffers> {
        let len = rows * VISUAL_VOCABULARY;
        match self.mix.matrix().storage {
            MatrixStorage::F32(_) => Input::F32(&buffers.distributions.read()[..len]),
            MatrixStorage::Quant { format, .. } => {
                let groups = len / format.group();
                Input::Quant(Quantized {
                    format,
                    quants: &buffers.quants.read()[..len],
                    scales: &buffers.scales.read()[..groups],
                    corrections: &buffers.corrections.read()[..groups],
                })
            }
        }
    }
}
