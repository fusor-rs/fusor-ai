//! The pixel head: four blocks over 16 channels per pixel. Each block compresses a patch's
//! pixels to one token of 1280 values, attends over the patches' tokens (10 heads of 128, the
//! image's rotary table), expands the result back to the pixels and adds it, scaled and shifted
//! per pixel by a modulation of the patch's trunk state; then a per-pixel MLP, added the same
//! way. A last RMSNorm and linear give each pixel's velocity.
//!
//! The per-pixel part of a block (the "tail") runs on each thread's own patches in chunks, on its
//! scratch, with no product split over the team: the modulations of a patch are 16 384 values,
//! too many to keep for every patch of an image.

use std::ops::Range;

use blocks::attention::{Call, Operand, attend};
use bundle::{Scope, Store, load_each};
use kernels::ops::{Exponentials, gelu_erf, rms_norm};
use kernels::{Linear, Vector};

use super::buffers::TailScratch;
use super::embedding::embed_pixels;
use super::{DIT, HEAD_DIM, Half, PAIRS, Pass, turn_heads};
use crate::Error;
use crate::model::{HIDDEN, IMAGE_CHANNELS, NORM_EPS, PATCH_PIXELS, PIXEL_BLOCKS, PIXEL_HIDDEN};

/// The width of a patch's compressed token.
pub(super) const PIXEL_ATTENTION: usize = 1280;
pub(super) const PIXEL_QKV: usize = 3 * PIXEL_ATTENTION;
const PIXEL_HEADS: usize = PIXEL_ATTENTION / HEAD_DIM;
const PIXEL_FEED_FORWARD: usize = 64;
/// A patch's pixels' channels.
const PATCH_VALUES: usize = PATCH_PIXELS * PIXEL_HIDDEN;
/// A pixel's modulation: the scale and shift of the attention's residual, then the MLP's.
const PIXEL_MODULATION: usize = 4 * PIXEL_HIDDEN;
const PATCH_MODULATION: usize = PATCH_PIXELS * PIXEL_MODULATION;
/// Patches a thread runs through the tail at a time.
const PATCH_CHUNK: usize = 32;

struct PixelBlock {
    norm1: Vector,
    compress: Linear,
    qkv: Linear,
    q_norm: Vector,
    k_norm: Vector,
    out: Linear,
    expand: Linear,
    modulation: Linear,
    norm2: Vector,
    ff1: Linear,
    ff2: Linear,
}

pub(super) struct PixelHead {
    embed: Linear,
    blocks: Vec<PixelBlock>,
    final_norm: Vector,
    last: Linear,
}

impl PixelHead {
    /// The head under `dit.pixel`, predicting `outputs` channels per pixel.
    pub(super) fn load(store: &mut Store, outputs: usize) -> Result<Self, Error> {
        let prefix = format!("{DIT}.pixel");
        let blocks = load_each(
            store,
            &format!("{prefix}.blocks"),
            PIXEL_BLOCKS,
            PixelBlock::load,
        )?;
        let mut scope = Scope::new(store, &prefix);
        Ok(Self {
            embed: scope.load("embed", [PIXEL_HIDDEN, IMAGE_CHANNELS])?,
            final_norm: scope.load("final_norm", PIXEL_HIDDEN)?,
            last: scope.load("final", [outputs, PIXEL_HIDDEN])?,
            blocks,
        })
    }

    /// `velocity` = the prediction for every pixel, from the trunk's patch states in `normed`.
    pub(super) fn forward(&self, pass: &Pass) {
        embed_pixels(pass, &self.embed);
        for block in &self.blocks {
            block.forward(pass);
        }
        self.finish(pass);
    }

    /// The final norm and linear over this thread's patches, put in place; then a barrier.
    fn finish(&self, pass: &Pass) {
        let Pass {
            ctx,
            buffers,
            patching,
            ..
        } = pass;
        let scratch = buffers.tail.get(ctx);
        let outputs = self.last.matrix().rows;
        for chunk in chunks(ctx.split(pass.patches(), 1)) {
            let pixels =
                &buffers.pixels.read()[chunk.start * PATCH_VALUES..chunk.end * PATCH_VALUES];
            scratch.normed.resize(pixels.len(), 0.0);
            rms_norm(
                pixels,
                self.final_norm.values(),
                NORM_EPS,
                &mut scratch.normed,
            );
            scratch
                .out
                .resize(chunk.len() * PATCH_PIXELS * outputs, 0.0);
            self.last
                .apply(&scratch.normed, &mut scratch.quant, &mut scratch.out);
            let patches = scratch.out.chunks_exact(PATCH_PIXELS * outputs);
            for (patch, values) in chunk.zip(patches) {
                patching.scatter((values, outputs), patch, &buffers.velocity);
            }
        }
        ctx.barrier();
    }
}

/// `range` in runs of `PATCH_CHUNK` patches.
fn chunks(range: Range<usize>) -> impl Iterator<Item = Range<usize>> {
    let end = range.end;
    range
        .step_by(PATCH_CHUNK)
        .map(move |start| start..(start + PATCH_CHUNK).min(end))
}

impl PixelBlock {
    fn load(store: &mut Store, prefix: &str) -> Result<Self, Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", PIXEL_HIDDEN)?,
            compress: scope.load("compress", [PIXEL_ATTENTION, PATCH_VALUES])?,
            qkv: scope.load("qkv", [PIXEL_QKV, PIXEL_ATTENTION])?,
            q_norm: scope.load("q_norm", HEAD_DIM)?,
            k_norm: scope.load("k_norm", HEAD_DIM)?,
            out: scope.load("out", [PIXEL_ATTENTION; 2])?,
            expand: scope.load("expand", [PATCH_VALUES, PIXEL_ATTENTION])?,
            modulation: scope.load("modulation", [PATCH_MODULATION, HIDDEN])?,
            norm2: scope.load("norm2", PIXEL_HIDDEN)?,
            ff1: scope.load("ff1", [PIXEL_FEED_FORWARD, PIXEL_HIDDEN])?,
            ff2: scope.load("ff2", [PIXEL_HIDDEN, PIXEL_FEED_FORWARD])?,
        })
    }

    /// `pixels` += the block's two residuals, then a barrier.
    fn forward(&self, pass: &Pass) {
        self.attend_patches(pass);
        let scratch = pass.buffers.tail.get(pass.ctx);
        for chunk in chunks(pass.ctx.split(pass.patches(), 1)) {
            self.residual(pass, chunk.clone(), scratch, Half::Attention);
            self.residual(pass, chunk, scratch, Half::FeedForward);
        }
        pass.ctx.barrier();
    }

    /// `compact` = the attention among the patches' compressed tokens, projected; every phase
    /// ends with a barrier.
    fn attend_patches(&self, pass: &Pass) {
        let (ctx, buffers, patches) = (pass.ctx, pass.buffers, pass.patches());
        let pixels = buffers.pixels.read();
        for patch in ctx.split(patches, 1) {
            let range = patch * PATCH_VALUES..(patch + 1) * PATCH_VALUES;
            let normed = buffers.pixels_normed.part(range.clone());
            rms_norm(&pixels[range], self.norm1.values(), NORM_EPS, normed);
        }
        ctx.barrier();
        let normed = (buffers.pixels_normed.read(), patches);
        buffers.project(ctx, &self.compress, normed, |row, col, value| {
            buffers.compact.set(row * PIXEL_ATTENTION + col, value);
        });
        let compact = (buffers.compact.read(), patches);
        buffers.project(ctx, &self.qkv, compact, |row, col, value| {
            buffers.pixel_qkv.set(row * PIXEL_QKV + col, value);
        });
        self.prepare_heads(pass);
        let operand = |offset| Operand {
            values: buffers.pixel_qkv.read(),
            stride: PIXEL_QKV,
            offset,
        };
        let call = Call {
            heads: PIXEL_HEADS,
            kv_heads: PIXEL_HEADS,
            head_dim: HEAD_DIM,
            images: 1,
            queries: patches,
            keys: patches,
            query: operand(0),
            key: operand(PIXEL_ATTENTION),
            value: operand(2 * PIXEL_ATTENTION),
            exponentials: Exponentials::Exact,
        };
        attend(
            ctx,
            &call,
            &buffers.pixel_attention,
            buffers.scratch.get(ctx),
        );
        ctx.barrier();
        let attended = (buffers.pixel_attention.read(), patches);
        buffers.project(ctx, &self.out, attended, |row, col, value| {
            buffers.compact.set(row * PIXEL_ATTENTION + col, value);
        });
    }

    /// Every patch's query and key heads RMS-normalized and turned by the patch's rotary angles;
    /// then a barrier.
    fn prepare_heads(&self, pass: &Pass) {
        let buffers = pass.buffers;
        let scratch = buffers.heads.get(pass.ctx);
        for patch in pass.ctx.split(pass.patches(), 1) {
            let start = patch * PIXEL_QKV;
            let heads = buffers.pixel_qkv.part(start..start + 2 * PIXEL_ATTENTION);
            let pairs = patch * PAIRS..(patch + 1) * PAIRS;
            let angles = (
                &buffers.image_cos.read()[pairs.clone()],
                &buffers.image_sin.read()[pairs],
            );
            let norms = (&self.q_norm, &self.k_norm);
            turn_heads(heads, PIXEL_ATTENTION, norms, angles, scratch);
        }
        pass.ctx.barrier();
    }

    /// The pixels of `chunk`'s patches += one residual: the expanded attention, or the per-pixel
    /// MLP, scaled and shifted by the patch's modulation (computed with the attention's).
    fn residual(
        &self,
        pass: &Pass,
        chunk: Range<usize>,
        scratch: &mut TailScratch,
        residual: Half,
    ) {
        let buffers = pass.buffers;
        let pixels = buffers
            .pixels
            .part(chunk.start * PATCH_VALUES..chunk.end * PATCH_VALUES);
        let branch = match residual {
            Half::Attention => {
                self.modulate(pass, chunk.clone(), scratch);
                let compact = &buffers.compact.read()
                    [chunk.start * PIXEL_ATTENTION..chunk.end * PIXEL_ATTENTION];
                scratch.expanded.resize(pixels.len(), 0.0);
                self.expand
                    .apply(compact, &mut scratch.quant, &mut scratch.expanded);
                &scratch.expanded
            }
            Half::FeedForward => {
                self.feed_forward(pixels, scratch);
                &scratch.out
            }
        };
        let modulated = pixels
            .as_chunks_mut::<PIXEL_HIDDEN>()
            .0
            .iter_mut()
            .zip(branch.as_chunks::<PIXEL_HIDDEN>().0)
            .zip(scratch.modulations.as_chunks::<PIXEL_MODULATION>().0);
        for ((pixel, added), modulation) in modulated {
            let (scale, shift) = terms(modulation, residual);
            for (((value, added), scale), shift) in
                pixel.iter_mut().zip(added).zip(scale).zip(shift)
            {
                *value += added * (1.0 + scale) + shift;
            }
        }
    }

    /// `scratch.modulations` = the modulations of `chunk`'s patches' trunk states.
    fn modulate(&self, pass: &Pass, chunk: Range<usize>, scratch: &mut TailScratch) {
        let condition = &pass.buffers.normed.read()[chunk.start * HIDDEN..chunk.end * HIDDEN];
        scratch
            .modulations
            .resize(chunk.len() * PATCH_MODULATION, 0.0);
        self.modulation
            .apply(condition, &mut scratch.quant, &mut scratch.modulations);
    }

    /// `scratch.out` = the per-pixel MLP of `pixels` (rows of `PIXEL_HIDDEN`).
    fn feed_forward(&self, pixels: &[f32], scratch: &mut TailScratch) {
        scratch.normed.resize(pixels.len(), 0.0);
        rms_norm(pixels, self.norm2.values(), NORM_EPS, &mut scratch.normed);
        let pixel_count = pixels.len() / PIXEL_HIDDEN;
        scratch.inner.resize(pixel_count * PIXEL_FEED_FORWARD, 0.0);
        self.ff1
            .apply(&scratch.normed, &mut scratch.quant, &mut scratch.inner);
        for value in &mut scratch.inner {
            *value = gelu_erf(*value);
        }
        scratch.out.resize(pixels.len(), 0.0);
        self.ff2
            .apply(&scratch.inner, &mut scratch.quant, &mut scratch.out);
    }
}

/// A pixel's scale and shift for one residual.
fn terms(modulation: &[f32], residual: Half) -> (&[f32], &[f32]) {
    let first = 2 * residual as usize * PIXEL_HIDDEN;
    let scale = &modulation[first..first + PIXEL_HIDDEN];
    let shift = &modulation[first + PIXEL_HIDDEN..first + 2 * PIXEL_HIDDEN];
    (scale, shift)
}
