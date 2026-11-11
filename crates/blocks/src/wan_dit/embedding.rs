//! A video's latents into tokens and back.

use super::{Dit, HeadChunk, Pass};
use crate::section;

impl Dit {
    /// `patched` = each token's patch of `latents`: its pixels row-major, a pixel's channels
    /// contiguous.
    pub(super) fn patchify(&self, pass: &Pass) {
        let Pass {
            ctx,
            buffers,
            shape,
            patching,
            ..
        } = pass;
        let patch_values = shape.patch_values();
        let latents = (buffers.latents.read(), shape.latent);
        for token in ctx.split(pass.tokens(), 1) {
            let patch = buffers
                .patched
                .part(token * patch_values..(token + 1) * patch_values);
            patching.gather(latents, token, patch);
        }
        ctx.barrier();
    }

    /// `velocity` = the head's output for every token, its patches put back where they came from.
    pub(super) fn finish(&self, pass: &Pass) {
        let Pass {
            ctx,
            buffers,
            shape,
            patching,
            ..
        } = pass;
        let hidden = shape.hidden;
        let patch_values = shape.patch_values();
        let modulation = buffers.head_modulation.read();
        let shifts = section(modulation, hidden, HeadChunk::Shift as usize);
        let scales = section(modulation, hidden, HeadChunk::Scale as usize);
        pass.modulated_norm(shifts, scales);
        let normed = (&buffers.normed, pass.tokens());
        buffers
            .rows
            .project_synced(ctx, &self.head, normed, |token, col, value| {
                buffers.inner.set(token * patch_values + col, value);
            });
        let head_output = buffers.inner.read();
        for token in ctx.split(pass.tokens(), 1) {
            let patch = &head_output[token * patch_values..(token + 1) * patch_values];
            patching.scatter((patch, shape.latent), token, &buffers.velocity);
        }
        ctx.barrier();
    }
}
