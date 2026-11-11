//! An image's pixels into patches and back: the patches before the embedding, and the head's
//! output after it.

use super::{Dit, HEAD_EPS, HeadChunk, Pass};
use crate::section;

impl Pass<'_> {
    /// `patched` = each token's patch of `image`: its pixels row-major, a pixel's channels
    /// contiguous, zeros past the image's edges.
    pub(super) fn patchify(&self) {
        let patch_values = self.shape.patch_values();
        let image = (self.buffers.image.read(), self.shape.channels);
        for token in self.ctx.split(self.tokens(), 1) {
            let patch = self
                .buffers
                .patched
                .part(token * patch_values..(token + 1) * patch_values);
            self.patching.gather(image, token, patch);
        }
        self.ctx.barrier();
    }
}

impl Dit {
    /// `prediction` = the head's output for every image token, its patches put back where they
    /// came from; the pixels past the image are dropped.
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
        for token in ctx.split(pass.tokens(), 1) {
            let row = token * hidden..(token + 1) * hidden;
            let normed = buffers.normed.part(row.clone());
            self.head_norm
                .apply(&buffers.hidden.read()[row], HEAD_EPS, normed);
            for ((value, &scale), &shift) in normed.iter_mut().zip(scales).zip(shifts) {
                *value = *value * scale + shift;
            }
        }
        ctx.barrier();
        let normed = (&buffers.normed, pass.tokens());
        buffers
            .rows
            .project_synced(ctx, &self.head, normed, |token, col, value| {
                buffers.patched.set(token * patch_values + col, value);
            });
        let head_output = buffers.patched.read();
        for token in ctx.split(pass.tokens(), 1) {
            let patch = &head_output[token * patch_values..(token + 1) * patch_values];
            patching.scatter((patch, shape.channels), token, &buffers.prediction);
        }
        ctx.barrier();
    }
}
