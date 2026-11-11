//! The context's rows: SigLIP's states and T5Gemma2's, projected to the transformer's width.

use kernels::ops::{gelu_erf, gelu_tanh};
use team::Ctx;

use super::{Buffers, Call, Contexts, HIDDEN, Transformer};
use crate::model::{IMAGE_HIDDEN, IMAGE_NORM_EPS, IMAGE_TOKENS, TEXT_HIDDEN};
use crate::session::Branch;

impl Transformer {
    /// The projection of SigLIP's `states` (`IMAGE_TOKENS` rows of `IMAGE_HIDDEN`) into
    /// `contexts`; nothing without a first image.
    pub(crate) fn embed_image(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        (states, contexts): (Option<&[f32]>, &Contexts),
    ) {
        let Some(states) = states else {
            return;
        };
        let image = &self.image;
        for row in ctx.split(IMAGE_TOKENS, 1) {
            let source = &states[row * IMAGE_HIDDEN..(row + 1) * IMAGE_HIDDEN];
            let normed = buffers
                .image_normed
                .part(row * IMAGE_HIDDEN..(row + 1) * IMAGE_HIDDEN);
            image.norm_in.apply(source, IMAGE_NORM_EPS, normed);
        }
        ctx.barrier();
        let normed = (buffers.image_normed.read(), IMAGE_TOKENS);
        buffers.project(ctx, &image.linear[0], normed, |row, col, value| {
            buffers
                .image_inner
                .set(row * IMAGE_HIDDEN + col, gelu_erf(value));
        });
        let inner = (buffers.image_inner.read(), IMAGE_TOKENS);
        buffers.project(ctx, &image.linear[1], inner, |row, col, value| {
            contexts.image.set(row * HIDDEN + col, value);
        });
        let scratch = buffers.row.get(ctx);
        for row in ctx.split(IMAGE_TOKENS, 1) {
            let projected = contexts.image.part(row * HIDDEN..(row + 1) * HIDDEN);
            scratch.clear();
            scratch.extend_from_slice(projected);
            image.norm_out.apply(scratch, IMAGE_NORM_EPS, projected);
        }
        ctx.barrier();
    }

    /// Branch `branch`'s context from `text` (T5Gemma2's states, rows of `TEXT_HIDDEN`).
    pub(crate) fn embed_text(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        (text, branch): (&[f32], Branch),
        contexts: &Contexts,
    ) {
        let rows = text.len() / TEXT_HIDDEN;
        buffers.project(ctx, &self.context[0], (text, rows), |row, col, value| {
            buffers
                .text_hidden
                .set(row * HIDDEN + col, gelu_tanh(value));
        });
        let hidden = (buffers.text_hidden.read(), rows);
        buffers.project(ctx, &self.context[1], hidden, |row, col, value| {
            contexts.text[branch as usize].set(row * HIDDEN + col, value);
        });
    }
}

/// `hidden` after the video's tokens = the call's image rows, then its branch's text rows.
pub(super) fn copy_context(ctx: &Ctx, buffers: &Buffers, call: &Call) {
    let contexts = call.contexts;
    let image_rows = contexts.image_rows();
    let image = contexts.image.read();
    let text = contexts.text[call.branch as usize].read();
    let tokens = call.patching.tokens();
    for row in ctx.split(image_rows + contexts.text_rows(call.branch), 1) {
        let source = match row.checked_sub(image_rows) {
            None => &image[row * HIDDEN..(row + 1) * HIDDEN],
            Some(text_row) => &text[text_row * HIDDEN..(text_row + 1) * HIDDEN],
        };
        let at = (tokens + row) * HIDDEN;
        buffers
            .blocks
            .hidden
            .part(at..at + HIDDEN)
            .copy_from_slice(source);
    }
    ctx.barrier();
}
