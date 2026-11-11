//! The image's embeddings: its patches as vectors, and its pixels as the pixel head's tokens
//! (a linear of the colour plus a fixed sine-cosine table of the position).

use kernels::Linear;

use super::rope::image_angles;
use super::{PATCH_PIXELS, Pass, SINUSOID_PAIRS};
use crate::model::{IMAGE_CHANNELS, PATCH, PIXEL_HIDDEN};

/// The table's values along one axis: a sine and a cosine of each frequency.
const AXIS_VALUES: usize = 2 * SINUSOID_PAIRS;

/// `grouped` = each patch's pixels, and the image's rotary angles and sinusoid tables; then a
/// barrier.
pub(super) fn group(pass: &Pass) {
    let Pass { ctx, buffers, .. } = pass;
    let image = (buffers.image.read(), IMAGE_CHANNELS);
    let values = PATCH_PIXELS * IMAGE_CHANNELS;
    for patch in ctx.split(pass.patches(), 1) {
        let grouped = buffers.grouped.part(patch * values..(patch + 1) * values);
        pass.patching.gather(image, patch, grouped);
    }
    image_angles(pass);
    if ctx.is_first() {
        let video = pass.patching.video;
        sinusoids(buffers.columns.part(0..video.width * AXIS_VALUES));
        sinusoids(buffers.rows.part(0..video.height * AXIS_VALUES));
    }
    ctx.barrier();
}

/// `table[position · AXIS_VALUES ..]` = `[sin(position·ω_k), cos(position·ω_k)]` for
/// `ω_k = blocks::SINUSOID_BASE^(−k / SINUSOID_PAIRS)`, in f64 then f32.
fn sinusoids(table: &mut [f32]) {
    let base = f64::from(blocks::SINUSOID_BASE);
    for (position, values) in table
        .as_chunks_mut::<AXIS_VALUES>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (sines, cosines) = values.split_at_mut(SINUSOID_PAIRS);
        for (turn, (sine, cosine)) in sines.iter_mut().zip(cosines).enumerate() {
            let omega = 1.0 / base.powf(turn as f64 / SINUSOID_PAIRS as f64);
            let angle = position as f64 * omega;
            *sine = angle.sin() as f32;
            *cosine = angle.cos() as f32;
        }
    }
}

/// `pixels` = every pixel's embedding: its colour through `embed` plus the table's values at its
/// position (the columns' sines and cosines, then the rows'); then a barrier.
pub(super) fn embed_pixels(pass: &Pass, embed: &Linear) {
    let Pass {
        ctx,
        buffers,
        patching,
        ..
    } = pass;
    let (columns, rows) = (buffers.columns.read(), buffers.rows.read());
    let pixels = patching.video.pixels();
    let grouped = buffers.grouped.read();
    buffers.project(ctx, embed, (grouped, pixels), |row, col, value| {
        let (patch, slot) = (row / PATCH_PIXELS, row % PATCH_PIXELS);
        let position = patching.position(patch);
        let x = position.width * PATCH + slot % PATCH;
        let y = position.height * PATCH + slot / PATCH;
        let table = match col.checked_sub(AXIS_VALUES) {
            None => columns[x * AXIS_VALUES + col],
            Some(offset) => rows[y * AXIS_VALUES + offset],
        };
        buffers.pixels.set(row * PIXEL_HIDDEN + col, value + table);
    });
}
