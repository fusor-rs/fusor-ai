//! The image's rotary embeddings: Iris's isotropic 2-D table (`rope_2d`).

use blocks::rotary::{Position, Rotary, Rotation};
use kernels::ops::rope_inv_freq;

use super::{HEAD_DIM, PAIRS, Pass, ROPE_BASE, ROPE_SPAN};
use crate::model::{PATCH, TEXT_ROWS};

/// cos and sin of each rotary pair of every text row, `PAIRS` per row: the 1-D embedding at
/// positions `0..TEXT_ROWS`.
pub(super) struct TextAngles {
    pub(super) cos: Vec<f32>,
    pub(super) sin: Vec<f32>,
}

impl TextAngles {
    pub(super) fn new() -> Self {
        let rotary = Rotary::new(Rotation::Sequence, ROPE_BASE, HEAD_DIM);
        let mut cos = vec![0.0; TEXT_ROWS * PAIRS];
        let mut sin = vec![0.0; TEXT_ROWS * PAIRS];
        let rows = cos
            .as_chunks_mut::<PAIRS>()
            .0
            .iter_mut()
            .zip(sin.as_chunks_mut::<PAIRS>().0);
        for (row, (row_cos, row_sin)) in rows.enumerate() {
            rotary.angles(Position::uniform(row), row_cos, row_sin);
        }
        Self { cos, sin }
    }
}

/// cos and sin of each rotary pair of every patch of the image, `PAIRS` per patch: pair `2j`
/// turned by the column, `2j + 1` by the row, at frequency `ROPE_BASE^(−4j/HEAD_DIM)`, positions
/// `index · ROPE_SPAN / (max(rows, columns) − 1)`, in f32. The patch head's attention has the
/// same head width, so one table serves both. By patches, no barrier.
pub(super) fn image_angles(pass: &Pass) {
    let video = pass.patching.video;
    let longest = (video.height / PATCH).max(video.width / PATCH);
    let step = (f64::from(ROPE_SPAN) / (longest - 1).max(1) as f64) as f32;
    let frequencies = rope_inv_freq(ROPE_BASE, HEAD_DIM / 2);
    for patch in pass.ctx.split(pass.patches(), 1) {
        let position = pass.patching.position(patch);
        let coordinates = [position.width as f32 * step, position.height as f32 * step];
        let pairs = patch * PAIRS..(patch + 1) * PAIRS;
        let cos = pass.buffers.image_cos.part(pairs.clone());
        let sin = pass.buffers.image_sin.part(pairs);
        for (turn, frequency) in frequencies.iter().enumerate() {
            for (axis, coordinate) in coordinates.into_iter().enumerate() {
                let angle = coordinate * frequency;
                cos[2 * turn + axis] = angle.cos();
                sin[2 * turn + axis] = angle.sin();
            }
        }
    }
}
