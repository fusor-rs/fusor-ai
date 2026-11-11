//! Iris's depth predictor around the transformer (`DepthPredictor`): the photo resized to a
//! multiple of the patch, scaled to [−1, 1], and the prediction resized back to the photo.

use media::image::{Filter, Pixels, Size, resize_bilinear};

use crate::model::PATCH;

/// The size the depth model sees: each side `max(PATCH, round(side · scale / PATCH) · PATCH)`,
/// `scale = min(1, max_side / longest side)` (1 when `max_side` is 0); Python's rounding, half
/// to even.
pub fn model_size(photo: Size, max_side: usize) -> Size {
    let longest = photo.width.max(photo.height);
    let scale = if max_side == 0 {
        1.0
    } else {
        (max_side as f64 / longest as f64).min(1.0)
    };
    let side = |length: usize| {
        let patches = (length as f64 * scale / PATCH as f64).round_ties_even() as usize;
        PATCH.max(patches * PATCH)
    };
    Size {
        width: side(photo.width),
        height: side(photo.height),
    }
}

/// The photo (RGB) resized with Pillow's Lanczos to `size` when it differs, as `v / 255 · 2 − 1`
/// (f32), pixel-major.
pub fn input(photo: &Pixels, size: Size) -> Vec<f32> {
    let resized;
    let pixels = if photo.size == size {
        photo
    } else {
        resized = photo.resize(size, Filter::Lanczos);
        &resized
    };
    pixels
        .values
        .iter()
        .map(|&value| f32::from(value) / 255.0 * 2.0 - 1.0)
        .collect()
}

/// The prediction (`from`, one value per pixel) resized to the photo (`to`) as
/// `F.interpolate(mode="bilinear", align_corners=False)` does when the sizes differ.
pub fn upsample(prediction: &[f32], from: Size, to: Size) -> Vec<f32> {
    if from == to {
        return prediction.to_vec();
    }
    resize_bilinear(prediction, from, to, 1)
}
