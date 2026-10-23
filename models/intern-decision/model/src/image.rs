//! An image as the model reads it: as RGB, resized with a bicubic filter so that both sides are
//! whole merged patches and the area is within the pixel budget, scaled to [−1, 1], and cut into
//! 16 × 16 patches in row-major order, each channel-major.

use blocks::vit::Preprocessing;
use blocks::vit::preprocessing::Patches;
use media::image::{Filter, Pixels, Size};

use crate::Error;
use crate::model::{MERGE, PATCH};

/// The fewest pixels an image is resized to: 64 language-model tokens (the checkpoint's own).
pub const MIN_PIXELS: usize = 65_536;
/// The most: 1024 tokens, so that an image leaves room in `MAX_ROWS` for the request's text.
pub const MAX_PIXELS: usize = 1_048_576;
pub(crate) const PREPROCESSING: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: MIN_PIXELS,
    max_pixels: MAX_PIXELS,
    filter: Filter::Bicubic,
};
/// The widest aspect ratio the processor accepts.
const MAX_ASPECT: usize = 200;
/// The processor's mean and deviation, scaled to 8-bit values: it normalizes `(x − 127.5) /
/// 127.5` in f32, its rescaling folded into them.
const CENTER: f32 = 127.5;

/// The patches of an RGB or RGBA image.
pub fn patches(image: &Pixels) -> Result<Patches, Error> {
    let Size { width, height } = image.size;
    if width == 0 || height == 0 {
        return Err(Error::Request("an image is empty".into()));
    }
    if width.max(height) > MAX_ASPECT * width.min(height) {
        return Err(Error::Request(format!(
            "an image is {width}×{height}: its aspect ratio must be at most {MAX_ASPECT}"
        )));
    }
    let size = PREPROCESSING.whole_blocks(height as f64, width as f64);
    Ok(PREPROCESSING.patches(image, size, |value| (f32::from(value) - CENTER) / CENTER))
}
