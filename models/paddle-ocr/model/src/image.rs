//! The image as the model reads it: as RGB, resized with Pillow's bicubic filter
//! so that both sides are whole merged patches and the area is within the pixel budget, scaled
//! to [−1, 1], and cut into 14 × 14 patches in row-major order, each channel-major.

use blocks::vit::Preprocessing;
use blocks::vit::preprocessing::Patches;
use media::image::{Filter, Pixels, Size};

use crate::Error;
use crate::model::{MERGE, PATCH};

/// The fewest pixels an image is resized to: 144 language-model tokens.
pub const MIN_PIXELS: usize = 112_896;
/// The most: 1280 tokens, the model card's budget for every task but spotting.
pub const MAX_PIXELS: usize = 1_003_520;
pub(crate) const PREPROCESSING: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: MIN_PIXELS,
    max_pixels: MAX_PIXELS,
    filter: Filter::Bicubic,
};
/// The widest aspect ratio the preprocessing accepts.
const MAX_ASPECT: f64 = 200.0;

/// The patches of an RGB or RGBA image.
pub fn patches(image: &Pixels) -> Result<Patches, Error> {
    let size = resized(image.size)?;
    let scaled = |value: u8| (f32::from(value) / f32::from(u8::MAX) - 0.5) / 0.5;
    Ok(PREPROCESSING.patches(image, size, scaled))
}

/// `smart_resize` of the reference image processor, with Python's rounding (half to even).
fn resized(size: Size) -> Result<Size, Error> {
    let factor = PREPROCESSING.factor() as f64;
    let (mut height, mut width) = (size.height as f64, size.width as f64);
    if height == 0.0 || width == 0.0 {
        return Err(Error::Request("the image is empty".into()));
    }
    if height < factor {
        width = (width * factor / height).round_ties_even();
        height = factor;
    }
    if width < factor {
        height = (height * factor / width).round_ties_even();
        width = factor;
    }
    if height.max(width) / height.min(width) > MAX_ASPECT {
        return Err(Error::Request(format!(
            "a {}×{} image: the sides must be within {MAX_ASPECT}:1",
            size.width, size.height
        )));
    }
    Ok(PREPROCESSING.whole_blocks(height, width))
}
