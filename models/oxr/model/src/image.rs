//! The image as the model reads it: as RGB, resized with Pillow's bilinear filter so that both
//! sides are whole merged patches and the area is within the pixel budget, scaled to [−1, 1],
//! and cut into 16 × 16 patches in row-major order, each channel-major.

use blocks::vit::Preprocessing;
use blocks::vit::preprocessing::Patches;
use media::image::{Filter, Pixels, Size};

use crate::Error;
use crate::model::{MERGE, PATCH};

/// The fewest pixels an image is resized to: 196 language-model tokens.
pub const MIN_PIXELS: usize = 200_704;
/// The most: 900 tokens, the budget SoMark serves the model with.
pub const MAX_PIXELS: usize = 921_600;
pub(crate) const PREPROCESSING: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: MIN_PIXELS,
    max_pixels: MAX_PIXELS,
    filter: Filter::Bilinear,
};
/// The widest aspect ratio the preprocessing keeps; a wider image is resized to it.
const MAX_ASPECT: f64 = 200.0;
/// The image processor's `rescale_factor`, applied in f64 before the f32 normalization.
const RESCALE: f64 = 1.0 / 255.0;
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;

/// The patches of an RGB or RGBA image.
pub fn patches(image: &Pixels) -> Result<Patches, Error> {
    let size = resized(image.size)?;
    let scaled = |value: u8| ((f64::from(value) * RESCALE) as f32 - MEAN) / STD;
    Ok(PREPROCESSING.patches(image, size, scaled))
}

/// `smart_resize` of the reference's processor, with Python's rounding (half to even).
fn resized(size: Size) -> Result<Size, Error> {
    let factor = PREPROCESSING.factor() as f64;
    let (mut height, mut width) = (size.height as f64, size.width as f64);
    if height == 0.0 || width == 0.0 {
        return Err(Error::Request("the image is empty".into()));
    }
    if height < factor || width < factor {
        if height < width {
            width = (factor / height * width).round_ties_even();
            height = factor;
        } else {
            height = (factor / width * height).round_ties_even();
            width = factor;
        }
    } else if height.max(width) / height.min(width) > MAX_ASPECT {
        if height > width {
            height = MAX_ASPECT * width;
        } else {
            width = MAX_ASPECT * height;
        }
    }
    Ok(PREPROCESSING.whole_blocks(height, width))
}
