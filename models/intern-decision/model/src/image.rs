//! How the model reads an image (`Preprocessing::centered_patches`): as RGB, resized with a
//! bicubic filter so that both sides are whole merged patches and the area is within the pixel
//! budget, scaled to [−1, 1], and cut into 16 × 16 patches in row-major order, each
//! channel-major.

use blocks::vit::Preprocessing;
use media::image::Filter;

use crate::model::{MERGE, PATCH};

/// The fewest pixels an image is resized to: 64 language-model tokens (the checkpoint's own).
pub const MIN_PIXELS: usize = 65_536;
/// The most: 1024 tokens, so that an image leaves room in `MAX_ROWS` for the request's text.
pub const MAX_PIXELS: usize = 1_048_576;
/// The widest aspect ratio the processor accepts.
pub const MAX_ASPECT: usize = 200;
pub const PREPROCESSING: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: MIN_PIXELS,
    max_pixels: MAX_PIXELS,
    filter: Filter::Bicubic,
};
