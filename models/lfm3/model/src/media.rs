//! An image as LFM3-VL-2B's processor reads it (`Preprocessing::centered_patches`): resized to
//! whole 2 × 2 blocks of 16-pixel patches at 65,536 to 1,638,400 pixels, every value scaled to
//! [−1, 1], each patch its frame twice.

use blocks::vit::Preprocessing;
use blocks::vit::preprocessing::Patches;
use media::image::{Filter, Pixels};

use crate::Error;
use crate::model::{MERGE, PATCH};

/// The checkpoint's image budget, its 16.8 million pixels lowered to 1,600 tokens.
pub const IMAGE: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 65_536,
    max_pixels: 1_638_400,
    filter: Filter::Bicubic,
};
/// The widest aspect ratio the processor accepts.
const MAX_ASPECT: usize = 200;
/// Values of one frame of one channel of a patch.
const PLANE: usize = PATCH * PATCH;

/// The two-frame patches of an RGB or RGBA image.
///
/// # Errors
/// When the image is empty, or wider or taller than 200 times its other side.
pub fn patches(image: &Pixels) -> Result<Patches, Error> {
    let patches = IMAGE.centered_patches(image, MAX_ASPECT)?;
    Ok(patches.doubled(PLANE))
}
