//! A first image as the image-to-video pipeline prepares it: resized to the video's size by
//! Pillow's Lanczos filter and scaled to [−1, 1] (diffusers' `VideoProcessor`); and SigLIP's
//! input from that frame, back to [0, 1], resized to 896 × 896 by PyTorch's antialiased bicubic
//! interpolation and normalized by mean and deviation 0.5 (transformers' `SiglipImageProcessor`),
//! cut into patches.

use media::image::{Filter, Pixels, Size, resize_bicubic_antialiased};

use crate::Error;
use crate::model::{IMAGE_CHANNELS, IMAGE_TOKENS, VISION_PATCH, VISION_SIDE};
use crate::session::check_size;

/// SigLIP's normalization: mean and deviation of every channel.
const SIGLIP_CENTER: f32 = 0.5;
const PATCH_PIXELS: usize = VISION_PATCH * VISION_PATCH;
const PATCHES_PER_SIDE: usize = VISION_SIDE / VISION_PATCH;

/// `image` (RGB or RGBA) resized to `size`, which the engines hold: pixel-major RGB,
/// `2 · (v / 255) − 1` in f32.
pub fn first_frame(image: &Pixels, size: Size) -> Result<Vec<f32>, Error> {
    check_size(size.width, size.height)?;
    let resized = image.rgb().resize(size, Filter::Lanczos);
    let scale = f32::from(u8::MAX);
    let scaled = resized.values.iter().map(|&value| f32::from(value));
    Ok(scaled.map(|value| 2.0 * (value / scale) - 1.0).collect())
}

/// SigLIP's input of `frame` (`first_frame`'s, of `size`): 896 × 896 pixels, pixel-major RGB.
pub fn vision_pixels(frame: &[f32], size: Size) -> Vec<f32> {
    let unit: Vec<f32> = frame
        .iter()
        .map(|value| ((value + 1.0) / 2.0).clamp(0.0, 1.0))
        .collect();
    let side = Size {
        width: VISION_SIDE,
        height: VISION_SIDE,
    };
    let mut pixels = resize_bicubic_antialiased(&unit, size, side, IMAGE_CHANNELS);
    for value in &mut pixels {
        *value = (*value - SIGLIP_CENTER) / SIGLIP_CENTER;
    }
    pixels
}

/// `pixels` (`vision_pixels`') as `IMAGE_TOKENS` rows of a patch's values, channel-major, then
/// row-major within the patch, as the patch convolution's weights read them.
pub(crate) fn patches(pixels: &[f32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(IMAGE_TOKENS * IMAGE_CHANNELS * PATCH_PIXELS);
    for token in 0..IMAGE_TOKENS {
        push_patch(pixels, token, &mut values);
    }
    values
}

fn push_patch(pixels: &[f32], token: usize, values: &mut Vec<f32>) {
    let first_row = token / PATCHES_PER_SIDE * VISION_PATCH;
    let first_column = token % PATCHES_PER_SIDE * VISION_PATCH;
    for channel in 0..IMAGE_CHANNELS {
        for offset in 0..PATCH_PIXELS {
            let row = first_row + offset / VISION_PATCH;
            let column = first_column + offset % VISION_PATCH;
            values.push(pixels[(row * VISION_SIDE + column) * IMAGE_CHANNELS + channel]);
        }
    }
}
