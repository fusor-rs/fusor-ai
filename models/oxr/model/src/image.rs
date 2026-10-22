//! The image as the model reads it: as RGB, resized with Pillow's bilinear filter so that both
//! sides are whole merged patches and the area is within the pixel budget, scaled to [−1, 1],
//! and cut into 16 × 16 patches in row-major order, each channel-major.

use blocks::vit::Grid;
use kernels::image::{Filter, Pixels, Size};
use serde::Serialize;

use crate::Error;
use crate::model::{CHANNELS, MERGE, PATCH, PATCH_VALUES};

/// The fewest pixels an image is resized to: 196 language-model tokens.
pub const MIN_PIXELS: usize = 200_704;
/// The most: 900 tokens, the budget SoMark serves the model with.
pub const MAX_PIXELS: usize = 921_600;
/// Side of a merged patch, in pixels: both sides of a resized image are multiples of it.
const FACTOR: usize = PATCH * MERGE;
/// The widest aspect ratio the preprocessing keeps; a wider image is resized to it.
const MAX_ASPECT: f64 = 200.0;
/// The image processor's `rescale_factor`, applied in f64 before the f32 normalization.
const RESCALE: f64 = 1.0 / 255.0;
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;

/// An image's patches, `grid.patches() × PATCH_VALUES`.
pub struct Patches {
    pub grid: Grid,
    pub values: Vec<f32>,
}

/// How an image is read: its size, the size it was resized to and its patch grid.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub resized_width: usize,
    pub resized_height: usize,
    pub rows: usize,
    pub columns: usize,
}

impl Patches {
    /// The patches of an RGB or RGBA image.
    pub fn of(image: &Pixels) -> Result<Self, Error> {
        let size = resized(image.size)?;
        let rgb = image.rgb().resize(size, Filter::Bilinear);
        let grid = Grid {
            rows: size.height / PATCH,
            columns: size.width / PATCH,
        };
        let scaled = |value: u8| ((f64::from(value) * RESCALE) as f32 - MEAN) / STD;
        let pixel = |y: usize, x: usize, channel: usize| {
            scaled(rgb.values[(y * size.width + x) * CHANNELS + channel])
        };
        let mut values = Vec::with_capacity(grid.patches() * PATCH_VALUES);
        for patch in 0..grid.patches() {
            let (top, left) = (patch / grid.columns * PATCH, patch % grid.columns * PATCH);
            for channel in 0..CHANNELS {
                for y in top..top + PATCH {
                    values.extend((left..left + PATCH).map(|x| pixel(y, x, channel)));
                }
            }
        }
        Ok(Self { grid, values })
    }

    pub fn layout(&self, image: &Pixels) -> Layout {
        Layout {
            width: image.size.width,
            height: image.size.height,
            resized_width: self.grid.columns * PATCH,
            resized_height: self.grid.rows * PATCH,
            rows: self.grid.rows,
            columns: self.grid.columns,
        }
    }
}

/// `smart_resize` of the reference's processor, with Python's rounding (half to even).
fn resized(size: Size) -> Result<Size, Error> {
    let factor = FACTOR as f64;
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
    let whole = |side: f64| (side / factor).round_ties_even() * factor;
    let (mut resized_height, mut resized_width) = (whole(height), whole(width));
    let area = height * width;
    if resized_height * resized_width > MAX_PIXELS as f64 {
        let beta = (area / MAX_PIXELS as f64).sqrt();
        let shrunk = |side: f64| factor.max((side / beta / factor).floor() * factor);
        (resized_height, resized_width) = (shrunk(height), shrunk(width));
    } else if resized_height * resized_width < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / area).sqrt();
        let grown = |side: f64| (side * beta / factor).ceil() * factor;
        (resized_height, resized_width) = (grown(height), grown(width));
    }
    Ok(Size {
        width: resized_width as usize,
        height: resized_height as usize,
    })
}
