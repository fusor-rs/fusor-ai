//! An image as a vision-language model's processor gives it to the encoder: resized to whole
//! merged blocks within a pixel budget, and cut into patches.

use media::image::{Filter, Pixels, Size};
use serde::Serialize;

use super::Grid;

/// An image's patches, `grid.patches()` rows of a patch's values.
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

/// How a vision-language model's processor cuts an image into the encoder's patches: `patch` ×
/// `patch` pixels each, merged `merge × merge` into one token, the image resized with `filter`
/// to an area of `min_pixels` to `max_pixels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preprocessing {
    pub patch: usize,
    pub merge: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub filter: Filter,
}

impl Preprocessing {
    /// The most patches an image has.
    pub const fn max_patches(&self) -> usize {
        self.max_pixels / (self.patch * self.patch)
    }

    /// The language-model tokens of `patches` patches: one per merged block.
    pub const fn tokens(&self, patches: usize) -> usize {
        patches / (self.merge * self.merge)
    }

    /// Side of a merged block, in pixels: both sides of a resized image are multiples of it.
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }

    /// The last step of the reference processors' `smart_resize`: `height × width` rounded to
    /// whole merged blocks, then scaled into the pixel budget, with Python's rounding (half to
    /// even).
    pub fn whole_blocks(&self, height: f64, width: f64) -> Size {
        let factor = self.factor() as f64;
        let whole = |side: f64| (side / factor).round_ties_even() * factor;
        let (mut resized_height, mut resized_width) = (whole(height), whole(width));
        let area = height * width;
        if resized_height * resized_width > self.max_pixels as f64 {
            let beta = (area / self.max_pixels as f64).sqrt();
            let shrunk = |side: f64| factor.max((side / beta / factor).floor() * factor);
            (resized_height, resized_width) = (shrunk(height), shrunk(width));
        } else if resized_height * resized_width < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / area).sqrt();
            let grown = |side: f64| (side * beta / factor).ceil() * factor;
            (resized_height, resized_width) = (grown(height), grown(width));
        }
        Size {
            width: resized_width as usize,
            height: resized_height as usize,
        }
    }

    /// An RGB or RGBA `image` as RGB, resized to `size` (whole patches) and cut into patches,
    /// row-major, each channel-major, every 8-bit value through `scaled`.
    pub fn patches(&self, image: &Pixels, size: Size, scaled: impl Fn(u8) -> f32) -> Patches {
        let rgb = image.rgb().resize(size, self.filter);
        let patch = self.patch;
        let grid = Grid {
            rows: size.height / patch,
            columns: size.width / patch,
        };
        let channels = rgb.channels;
        let pixel = |y: usize, x: usize, channel: usize| {
            scaled(rgb.values[(y * size.width + x) * channels + channel])
        };
        let mut values = Vec::with_capacity(grid.patches() * channels * patch * patch);
        for index in 0..grid.patches() {
            let (top, left) = (index / grid.columns * patch, index % grid.columns * patch);
            for channel in 0..channels {
                for y in top..top + patch {
                    values.extend((left..left + patch).map(|x| pixel(y, x, channel)));
                }
            }
        }
        Patches { grid, values }
    }

    /// How `image` is read as patches of `grid`.
    pub fn layout(&self, image: &Pixels, grid: Grid) -> Layout {
        Layout {
            width: image.size.width,
            height: image.size.height,
            resized_width: grid.columns * self.patch,
            resized_height: grid.rows * self.patch,
            rows: grid.rows,
            columns: grid.columns,
        }
    }
}
