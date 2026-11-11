//! What a session may ask for, and the checks of the sizes the engine's buffers hold.

use media::image::{Pixels, Size};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::model::{
    IMAGE_CHANNELS, MAX_PIXELS, MAX_SIDE, PATCH, RGBA_CHANNELS, SIZES, TILE, Variant,
};
use crate::restore::{self, Work};

/// A generation is one of `SIZES`. The stages alone (`ImageEngine::sample`, `predict`) take sides
/// that are multiples of `PATCH` from `MIN_SIDE` to `MAX_SIDE`, an area of at most `MAX_PIXELS`.
/// The buffers are sized for these: with the bundle they fit a wasm32 memory (4 GiB).
const MIN_SIDE: usize = 256;
const MAX_STEPS: usize = 100;
const MAX_GUIDANCE: f32 = 20.0;
const MAX_SHIFT: f32 = 20.0;
/// A photo in: at most this many pixels (depth's and the restorer's inputs).
const MAX_PHOTO_PIXELS: usize = 4096 * 4096;
/// The work image a restoration cuts into tiles, padded, at most 2048 × 2048: its full-size planes
/// and the restored image's fit beside the model.
const MAX_RESTORED_PIXELS: usize = 2048 * 2048;

/// What to generate. Turbo reads the sizes and the seed only.
#[derive(Debug, Clone, Deserialize)]
pub struct Generation {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Classifier-free guidance (base): at 1 only the prompt's branch runs, as the reference.
    pub guidance: f32,
    pub shift: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DepthRequest {
    /// The longest side the model sees; 0 keeps the photo's.
    pub max_side: usize,
}

/// The budget the photo is shrunk to before it is restored: its short and its long side.
#[derive(Debug, Clone, Deserialize)]
pub struct Upscale {
    pub short_side: usize,
    pub long_side: usize,
}

/// A started job: its steps (denoising steps, one for the depth pass, or restored tiles) and the
/// size of its image.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub steps: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub step: usize,
    pub steps: usize,
    /// Every step is done (`take_image`).
    pub done: bool,
}

impl Generation {
    /// # Errors
    /// When the size is outside the limits, or (base) the steps, guidance or shift are.
    pub(super) fn check_base(&self) -> Result<(), Error> {
        if !(1..=MAX_STEPS).contains(&self.steps) {
            return Err(Error::Request(format!(
                "{} steps: 1 to {MAX_STEPS}",
                self.steps
            )));
        }
        if !(1.0..=MAX_GUIDANCE).contains(&self.guidance) {
            return Err(Error::Request(format!(
                "guidance {}: 1 (the prompt alone) to {MAX_GUIDANCE}",
                self.guidance
            )));
        }
        if !(1.0..=MAX_SHIFT).contains(&self.shift) {
            return Err(Error::Request(format!(
                "shift {}: 1 to {MAX_SHIFT}",
                self.shift
            )));
        }
        self.check_size()
    }

    /// Whether a run on `variant` guides the prompt's branch by the negative prompt's: the base
    /// model, at a guidance past 1.
    pub fn guided(&self, variant: Variant) -> bool {
        variant == Variant::Base && self.guidance != 1.0
    }

    pub(super) fn size(&self) -> Size {
        Size {
            width: self.width,
            height: self.height,
        }
    }

    pub(super) fn check_size(&self) -> Result<(), Error> {
        check_size(self.size(), MIN_SIDE)
    }

    /// # Errors
    /// Unless the size is one of `SIZES`, the ones the model was trained to generate.
    pub(super) fn check_trained_size(&self) -> Result<(), Error> {
        if SIZES.contains(&self.size()) {
            return Ok(());
        }
        let sizes: Vec<String> = SIZES
            .iter()
            .map(|size| format!("{}×{}", size.width, size.height))
            .collect();
        Err(Error::Request(format!(
            "{}×{}: Iris-3B generates {}",
            self.width,
            self.height,
            sizes.join(", ")
        )))
    }
}

/// Whether `image` holds the RGB pixels of `size`.
pub(super) fn check_image(image: &[f32], size: Size) -> Result<(), Error> {
    if image.len() == size.width * size.height * IMAGE_CHANNELS {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{} image values for {}×{} RGB pixels",
        image.len(),
        size.width,
        size.height
    )))
}

/// Whether an image of `size` is within the sides (multiples of `PATCH` from `min_side`) and
/// area the buffers hold.
pub(super) fn check_size(size: Size, min_side: usize) -> Result<(), Error> {
    let Size { width, height } = size;
    let side =
        |length: usize| (min_side..=MAX_SIDE).contains(&length) && length.is_multiple_of(PATCH);
    if side(width) && side(height) && width * height <= MAX_PIXELS {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{width}×{height}: sides are multiples of {PATCH} from {min_side} to {MAX_SIDE}, at most \
         {MAX_PIXELS} pixels"
    )))
}

/// The photo of `rgba` (`width` pixels wide), as RGB.
///
/// # Errors
/// When the bytes are not `width × height` RGBA pixels, or there are more than a photo may have.
pub fn photo(rgba: &[u8], width: usize, height: usize) -> Result<Pixels, Error> {
    let bytes = width
        .checked_mul(height)
        .filter(|&pixels| pixels > 0 && pixels <= MAX_PHOTO_PIXELS)
        .and_then(|pixels| pixels.checked_mul(RGBA_CHANNELS));
    if bytes != Some(rgba.len()) {
        return Err(Error::Request(format!(
            "{} bytes are not {width}×{height} RGBA pixels (1 to {MAX_PHOTO_PIXELS})",
            rgba.len()
        )));
    }
    let image = Pixels {
        size: Size { width, height },
        channels: RGBA_CHANNELS,
        values: rgba.to_vec(),
    };
    Ok(image.rgb())
}

/// Whether `photo` is one the session reads: RGB, within `MAX_PHOTO_PIXELS`.
pub(super) fn check_photo(photo: &Pixels) -> Result<(), Error> {
    let Size { width, height } = photo.size;
    let expected = width * height * IMAGE_CHANNELS;
    if photo.channels == IMAGE_CHANNELS
        && width * height <= MAX_PHOTO_PIXELS
        && photo.values.len() == expected
    {
        return Ok(());
    }
    Err(Error::Request(format!(
        "a photo of {width}×{height} RGB pixels, at most {MAX_PHOTO_PIXELS}"
    )))
}

/// The work image of a restoration's output of `size`, when its padded size (the largest plane
/// the restoration allocates) is within `MAX_RESTORED_PIXELS`.
pub(super) fn check_restored(size: Size) -> Result<Work, Error> {
    let within = |work: &Work| {
        let Size { width, height } = work.padded;
        width
            .checked_mul(height)
            .is_some_and(|pixels| pixels <= MAX_RESTORED_PIXELS)
    };
    restore::layout(size).filter(within).ok_or_else(|| {
        let Size { width, height } = size;
        Error::Request(format!(
            "{width}×{height} pixels to restore: a work image of at most \
             {MAX_RESTORED_PIXELS} pixels; lower the budget"
        ))
    })
}

/// Whether `values` count the RGB pixels of a work image of `size` that the tiles cut: at least
/// `TILE` along each side, a multiple of `PATCH`, within `MAX_RESTORED_PIXELS`.
pub(super) fn check_tiled(values: usize, size: Size) -> Result<(), Error> {
    let Size { width, height } = size;
    let side = |length: usize| length >= TILE && length.is_multiple_of(PATCH);
    if side(width)
        && side(height)
        && width * height <= MAX_RESTORED_PIXELS
        && values == width * height * IMAGE_CHANNELS
    {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{values} work values for {width}×{height}: sides multiples of {PATCH} from {TILE}, at \
         most {MAX_RESTORED_PIXELS} pixels"
    )))
}
