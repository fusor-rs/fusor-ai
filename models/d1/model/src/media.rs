//! A request's media as d1's reference reads them. An image is scaled down (Pillow's bicubic) to
//! at most the checkpoint's pixel budget (`d1_vision_max_pixels`), then as LFM3-VL's processor
//! reads it (`Preprocessing::centered_patches`): resized to whole 2 × 2 blocks of 16-pixel
//! patches, every value scaled to [−1, 1], each patch its frame twice. A clip, mono at 16 kHz,
//! is padded to whole windows of 400 samples every 160 and read as 128 log-mel features: the
//! power through the Slaney filterbank, its base-10 logarithm floored at −12, plus 4, over 4.

use blocks::vit::Preprocessing;
use blocks::vit::preprocessing::Patches;
use media::image::{Filter, Pixels, Size};
use media::mel::{self, MelSpectrogram};

use crate::Error;
use crate::model::{MELS, MERGE, PATCH};

/// The processor's budget: an image within the checkpoint's is never shrunk by it.
pub const IMAGE: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 65_536,
    max_pixels: 2_097_152,
    filter: Filter::Bicubic,
};
/// The widest aspect ratio the processor accepts.
const MAX_ASPECT: usize = 200;
/// Values of one frame of one channel of a patch.
const PLANE: usize = PATCH * PATCH;

/// The sample rate a clip is read at.
pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL: mel::Shape = mel::Shape {
    sample_rate: SAMPLE_RATE,
    fft: 400,
    hop: 160,
    mels: MELS,
};
const LOG_FLOOR: f32 = -12.0;
const OFFSET: f32 = 4.0;
const DIVISOR: f32 = 4.0;
/// The longest clip, in seconds.
pub const MAX_SECONDS: usize = 60;
/// The mel frames of the longest clip.
pub const MAX_MEL_FRAMES: usize = MAX_SECONDS * SAMPLE_RATE as usize / MEL.hop + 1;

/// The two-frame patches of an RGB or RGBA image, scaled down to at most `max_pixels` first.
///
/// # Errors
/// When the image is empty, or wider or taller than 200 times its other side.
pub fn patches(image: &Pixels, max_pixels: usize) -> Result<Patches, Error> {
    let rgb = image.rgb();
    let Size { width, height } = rgb.size;
    let pixels = width * height;
    let capped = if pixels > max_pixels {
        let scale = (max_pixels as f64 / pixels as f64).sqrt();
        let side = |length: usize| ((length as f64 * scale) as usize).max(1);
        let size = Size {
            width: side(width),
            height: side(height),
        };
        rgb.resize(size, Filter::Bicubic)
    } else {
        rgb
    };
    let patches = IMAGE.centered_patches(&capped, MAX_ASPECT)?;
    Ok(patches.doubled(PLANE))
}

/// A clip's features, `frames × mels` row-major.
///
/// # Errors
/// When the clip is empty or longer than `MAX_SECONDS`.
pub fn features(spectrogram: &MelSpectrogram, samples: &[f32]) -> Result<Vec<f32>, Error> {
    if samples.is_empty() || samples.len() > MAX_SECONDS * SAMPLE_RATE as usize {
        return Err(Error::Request(format!(
            "a clip has {} samples: it holds 1 to {MAX_SECONDS} s at {SAMPLE_RATE} Hz",
            samples.len()
        )));
    }
    let (fft, hop) = (MEL.fft, MEL.hop);
    let padded = match samples.len().checked_sub(fft) {
        None => fft,
        Some(past) => samples.len() + (hop - past % hop) % hop,
    };
    let mut clip = samples.to_vec();
    clip.resize(padded, 0.0);
    let mut features = Vec::new();
    spectrogram.power(&clip, &mut features);
    for value in &mut features {
        *value = (value.log10().max(LOG_FLOOR) + OFFSET) / DIVISOR;
    }
    Ok(features)
}
