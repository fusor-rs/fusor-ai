//! What a generation may ask for, and the checks of the sizes the engine's buffers hold.

use blocks::video::Patching;
use blocks::wan_vae::Capacity;
use serde::Deserialize;

use crate::model::{FRAME_STEP, HEIGHT, LATENT_FACTOR, PATCH, WIDTH};
use crate::{Error, Video};

/// A generation is `WIDTH × HEIGHT`, its frames `FRAME_STEP·n + 1` up to `MAX_FRAMES`, and with a
/// first image at least `MIN_IMAGE_FRAMES` (the pipeline conditions only videos of two latent
/// frames or more). `MAX_FRAMES` keeps the transformer's rows beside the video bundle (2413 MB)
/// inside a wasm32 memory (4 GiB): its buffers take 1281 MB at 5 frames, 1882 MB at 9.
pub const MAX_FRAMES: usize = 5;
pub(super) const MIN_IMAGE_FRAMES: usize = FRAME_STEP + 1;
const MAX_STEPS: usize = 100;
const MAX_GUIDANCE: f32 = 20.0;
/// The stages alone (`VideoEngine::predict` and the others, the text engine's first frame) take
/// any size within the buffers: sides multiples of `SIDE_STEP` (the latents' 8 times the patch's
/// 2) from `MIN_SIDE`, at most `WIDTH` wide and `WIDTH × HEIGHT` pixels.
const SIDE_STEP: usize = 16;
const MIN_SIDE: usize = 64;
pub(super) const MAX_LATENT_PIXELS: usize = WIDTH * HEIGHT / (LATENT_FACTOR * LATENT_FACTOR);
pub(super) const MAX_TOKENS: usize =
    latent_frames(MAX_FRAMES) * MAX_LATENT_PIXELS / (PATCH * PATCH);
/// Latent rows the decoder's levels decode at a time. At 4 its buffers take 1.4 GB and the
/// attention's scratch, which a wasm32 memory holds beside the video bundle once the
/// transformer's are dropped; each strip computes 13 more latent rows above and below.
const DECODER_STRIP: usize = 4;
pub(super) const DECODER: Capacity = Capacity {
    pixels: MAX_LATENT_PIXELS,
    width: WIDTH / LATENT_FACTOR,
    video: MAX_TOKENS * PATCH * PATCH,
    strip: DECODER_STRIP,
};

/// What to generate.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: usize,
    /// Adaptive projected guidance's scale, 0 to `MAX_GUIDANCE`; at 0 only the prompt's branch
    /// runs.
    pub guidance: f32,
    pub seed: u64,
}

impl Request {
    /// The latents of the requested video, if the request is within the limits.
    pub fn video(&self) -> Result<Video, Error> {
        check(self)?;
        Ok(Video {
            frames: latent_frames(self.frames),
            height: self.height / LATENT_FACTOR,
            width: self.width / LATENT_FACTOR,
        })
    }
}

/// Latent frames of a video of `frames` frames.
pub(super) const fn latent_frames(frames: usize) -> usize {
    (frames - 1) / FRAME_STEP + 1
}

/// Whether a frame of `width × height` pixels is within the sides and area the engines hold.
pub(crate) fn check_size(width: usize, height: usize) -> Result<(), Error> {
    let side = |length: usize| length >= MIN_SIDE && length.is_multiple_of(SIDE_STEP);
    if side(width) && side(height) && width <= WIDTH && width * height <= WIDTH * HEIGHT {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{width}×{height}: sides are multiples of {SIDE_STEP} from {MIN_SIDE}, at most {WIDTH} \
         wide and {} pixels",
        WIDTH * HEIGHT
    )))
}

/// `values` (`channels` per pixel) of `video` the buffers hold.
pub(super) fn check_values(values: &[f32], video: Video, channels: usize) -> Result<(), Error> {
    let latent_step = SIDE_STEP / LATENT_FACTOR;
    let sides = video.height.is_multiple_of(latent_step) && video.width.is_multiple_of(latent_step);
    let tokens = Patching {
        video,
        patch: PATCH,
    }
    .tokens();
    if sides
        && video.frames > 0
        && video.height > 0
        && video.width > 0
        && video.width <= DECODER.width
        && video.height * video.width <= MAX_LATENT_PIXELS
        && tokens <= MAX_TOKENS
        && values.len() == video.pixels() * channels
    {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{} values of {channels} channels for {} frames of {}×{}: sides multiples of \
         {latent_step}, at most {} wide, {MAX_LATENT_PIXELS} pixels a frame and {MAX_TOKENS} \
         tokens",
        values.len(),
        video.frames,
        video.width,
        video.height,
        DECODER.width
    )))
}

fn check(request: &Request) -> Result<(), Error> {
    if (request.width, request.height) != (WIDTH, HEIGHT) {
        return Err(Error::Request(format!(
            "{}×{}: Motif-Video-2B generates {WIDTH}×{HEIGHT} videos only",
            request.width, request.height
        )));
    }
    if !(1..=MAX_FRAMES).contains(&request.frames)
        || !(request.frames - 1).is_multiple_of(FRAME_STEP)
    {
        return Err(Error::Request(format!(
            "{} frames: {FRAME_STEP}·n + 1, up to {MAX_FRAMES}",
            request.frames
        )));
    }
    if !(1..=MAX_STEPS).contains(&request.steps) {
        return Err(Error::Request(format!(
            "{} steps: 1 to {MAX_STEPS}",
            request.steps
        )));
    }
    if !(0.0..=MAX_GUIDANCE).contains(&request.guidance) {
        return Err(Error::Request(format!(
            "guidance {}: 0 (the prompt alone) to {MAX_GUIDANCE}",
            request.guidance
        )));
    }
    Ok(())
}
