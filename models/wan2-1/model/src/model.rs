//! The model's dimensions and tensor names. The text bundle holds umT5-XXL (`TEXT`) and its
//! tokenizer, the video bundle the diffusion transformer (`DIT`) and the autoencoder's decoder
//! (`VAE`); the compiler (`wan2-1-compile`) writes exactly these names.

use blocks::t5;
use blocks::wan_dit::{self, Dit};
use blocks::wan_vae::{self, Decoder};
use bundle::{Load, Store};

use crate::Error;

/// The directories of the two bundles under a bundle root.
pub const TEXT_BUNDLE: &str = "text";
pub const VIDEO_BUNDLE: &str = "video";
pub const TEXT: &str = "text";
pub const DIT: &str = "video.dit";
pub const VAE: &str = "video.vae";

/// umT5-XXL's encoder.
pub const TEXT_SHAPE: t5::Shape = t5::Shape {
    hidden: 4096,
    layers: 24,
    heads: 64,
    head_dim: 64,
    feed_forward: 10_240,
    vocabulary: 256_384,
    buckets: 32,
    max_distance: 128,
    norm_eps: 1e-6,
};
/// Tokens umT5 reads: the prompt's, then `</s>`.
pub const TEXT_TOKENS: usize = 512;
pub const END: u32 = 1;

/// Wan2.1's 1.3B diffusion transformer.
pub const DIT_SHAPE: wan_dit::Shape = wan_dit::Shape {
    hidden: 1536,
    heads: 12,
    feed_forward: 8960,
    depth: 30,
    latent: LATENT_CHANNELS,
    patch: 2,
    text: TEXT_SHAPE.hidden,
    text_tokens: TEXT_TOKENS,
    frequencies: 256,
    rope_sections: [22, 21, 21],
    rope_base: 10_000.0,
    eps: 1e-6,
};
/// The causal video autoencoder's decoder.
pub const VAE_SHAPE: wan_vae::Shape = wan_vae::Shape {
    latent: LATENT_CHANNELS,
    channels: &[384, 384, 192, 96],
    blocks: 3,
    temporal: &[true, true, false],
    outputs: IMAGE_CHANNELS,
    kernel: 3,
};
pub const LATENT_CHANNELS: usize = 16;
/// Red, green, blue.
pub const IMAGE_CHANNELS: usize = 3;
/// Bytes of an image pixel for the page: red, green, blue, opaque alpha.
pub const RGBA: usize = 4;
/// Video pixels per latent pixel along each side: the decoder's three doublings.
pub const LATENT_FACTOR: usize = VAE_SHAPE.scale();
/// Video frames per latent frame after the first.
pub const FRAME_STEP: usize = VAE_SHAPE.frames(1);
/// The autoencoder's latents are normalized by these per channel in training; the compiler folds
/// `z · std + mean` into the decoder's `post_quant`.
pub const LATENT_MEAN: [f32; LATENT_CHANNELS] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENT_STD: [f32; LATENT_CHANNELS] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

/// The noise schedule the transformer was trained on: timesteps over flow-matching levels.
pub const TRAINING_TIMESTEPS: usize = 1000;
/// The frame rate the model was trained at.
pub const FPS: usize = 16;
pub const DEFAULT_SHIFT: f32 = 5.0;
pub const DEFAULT_GUIDANCE: f32 = 5.0;
/// Wan's `sample_neg_prompt`, the negative prompt when none is given.
pub const NEGATIVE: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，\
    整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，\
    画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，\
    三条腿，背景人很多，倒着走";

pub(crate) struct VideoModel {
    pub(crate) dit: Dit,
    pub(crate) decoder: Decoder,
}

impl VideoModel {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            dit: Dit::load(store, DIT, DIT_SHAPE)?,
            decoder: Decoder::load(store, VAE, VAE_SHAPE)?,
        })
    }
}
