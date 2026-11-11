//! The model's dimensions and tensor names. The text bundle holds T5Gemma2's text encoder
//! (`TEXT`), SigLIP (`VISION`) and the tokenizer, the video bundle the transformer (`DIT`) and
//! the autoencoder's encoder and decoder (`VAE`); the compiler (`motif-video-compile`) writes
//! exactly these names.

use std::ops::RangeInclusive;

use blocks::apg;
use blocks::hybrid::{AttentionShape, Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Mixer, Norms, Window};
use blocks::mmdit;
use blocks::rotary::Rotation;
use blocks::vit;
use blocks::vit::tower::{self, Ending, Resample};
use blocks::wan_vae::{self, Decoder, Encoder};
use bundle::{Load, Store};
use kernels::ops::Exponentials;

use crate::Error;
use crate::transformer::Transformer;

/// The directories of the two bundles under a bundle root.
pub const TEXT_BUNDLE: &str = "text";
pub const VIDEO_BUNDLE: &str = "video";
pub const TEXT: &str = "text";
pub const VISION: &str = "vision";
pub const DIT: &str = "video.dit";
pub const VAE: &str = "video.vae";

pub const TEXT_HIDDEN: usize = 2560;
pub const VOCABULARY: usize = 262_144;
/// Tokens T5Gemma2 reads: `<bos>`, then the prompt's.
pub const TEXT_TOKENS: usize = 512;
pub const BEGIN: u32 = 2;
/// `<end_of_image>`, whose embedding is its own, unscaled.
pub const END_OF_IMAGE: u32 = 256_000;
pub(crate) const NORM_EPS: f32 = 1e-6;

const TEXT_LAYERS: usize = 34;
/// The last of every this many layers attends to every row, the others over a window.
const FULL_ATTENTION_INTERVAL: usize = 6;

const TEXT_MIXERS: [Mixer; TEXT_LAYERS] = {
    let mut mixers = [Mixer::Window; TEXT_LAYERS];
    let mut layer = FULL_ATTENTION_INTERVAL - 1;
    while layer < TEXT_LAYERS {
        mixers[layer] = Mixer::Full;
        layer += FULL_ATTENTION_INTERVAL;
    }
    mixers
};

/// T5Gemma2's text encoder: every sixth layer full attention (positions interpolated by 8), the
/// others a bidirectional window of 1024 rows; 8 query heads over 4 key and value heads of 256.
pub const TEXT_SHAPE: interleaved::Shape = interleaved::Shape {
    hidden: TEXT_HIDDEN,
    mixers: &TEXT_MIXERS,
    attention: AttentionShape {
        heads: 8,
        kv_heads: 4,
        head_dim: 256,
        rotary_dim: 256,
        rope_base: 1_000_000.0,
        rotation: Rotation::Linear(8),
        head_norm: HeadNorm::Rms,
        gate: Gate::None,
    },
    window: Window {
        rows: 1024,
        rotary_dim: 256,
        rope_base: 10_000.0,
        rotation: Rotation::Sequence,
    },
    conv_kernel: 0,
    feed_forward: 10_240,
    activation: Activation::GeluTanh,
    norms: Norms::Sandwich,
    norm_eps: NORM_EPS,
};

/// SigLIP So400m/14 at 896: 4096 patches of 14 × 14.
pub const VISION_SIDE: usize = 896;
pub const VISION_PATCH: usize = 14;
pub const IMAGE_TOKENS: usize = (VISION_SIDE / VISION_PATCH) * (VISION_SIDE / VISION_PATCH);
pub const IMAGE_HIDDEN: usize = 1152;
/// SigLIP's 4304 MLP units, padded with zero units to whole quantization blocks (`gelu(0) = 0`,
/// so they add nothing).
pub const VISION_FEED_FORWARD: usize = 4320;
/// Its encoder has no rotary embeddings: the tower's learned positions are all it has.
pub const VISION_SHAPE: vit::Shape = vit::Shape {
    hidden: IMAGE_HIDDEN,
    layers: 27,
    heads: 16,
    feed_forward: VISION_FEED_FORWARD,
    rope_base: None,
    norm_eps: NORM_EPS,
    exponentials: Exponentials::Exact,
};
pub(crate) const TOWER: tower::Shape = tower::Shape {
    encoder: VISION_SHAPE,
    patch_values: IMAGE_CHANNELS * VISION_PATCH * VISION_PATCH,
    position_grid: VISION_SIDE / VISION_PATCH,
    resample: Resample::Bicubic,
    ending: Ending::Norm,
};

/// Motif-Video's transformer: 12 dual-stream blocks, then 24 single-stream ones.
pub const DIT_SHAPE: mmdit::Shape = mmdit::Shape {
    hidden: 1536,
    heads: 12,
    feed_forward: 6144,
    rope_sections: [8, 28, 28],
    rope_base: 10_000.0,
    eps: NORM_EPS,
};
pub const DUAL_BLOCKS: usize = 12;
pub const SINGLE_BLOCKS: usize = 24;
/// The single-stream blocks after the first `ENCODER_SINGLE_BLOCKS` are the DDT decoder.
pub const DECODER_BLOCKS: usize = 8;
/// The single-stream blocks before the decoder's, which carry the cross-attention to the text.
pub const ENCODER_SINGLE_BLOCKS: usize = SINGLE_BLOCKS - DECODER_BLOCKS;
/// Latent pixels per token along each side.
pub const PATCH: usize = 2;
/// Channels of a latent pixel the transformer reads: its noisy latents, the condition's, the mask.
pub const INPUT_CHANNELS: usize = LATENT_CHANNELS + CONDITION_CHANNELS;
/// Channels of a latent pixel's conditioning: the first frame's latents, and the mask.
pub const CONDITION_CHANNELS: usize = LATENT_CHANNELS + 1;
pub const TIME_FREQUENCIES: usize = 256;
/// The image projection's LayerNorms (`nn.LayerNorm`'s default).
pub const IMAGE_NORM_EPS: f32 = 1e-5;

/// The causal video autoencoder: its first-frame encoder and its decoder.
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
/// Video pixels per latent pixel along each side: the decoder's three doublings.
pub const LATENT_FACTOR: usize = VAE_SHAPE.scale();
/// Video frames per latent frame after the first.
pub const FRAME_STEP: usize = VAE_SHAPE.frames(1);
/// `latents_mean` and `latents_std` of the checkpoint's `vae/config.json`: the compiler folds
/// `z · std + mean` into the decoder's `post_quant`; a first frame's latents are
/// `(z − mean) / std`.
pub const LATENT_MEAN: [f32; LATENT_CHANNELS] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENT_STD: [f32; LATENT_CHANNELS] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];
/// `DiagonalGaussianDistribution`'s clamp of a log-variance.
pub const LOG_VARIANCE: RangeInclusive<f32> = -30.0..=20.0;

/// The noise schedule the transformer was trained on: timesteps over flow-matching levels.
pub const TRAINING_TIMESTEPS: usize = 1000;
/// `scheduler_config.json`'s `flow_shift`.
pub const SHIFT: f64 = 15.0;
/// `guider_config.json`: adaptive projected guidance at scale 8 (the default the page offers).
pub const GUIDANCE: apg::Settings = apg::Settings {
    scale: 8.0,
    momentum: 0.1,
    rescale: 12.0,
    eta: 1.0,
};
/// The clips' frame rate (the model card exports at 24 fps).
pub const FPS: usize = 24;
/// The size the checkpoint generates at, the model card's and the pipeline's default: its
/// guidance bounds a latent frame's norm for this many pixels, and other sizes come out burned.
pub const WIDTH: usize = 1280;
pub const HEIGHT: usize = 736;
/// The model card's negative prompt, used when none is given.
pub const NEGATIVE: &str = "text overlay, graphic overlay, watermark, logo, subtitles, \
    timestamp, broadcast graphics, UI elements, random letters, frozen pose, rigid, static \
    expression, jerky motion, mechanical motion, discontinuous motion, flat framing, depthless, \
    dull lighting, monotone, crushed shadows, blown-out highlights, shifting background, fading \
    background, poor continuity, identity drift, deformation, flickering, ghosting, smearing, \
    duplication, mutated proportions, inconsistent clothing, flat colors, desaturated, tonally \
    compressed, poor background separation, exposure shift, uneven brightness, color balance shift";

pub(crate) struct VideoModel {
    pub(crate) transformer: Transformer,
    pub(crate) encoder: Encoder,
    pub(crate) decoder: Decoder,
}

impl VideoModel {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            transformer: Transformer::load(store)?,
            encoder: Encoder::load(store, VAE, VAE_SHAPE)?,
            decoder: Decoder::load(store, VAE, VAE_SHAPE)?,
        })
    }
}
