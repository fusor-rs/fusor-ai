//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `text`, `unet`, `vae` — so this is the table of contents; the compiler
//! (`realistic-vision-compile`) writes exactly these names.

use blocks::transformer::{Architecture, Shape};
use blocks::unet::{self, Unet};
use blocks::vae::{self, Decoder};
use bundle::{Load, Store};

use crate::Error;
use crate::text::TextEncoder;

/// CLIP ViT-L/14's text encoder.
pub const TEXT_SHAPE: Shape = Shape {
    architecture: Architecture::Clip,
    hidden: 768,
    layers: 12,
    heads: 12,
    kv_heads: 12,
    head_dim: 64,
    feed_forward: 3072,
    rope_base: 0.0,
    norm_eps: 1e-5,
};
/// The rows of CLIP's token embeddings.
pub const VOCABULARY: usize = 49_408;
/// Tokens of an encoded prompt: the start, up to `TOKENS − 2` of the prompt's, the end, then the
/// end again to fill them.
pub const TOKENS: usize = 77;
pub const START: u32 = 49_406;
pub const END: u32 = 49_407;

/// Stable Diffusion 1.5's UNet.
pub const UNET_SHAPE: unet::Shape = unet::Shape {
    latent: LATENT_CHANNELS,
    channels: &[320, 640, 1280, 1280],
    depths: &[1, 1, 1, 0],
    middle_depth: 1,
    blocks: 2,
    kernel: 3,
    heads: 8,
    feed_forward: 4,
    context: TEXT_SHAPE.hidden,
    time: 4,
    groups: 32,
    norm_eps: 1e-5,
    spatial_eps: 1e-6,
};
/// The KL autoencoder's decoder (`kl-f8`).
pub const VAE_SHAPE: vae::Shape = vae::Shape {
    latent: LATENT_CHANNELS,
    channels: &[128, 256, 512, 512],
    blocks: 2,
    kernel: 3,
    groups: 32,
    eps: 1e-6,
    outputs: IMAGE_CHANNELS,
};
pub const LATENT_CHANNELS: usize = 4;
/// Red, green, blue.
pub const IMAGE_CHANNELS: usize = 3;
/// Image pixels per latent pixel along each side: the decoder's three doublings.
pub const LATENT_FACTOR: usize = 8;
/// The autoencoder's latents are scaled by this in training; the compiler folds its inverse into
/// the decoder's `post_quant`.
pub const LATENT_SCALING: f32 = 0.18215;

/// The noise schedule the UNet was trained on: `β` scaled-linear from `BETA_START` to
/// `BETA_END` over `TRAINING_TIMESTEPS`.
pub const BETA_START: f64 = 0.000_85;
pub const BETA_END: f64 = 0.012;
pub const TRAINING_TIMESTEPS: usize = 1000;

pub(crate) struct Model {
    pub(crate) text: TextEncoder,
    pub(crate) unet: Unet,
    pub(crate) decoder: Decoder,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            text: TextEncoder::load(store, "text", ())?,
            unet: Unet::load(store, "unet", UNET_SHAPE)?,
            decoder: Decoder::load(store, "vae", VAE_SHAPE)?,
        })
    }
}
