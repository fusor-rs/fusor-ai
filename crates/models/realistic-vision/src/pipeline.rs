//! How Realistic Vision makes an image, stage by stage.
//!
//! A request's prompts run `Prompt` and `Text` once; `Unet` and `Sampler` then repeat once per
//! step, and `Decode` runs once at the end (`session::Engine::step`).

/// One stage of generation, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The prompt and the negative prompt to 77 CLIP tokens each.
    Prompt,
    /// The tokens through CLIP's text encoder: the UNet's context, 77 rows of 768.
    Text,
    /// The latents (both branches of classifier-free guidance) through the UNet at a timestep:
    /// the noise it predicts in each.
    Unet,
    /// The guided noise through a DPM-Solver++(2M) step at Karras noise levels: the next latents.
    Sampler,
    /// The last latents through the autoencoder's decoder: the image.
    Decode,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Prompt,
        Stage::Text,
        Stage::Unet,
        Stage::Sampler,
        Stage::Decode,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Prompt => "prompt",
            Stage::Text => "text",
            Stage::Unet => "unet",
            Stage::Sampler => "sampler",
            Stage::Decode => "decode",
        }
    }
}
