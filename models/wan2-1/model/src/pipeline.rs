//! How Wan2.1 makes a video, stage by stage.
//!
//! A request's prompts run `Prompt` and `Text` once; `Dit` and `Sampler` then repeat once per
//! step, and `Decode` runs once per latent frame at the end (`session::VideoEngine::step`).

/// One stage of generation, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The prompt and the negative prompt to umT5 tokens.
    Prompt,
    /// The tokens through umT5-XXL: rows of 4096.
    Text,
    /// The latents through the DiT at a timestep under one branch's text: the velocity.
    Dit,
    /// The guided velocity through a DPM-Solver++(2M) step at shifted flow levels: the next
    /// latents.
    Sampler,
    /// The last latents through the autoencoder's decoder, a latent frame at a time: the frames.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Prompt,
        Stage::Text,
        Stage::Dit,
        Stage::Sampler,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prompt => "prompt",
            Stage::Text => "text",
            Stage::Dit => "dit",
            Stage::Sampler => "sampler",
            Stage::Decode => "decode",
        }
    }
}
