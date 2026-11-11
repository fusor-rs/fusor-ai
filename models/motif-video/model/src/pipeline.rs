//! How Motif-Video makes a video, stage by stage.
//!
//! A request's prompts run `Prompt` and `Text` once, and a first image `Frame`, `Vision` and
//! `Encode`; `Dit`, `Guidance` and `Sampler` then repeat once per step, and `Decode` runs once
//! per strip of rows at the end (`session::VideoEngine::step`).

/// One stage of generation, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The prompt and the negative prompt to T5Gemma2's tokens.
    Prompt,
    /// A first image to the video's size and SigLIP's input.
    Frame,
    /// The tokens through T5Gemma2's encoder: rows of 2560.
    Text,
    /// The first frame through SigLIP: 4096 rows of 1152.
    Vision,
    /// The first frame through the autoencoder's encoder: its latents' moments.
    Encode,
    /// The latents through the transformer at a timestep under one branch: the velocity.
    Dit,
    /// Adaptive projected guidance of the two branches' velocities.
    Guidance,
    /// The guided velocity through a DPM-Solver++(2M) step at flow levels: the next latents.
    Sampler,
    /// The last latents through the autoencoder's decoder, a strip of rows at a time: the frames.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Prompt,
        Stage::Frame,
        Stage::Text,
        Stage::Vision,
        Stage::Encode,
        Stage::Dit,
        Stage::Guidance,
        Stage::Sampler,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prompt => "prompt",
            Stage::Frame => "frame",
            Stage::Text => "text",
            Stage::Vision => "vision",
            Stage::Encode => "encode",
            Stage::Dit => "dit",
            Stage::Guidance => "guidance",
            Stage::Sampler => "sampler",
            Stage::Decode => "decode",
        }
    }
}
