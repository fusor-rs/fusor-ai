//! How P-JiT 21K makes an image, stage by stage.
//!
//! A request's prompts run `Prompt` and `Text` once (the text engine); `Dit` and `Sampler` then
//! repeat once per step (the image engine); the last step's image is the result: pixel space
//! needs no decoder.

/// One stage of generation, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The caption and the negative prompt to Qwen3.5-4B's tokens, chat template applied.
    Prompt,
    /// The tokens through Qwen3.5-4B's first 28 layers: the hidden states after layers 7, 15 and
    /// 27, rows of 7680.
    Text,
    /// The image through the DiT at a timestep under one branch's text: the clean image it
    /// predicts.
    Dit,
    /// Both branches' velocities through adaptive projected guidance and an Euler step: the next
    /// image.
    Sampler,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[Stage::Prompt, Stage::Text, Stage::Dit, Stage::Sampler];

    fn name(self) -> &'static str {
        match self {
            Stage::Prompt => "prompt",
            Stage::Text => "text",
            Stage::Dit => "dit",
            Stage::Sampler => "sampler",
        }
    }
}
