//! How Iris makes an image, stage by stage.
//!
//! A prompt runs `Prompt`, `Text` and `Adapter` once; `Dit` and `Sampler` then repeat once per
//! step. Depth and the restorer have no prompt: `Depth` and `Restore` run once on a photo.

/// One stage of work, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The prompt into Qwen3-VL's template tokens.
    Prompt,
    /// The tokens through Qwen3-VL: 12 layers' states per row.
    Text,
    /// The states through the layer-wise blocks and the refiner: 300 rows of 2560.
    Adapter,
    /// An image at a timestep through the transformer: the velocity.
    Dit,
    /// The guided velocity through a step of DPM-Solver++(2M) or of Turbo's Euler: the next
    /// image.
    Sampler,
    /// A photo through the depth model and back to its size, coloured.
    Depth,
    /// A photo through the restorer's tiles, fused and colour-fixed.
    Restore,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Prompt,
        Stage::Text,
        Stage::Adapter,
        Stage::Dit,
        Stage::Sampler,
        Stage::Depth,
        Stage::Restore,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prompt => "prompt",
            Stage::Text => "text",
            Stage::Adapter => "adapter",
            Stage::Dit => "dit",
            Stage::Sampler => "sampler",
            Stage::Depth => "depth",
            Stage::Restore => "restore",
        }
    }
}
