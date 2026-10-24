//! How Qwen3.5 replies to a turn, stage by stage.

/// One stage of a reply, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// An image or two video frames to normalized two-frame 16 × 16 patches (`media`).
    Preprocess,
    /// The patches through the patch embedding, plus the position grid resampled to them.
    Embed,
    /// The embedded patches through the encoder's layers.
    Encode,
    /// Every 2 × 2 encoded patches into one language-model token.
    Merge,
    /// The conversation's prompt, from the last reply's opener (or its first row), through the
    /// language model: the last row's final-normed state, its logits and the reply's first
    /// token.
    Prefill,
    /// The reply's last token through the language model: the next one.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Preprocess,
        Stage::Embed,
        Stage::Encode,
        Stage::Merge,
        Stage::Prefill,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Preprocess => "preprocess",
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Merge => "merge",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        }
    }
}
