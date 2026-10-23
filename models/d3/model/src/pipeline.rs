//! How d3 answers a request, stage by stage.

/// One stage of a request, with what it consumes and produces.
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
    /// One question's prompt, the media's tokens in it, through the language model in one
    /// call: the last row's final-normed state.
    Read,
    /// The state read out over the question's answer codes: its options' probabilities and its
    /// answer.
    Answer,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Preprocess,
        Stage::Embed,
        Stage::Encode,
        Stage::Merge,
        Stage::Read,
        Stage::Answer,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Preprocess => "preprocess",
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Merge => "merge",
            Stage::Read => "read",
            Stage::Answer => "answer",
        }
    }
}
