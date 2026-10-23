//! How Intern-Decision decides a request, stage by stage.

/// One stage of a decision, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// An image to normalized 16 × 16 patches (`image::patches`).
    Preprocess,
    /// The patches through the patch embedding, plus the position grid resampled to the image.
    Embed,
    /// The embedded patches through the 24 encoder layers.
    Encode,
    /// Every 2 × 2 encoded patches into one language-model token.
    Merge,
    /// The prompt, the images' tokens in it, through the 24 language-model layers, a chunk of
    /// rows at a time: the final-normed state of each question's deciding row.
    Read,
    /// Each question's options' logits to its calibrated answer.
    Decide,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Preprocess,
        Stage::Embed,
        Stage::Encode,
        Stage::Merge,
        Stage::Read,
        Stage::Decide,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Preprocess => "preprocess",
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Merge => "merge",
            Stage::Read => "read",
            Stage::Decide => "decide",
        }
    }
}
