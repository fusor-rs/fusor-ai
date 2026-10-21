//! How PaddleOCR-VL reads an image, stage by stage.

/// One stage of reading, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The image to normalized 14 × 14 patches (`image::Patches`).
    Preprocess,
    /// The patches through the patch embedding, plus the position grid resampled to the image.
    Embed,
    /// The embedded patches through the 27 encoder layers and the final norm.
    Encode,
    /// Every 2 × 2 encoded patches through the projector: one language-model token.
    Project,
    /// The prompt, the image's tokens in it, through the 18 language-model layers: the first
    /// token of the answer.
    Prefill,
    /// One more token of the greedy answer.
    Decode,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::Preprocess,
        Stage::Embed,
        Stage::Encode,
        Stage::Project,
        Stage::Prefill,
        Stage::Decode,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Preprocess => "preprocess",
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Project => "project",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        }
    }
}
