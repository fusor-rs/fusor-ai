//! How OXR reads an image, stage by stage.

/// One stage of reading, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The image to normalized 16 × 16 patches (`image::Patches`).
    Preprocess,
    /// The patches through the patch embedding, plus the position grid resampled to the image.
    Embed,
    /// The embedded patches through the 27 encoder layers.
    Encode,
    /// Every 2 × 2 encoded patches into one language-model token: their distribution over the
    /// visual words, then the mixture of the words' embeddings.
    Tokenize,
    /// The prompt, the image's tokens in it, through the 28 language-model layers: the first
    /// token of the answer.
    Prefill,
    /// One more token of the greedy answer.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Preprocess,
        Stage::Embed,
        Stage::Encode,
        Stage::Tokenize,
        Stage::Prefill,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Preprocess => "preprocess",
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Tokenize => "tokenize",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        }
    }
}
