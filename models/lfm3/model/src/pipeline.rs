//! How LFM3 replies to a turn, stage by stage.

/// One stage of a reply, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// One of the turn's images through the vision tower and the projector: its tokens
    /// (LFM3-VL-2B, LFM3-Omni-2B).
    See,
    /// One of the turn's clips through its features, the audio encoder and the projector: its
    /// tokens (LFM3-Audio-2B, LFM3-Omni-2B).
    Hear,
    /// The conversation's prompt, its images' tokens in their rows, through the language model:
    /// the last row's final-normed state, its logits and the reply's first token.
    Prefill,
    /// The reply's last token through the language model: the next one.
    Decode,
    /// The reply's last row through the language model, then the depth decoder: the next frame
    /// of speech, and its samples through the codec.
    Speak,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::See,
        Stage::Hear,
        Stage::Prefill,
        Stage::Decode,
        Stage::Speak,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::See => "see",
            Stage::Hear => "hear",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
            Stage::Speak => "speak",
        }
    }
}
