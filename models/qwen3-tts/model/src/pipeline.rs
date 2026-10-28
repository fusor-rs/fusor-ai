//! How Qwen3-TTS speaks, stage by stage.
//!
//! A request's voice clip goes through `Embed`, and, when the clip's transcript is given,
//! `Encode`; the prompt through `Prefill`; then each 80 ms frame is one `Speak`; `Match` ends the
//! request by comparing the speech with the voice.

/// One stage of a request, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The voice clip through the speaker encoder: its embedding (once per clip).
    Embed,
    /// The voice clip through the speech tokenizer's encoder: its frames (once per clip, with a
    /// transcript).
    Encode,
    /// The prompt through the talker (a transcript's frames through the decoder, too): the
    /// first frame's state and logits.
    Prefill,
    /// A frame: the talker's first code, the code predictor's other 15, the decoder's 1,920
    /// samples, and the talker's next row.
    Speak,
    /// The speech through the speaker encoder: its likeness to the voice.
    Match,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Embed,
        Stage::Encode,
        Stage::Prefill,
        Stage::Speak,
        Stage::Match,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Embed => "embed",
            Stage::Encode => "encode",
            Stage::Prefill => "prefill",
            Stage::Speak => "speak",
            Stage::Match => "match",
        }
    }
}
