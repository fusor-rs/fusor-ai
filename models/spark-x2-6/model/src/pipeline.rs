//! How Spark replies to a turn, stage by stage.

/// One stage of a reply, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The conversation's prompt through the language model: the last row's final-normed state,
    /// its logits and the reply's first token.
    Prefill,
    /// The reply's last token through the language model: the next one.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[Stage::Prefill, Stage::Decode];

    fn name(self) -> &'static str {
        match self {
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        }
    }
}
