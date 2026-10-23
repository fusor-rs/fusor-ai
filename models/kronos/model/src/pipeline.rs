//! How Kronos forecasts a series, stage by stage.

/// One stage of a forecast, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The context's rows to their columns' z-scores, clipped (`series::Normalized`).
    Normalize,
    /// The normalized rows through the tokenizer's encoder: each row's code and its token.
    Encode,
    /// The tokens and their stamps through the predictor's stack: the last row's `s1` logits.
    Read,
    /// A drawn `s1` through the dependency layer: the `s2` logits; both halves drawn, the
    /// token appended and read in turn.
    Sample,
    /// Each sample's last window of tokens through the tokenizer's decoder, averaged and
    /// brought back to the series' units.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Normalize,
        Stage::Encode,
        Stage::Read,
        Stage::Sample,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Normalize => "normalize",
            Stage::Encode => "encode",
            Stage::Read => "read",
            Stage::Sample => "sample",
            Stage::Decode => "decode",
        }
    }
}
