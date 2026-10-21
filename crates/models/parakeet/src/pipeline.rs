//! How Parakeet Redux reads speech, stage by stage.
//!
//! A recording longer than a segment is first marked by `Speech` and cut at its pauses; each
//! segment then runs `Features`, `Subsampling`, `Encoder` and `Decoder` once, whole.

/// One stage of recognition, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// 16 kHz samples to normalized log-mel features, one frame per 10 ms.
    Features,
    /// The features through the convolutional subsampler: one frame per 80 ms.
    Subsampling,
    /// The subsampled frames through the speech head: a speech probability per frame.
    Speech,
    /// The features through the subsampler, the 24 conformer layers and the joint's encoder
    /// projection.
    Encoder,
    /// The encoded frames through the greedy token-and-duration transducer: tokens and their
    /// frames.
    Decoder,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Features,
        Stage::Subsampling,
        Stage::Speech,
        Stage::Encoder,
        Stage::Decoder,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Features => "features",
            Stage::Subsampling => "subsampling",
            Stage::Speech => "speech",
            Stage::Encoder => "encoder",
            Stage::Decoder => "decoder",
        }
    }
}
