//! How Ming makes speech, stage by stage.
//!
//! A request's text is cut into fragments (`text`), each synthesized on its own. A fragment
//! starts with `Prefill`, then repeats `Flow`, `Vocode`, `Aggregate` and `Decode` once per 80 ms
//! frame until the stop head, which reads `z` before `Flow`, ends it (`session::Engine::step`).

/// One stage of synthesis, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The prompt (system and user turns, the voice's caption, the fragment) through the Qwen2
    /// LLM: the conditioning vector `z` of the first frame.
    Prefill,
    /// `z`, the previous latent and Gaussian noise through the flow head's guided Euler steps:
    /// this frame's 64-d latent (`flow_head::FlowHead`).
    Flow,
    /// The latent through the streaming VAE decoder: 3528 samples at 44.1 kHz (`vocoder`).
    Vocode,
    /// The latent through the aggregator: the LLM's next input embedding.
    Aggregate,
    /// That embedding through the LLM at the next position: the next frame's `z`.
    Decode,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Prefill,
        Stage::Flow,
        Stage::Vocode,
        Stage::Aggregate,
        Stage::Decode,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prefill => "prefill",
            Stage::Flow => "flow",
            Stage::Vocode => "vocode",
            Stage::Aggregate => "aggregate",
            Stage::Decode => "decode",
        }
    }
}
