//! How Phonon makes speech, stage by stage.
//!
//! A request's text is normalized and cut into chunks of whole sentences. A chunk starts from the
//! voice's primed state, runs `Text`, then repeats `Flow` and `Decode` once per 80 ms frame until
//! the end-of-speech logit (plus a few tail frames) or the chunk's frame budget ends it.

/// One stage of synthesis, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The voice's 125 embedded latents through the transformer: its KV cache prefix.
    Voice,
    /// The chunk's text token embeddings through the transformer, after the voice prefix.
    Text,
    /// The previous latent through the transformer at the next position, then the end-of-speech
    /// logit and the one-step flow head: this frame's 32-d latent.
    Flow,
    /// The latent through the streaming Mimi decoder: 1920 samples at 24 kHz.
    Decode,
}

impl Stage {
    pub const ALL: [Stage; 4] = [Stage::Voice, Stage::Text, Stage::Flow, Stage::Decode];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Voice => "voice",
            Stage::Text => "text",
            Stage::Flow => "flow",
            Stage::Decode => "decode",
        }
    }
}
