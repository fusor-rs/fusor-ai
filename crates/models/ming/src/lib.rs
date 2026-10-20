//! Ming-3.0-tts-0.5B on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels,
//! thread team, blocks and bundle format. "Plan §" references are to the design in ming-wasm's
//! `docs/plan.md`.
//!
//! The model: a Qwen2-0.5B LLM whose last hidden state conditions a DiT flow-matching head that
//! samples one 64-d latent per 80 ms frame; an aggregator turns each latent into the LLM's next
//! input; a Qwen2-backbone VAE decoder with an iSTFT head turns latents into 44.1 kHz audio.
//! `pipeline` lists the stages in order.

pub mod flow_head;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;
mod text;
pub mod vocoder;
pub mod wav;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("format: {0}")]
    Format(String),
    #[error("integrity: {0}")]
    Integrity(String),
    #[error("request: {0}")]
    Request(String),
}

impl Error {
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Format(_) => "format",
            Error::Integrity(_) => "integrity",
            Error::Request(_) => "request",
        }
    }
}

impl From<kernels::Error> for Error {
    fn from(error: kernels::Error) -> Self {
        Error::Format(error.to_string())
    }
}

impl From<bundle::Error> for Error {
    fn from(error: bundle::Error) -> Self {
        match error {
            bundle::Error::Integrity(message) => Error::Integrity(message),
            bundle::Error::Format(message) => Error::Format(message),
        }
    }
}
