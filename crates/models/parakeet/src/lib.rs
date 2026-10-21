//! Moondream's Parakeet Redux on the CPU, for native hosts and WebAssembly, built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! The model is NVIDIA's parakeet-tdt-0.6b-v3 with every encoder projection ternary: 16 kHz
//! audio becomes normalized log-mel features, a FastConformer encoder (`blocks::conformer`)
//! turns them into one frame per 80 ms, and a token-and-duration transducer reads text off the
//! frames, each decision a token (or blank) and how many frames to skip. A small speech head on
//! the encoder's subsampler marks the pauses that long recordings are cut at. `pipeline` lists
//! the stages in order; `session` runs them on a recording, or live on audio as it arrives.

mod decoder;
mod features;
pub mod live;
pub mod model;
pub mod pipeline;
mod segment;
pub mod session;
pub mod speech;
pub mod vocabulary;
pub use kernels::wav;

pub use decoder::Step;
pub use model::SAMPLE_RATE;

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
