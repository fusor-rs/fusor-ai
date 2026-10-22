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

pub use bundle::Error;
pub use decoder::Step;
pub use kernels::wav;
pub use model::SAMPLE_RATE;
