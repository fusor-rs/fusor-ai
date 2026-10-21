//! Phonon-31m-en on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels,
//! thread team, blocks and bundle format.
//!
//! The model: a 6-layer causal transformer reads a voice prefix and the text, then emits one
//! 32-d latent per 80 ms frame through a one-step flow head, plus an end-of-speech logit; the
//! Mimi decoder turns latents into 24 kHz audio. `pipeline` lists the stages in order.

mod lm;
mod mimi;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;
mod text;

pub use bundle::Error;
pub use kernels::wav;
pub use model::{FRAME_SAMPLES, SAMPLE_RATE};
