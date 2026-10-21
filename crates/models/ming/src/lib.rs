//! Ming-3.0-tts-0.5B on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels,
//! thread team, blocks and bundle format.
//!
//! The model: a Qwen2-0.5B LLM whose last hidden state conditions a DiT flow-matching head that
//! samples one 64-d latent per 80 ms frame; an aggregator turns each latent into the LLM's next
//! input; a Qwen2-backbone VAE decoder with an iSTFT head turns latents into 44.1 kHz audio.
//! `pipeline` lists the stages in order.

pub mod flow_head;
pub mod model;
pub mod pipeline;
pub mod preset;
pub mod prompt;
pub mod session;
mod silence;
mod text;
pub mod vocoder;

pub use bundle::Error;
pub use kernels::wav;
pub use model::SAMPLE_RATE;
