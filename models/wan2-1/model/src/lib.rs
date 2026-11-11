//! Wan2.1-T2V-1.3B (Wan-AI) text to video on the CPU, for native hosts and WebAssembly; built on
//! fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! Two engines, one bundle each. The text engine (`session::TextEngine`) turns a prompt into
//! umT5-XXL's hidden states, one row per token (`prompt`). The video engine
//! (`session::VideoEngine`) takes the states of the prompt and of the negative prompt: from
//! seeded Gaussian latents, DPM-Solver++(2M) at shifted flow-matching levels asks the diffusion
//! transformer for the velocity under each prompt at every step and guides between them; the
//! autoencoder's decoder then turns the last latents into frames, one latent frame at a time.
//! `pipeline` lists the stages in order; `session` runs them a step at a time.

pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;

pub use blocks::video::Video;
pub use bundle::Error;
