//! Motif-Video-2B (Motif Technologies) text and image to video on the CPU, for native hosts and
//! WebAssembly; built on fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! Two engines, one bundle each. The text engine (`session::TextEngine`) turns a prompt into
//! T5Gemma2's hidden states, one row per token (`prompt`), and a first image into SigLIP's
//! (`image`). The video engine (`session::VideoEngine`) takes those states, encodes the first
//! frame's latents, and from seeded Gaussian latents asks the transformer for the velocity under
//! the prompt and the negative prompt at every step of a DPM-Solver++(2M) run, guiding between
//! them adaptively; the autoencoder's decoder then turns the last latents into frames, a strip of
//! rows at a time. Videos are 1280 × 736, the model card's size. `pipeline` lists the stages in
//! order; `session` runs them a step at a time.

pub mod image;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;
mod text;
mod transformer;

pub use blocks::video::Video;
pub use bundle::Error;
pub use media::image::{Pixels, Size};
