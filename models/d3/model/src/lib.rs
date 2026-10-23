//! vLLM Semantic Router's d3-nano and d3-lite (Decision 3.0) on the CPU, for native hosts and
//! WebAssembly, built on fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! d3 answers a request's questions about a state, with images and videos, writing nothing:
//! each question is its own prompt (`prompt`), its options listed under answer codes; each
//! image, and each two frames of a video, resized to whole 32-pixel blocks and cut into
//! two-frame patches (`media`), goes through a NaViT encoder and a merger into one token per
//! 2 × 2 patches (`blocks::vit`); Qwen3.5 (`blocks::hybrid`, its attention layers reading the
//! whole prompt both ways) reads the prompt with those tokens in it, and a readout of the last
//! row over the codes gives every option's probability (`answer`). The two sizes share all of
//! it, on Qwen3.5's 2B and 0.8B backbones (`model::Size`). `pipeline` lists the stages in
//! order; `session` runs them.

pub mod answer;
pub mod media;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod request;
pub mod session;

pub use ::media::image::{Pixels, Size};
pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::Patches;
pub use bundle::Error;
