//! Qwen's Qwen3.5-2B on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels,
//! thread team, blocks and bundle format.
//!
//! The model replies to a conversation, with images and videos: each turn's prompt is the
//! conversation as its chat template writes it (`prompt`); each image, and each two frames of a
//! video, resized to whole 32-pixel blocks and cut into two-frame patches (`media`), goes
//! through a NaViT encoder and a merger into one token per 2 × 2 patches (`blocks::vit`); Qwen3.5
//! (`blocks::hybrid`: the gated delta rule and gated attention) reads the prompt with those
//! tokens in it, and the reply is drawn token by token from the head over the vocabulary as the
//! model card samples it (`sampling`). `pipeline` lists the stages in order; `session` runs them.

pub mod media;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
pub mod session;

pub use ::media::image::{Pixels, Size};
pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::{Patches, Video};
pub use bundle::Error;
