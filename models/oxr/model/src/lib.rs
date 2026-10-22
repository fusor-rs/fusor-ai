//! SoMark's OXR-1.0 on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels,
//! thread team, blocks and bundle format.
//!
//! The model reads a document image under a task's instruction (a whole page, text, a table, a
//! formula, code, a chemical structure): the image, resized to whole 32-pixel blocks, is cut into
//! 16 × 16 patches; a SigLIP2 NaViT encoder (`blocks::vit`) reads all of them at once with 2-D
//! rotary positions; each 2 × 2 block of encoded patches becomes a distribution over a visual
//! vocabulary of 65 532 words, and its token is the mixture of their embeddings; and Qwen3 0.6B
//! (`blocks::transformer`) reads the prompt with the image's tokens in it and writes the answer
//! greedily. `pipeline` lists the stages in order; `session` runs them.

mod image;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;
mod vision;

pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::{Layout, Patches};
pub use bundle::Error;
pub use image::{MAX_PIXELS, MIN_PIXELS, patches};
pub use media::image::{Pixels, Size};
