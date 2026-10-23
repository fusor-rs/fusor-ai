//! PaddlePaddle's PaddleOCR-VL-1.6 on the CPU, for native hosts and WebAssembly, built on
//! fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! The model reads a document image under a task's instruction (text, a table, a formula, a
//! chart, a seal): the image, resized to whole 28-pixel blocks, is cut into 14 × 14 patches; a
//! NaViT vision encoder (`blocks::vit`) reads all of them at once with 2-D rotary positions; a
//! projector folds each 2 × 2 block into one token; and ERNIE 4.5 0.3B (`blocks::transformer`,
//! with multimodal RoPE) reads the prompt with the image's tokens in it and writes the answer
//! greedily. `pipeline` lists the stages in order; `session` runs them.

mod image;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;

pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::{Layout, Patches};
pub use bundle::Error;
pub use image::{MAX_PIXELS, MIN_PIXELS, patches};
pub use media::image::{Pixels, Size};
