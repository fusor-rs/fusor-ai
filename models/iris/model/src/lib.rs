//! Iris-3B (speridlabs) on the CPU, for native hosts and WebAssembly; built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! Two engines, one bundle each. The text engine (`session::TextEngine`) runs Qwen3-VL's
//! language model on a prompt and returns the 12 layers' states of each row. The image engine
//! (`session::ImageEngine`) holds one of the four transformers a bundle ships: from the prompt's
//! states (and a negative prompt's) it makes an image at the base model's DPM-Solver++(2M) run
//! with classifier-free guidance or at Turbo's four levels, or it reads a photo for a depth map
//! or a four times enlarged restoration. `pipeline` lists the stages in order; `session` runs
//! them a step at a time.

mod adapter;
pub mod depth;
mod dit;
mod inferno;
pub mod model;
pub mod picture;
pub mod pipeline;
mod prompt;
pub mod restore;
pub mod session;

pub use bundle::Error;
pub use media::image::{Pixels, Size};
