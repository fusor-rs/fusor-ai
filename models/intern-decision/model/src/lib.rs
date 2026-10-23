//! InternLM's Intern-Decision-2B on the CPU, for native hosts and WebAssembly, built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! The model answers a request's questions about a state, and images, in one pass, writing
//! nothing: the request is compiled into a chat whose answer is already written with a
//! `<decision>` marker per question (`prompt`); each image, resized to whole 32-pixel blocks and
//! cut into 16 × 16 patches, goes through a NaViT encoder (`blocks::vit`) and a merger into one
//! token per 2 × 2 patches; Qwen3.5 2B (`blocks::hybrid`: the gated delta rule and gated
//! attention) reads the prompt with the images' tokens in it; and the logits of the row before
//! each marker, over that question's options' symbols, are its calibrated answer (`decision`).
//! `pipeline` lists the stages in order; `session` runs them.

pub mod decision;
mod image;
mod json;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod request;
pub mod session;

pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::{Layout, Patches};
pub use bundle::Error;
pub use image::{MAX_PIXELS, MIN_PIXELS, patches};
pub use media::image::{Pixels, Size};
