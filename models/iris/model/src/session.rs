//! The engines of a session: the text engine turns a prompt into Qwen3-VL's layer states, the
//! image engine holds one of the four transformers a bundle ships and makes an image, a depth
//! map or a restoration of a photo. Each is built from its own bundle, so a page runs them on
//! separate memories.

mod image;
mod jobs;
mod request;
mod text;

pub use image::{Branch, ImageEngine};
pub use request::{DepthRequest, Generation, Plan, Progress, Upscale, photo};
pub use text::TextEngine;
