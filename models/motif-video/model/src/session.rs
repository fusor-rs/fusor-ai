//! The engines of a session: the text engine encodes prompts and a first image, the video engine
//! generates from their states. Each is built from its own bundle, so a page runs them on
//! separate memories.

mod request;
mod text;
mod video;

pub(crate) use request::check_size;
pub use request::{MAX_FRAMES, Request};
pub use text::TextEngine;
pub use video::{
    Branch, FirstImage, Plan, Progress, VideoEngine, conditioning, first_latents, transformer_input,
};
