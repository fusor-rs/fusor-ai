//! The engines of a session: the text engine encodes prompts, the video engine generates from
//! their states. Each is built from its own bundle, so a page runs them on separate memories.

mod text;
mod video;

pub use text::TextEngine;
pub use video::{Branch, Plan, Progress, Request, VideoEngine};
