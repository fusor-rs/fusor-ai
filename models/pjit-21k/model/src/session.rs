//! Generation in two engines, each its own runtime instance in the browser: the text engine
//! encodes captions; the image engine samples from their states.

mod image;
mod text;

pub use image::{Branch, ImageEngine, Plan, Progress, Request, knots, timestep};
pub use text::TextEngine;
