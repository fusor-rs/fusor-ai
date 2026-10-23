//! Indra OCR on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels, thread
//! team, blocks and bundle format: the reference's networks (line detection, page orientation,
//! script and math classification, CTC recognition with language models, the math reader, the
//! reading-order network) and the pipeline around them. `pipeline` lists the stages in order;
//! `session` runs them.

pub mod assemble;
pub mod classifier;
pub mod crops;
pub mod detector;
pub mod layout;
pub mod lines;
pub mod lm;
pub mod math;
pub mod model;
pub mod order;
pub mod orient;
pub mod page;
pub mod pipeline;
pub mod reading;
pub mod recognizer;
pub mod session;
pub mod vocab;

mod mobile;
mod plane;
mod rtl;
mod sequence;
mod text;

pub use bundle::Error;
pub use media::image::{Pixels, Size};
pub use plane::Scratch;
