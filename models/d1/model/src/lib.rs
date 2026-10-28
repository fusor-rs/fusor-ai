//! LiquidAI's d1-2B and d1-Omni-2B on the CPU, for native hosts and WebAssembly, built on
//! fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! d1 answers a request's named questions about a state (text or JSON) and its media in one
//! pass, generating nothing: each question is a suffix after the state (`prompt`), up to the
//! checkpoint's set size of them packed after one state; each image is cut into patches
//! (`media`) and through LFM3-VL's vision tower and projector (`blocks::vit::interleaved`), each
//! d1-Omni-2B clip into log-mel features and through its audio encoder (`blocks::audio`); LFM3
//! (`blocks::interleaved`) reads the pack, its full attention layers reading the state with
//! every question both ways and each question with the state and itself (`Mask::Groups`), and
//! each answer is read at its slots (`answer`): the head's logits over its options' tokens, or
//! for a read the state's positions (`pointer`), calibrated (`calibration`). `pipeline` lists
//! the stages in order; `session` runs them.

pub mod answer;
pub mod calibration;
pub mod grid;
pub mod media;
pub mod model;
mod parsing;
pub mod pipeline;
pub mod pointer;
pub mod prompt;
pub mod request;
pub mod session;

pub use ::media::image::{Pixels, Size};
pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::Patches;
pub use bundle::Error;
