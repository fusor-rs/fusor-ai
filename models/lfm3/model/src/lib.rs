//! LiquidAI's LFM3-1.7B, LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B on the CPU, for native hosts
//! and WebAssembly, built on fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! The model replies to a conversation: each turn's prompt is the conversation as its chat
//! template writes it (`prompt`); LFM3 (`blocks::interleaved`: gated short convolutions between
//! sliding-window and full gated attention) reads it, and the reply is drawn token by token from
//! the head over the vocabulary as the model card samples it (`sampling`). LFM3-VL-2B and
//! LFM3-Omni-2B read a turn's images too: each cut into patches (`media`), through the vision tower
//! (`blocks::vit::interleaved`) and projector into the prompt's image tokens. LFM3-Audio-2B and
//! LFM3-Omni-2B hear a turn's clips (`audio::Hearing`) and speak (`audio::Voice`): a reply is text,
//! speech, or both in turns (`output`). `pipeline` lists the stages in order; `session` runs them.

pub mod audio;
pub mod media;
pub mod model;
mod output;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
pub mod session;

pub use ::media::image::{Pixels, Size};
pub use blocks::vit::Grid;
pub use blocks::vit::preprocessing::Patches;
pub use bundle::Error;
