//! XHToken's Spark-X2.6-1.7B on the CPU, for native hosts and WebAssembly, built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! The model replies to a conversation: each turn's prompt is the conversation as its chat
//! template writes it (`prompt`); Spark (`blocks::interleaved`: sliding-window and full gated
//! attention) reads it, and the reply is drawn token by token from the head over the vocabulary
//! as the model card samples it (`sampling`). `pipeline` lists the stages in order; `session`
//! runs them.

pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
pub mod session;

pub use bundle::Error;
