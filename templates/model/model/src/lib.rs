//! __name__ on the CPU, for native hosts and WebAssembly, built on fusor-ai's kernels, thread
//! team, blocks and bundle format. `pipeline` lists the stages in order; `session` runs them.

pub mod pipeline;
pub mod session;

pub use bundle::Error;
