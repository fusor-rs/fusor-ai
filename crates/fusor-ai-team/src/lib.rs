//! An SPMD thread team for inference kernels.
//!
//! Model code runs as one function on every team thread: each thread takes its share of the
//! output with `Ctx::split`, and dependent phases are separated by `Ctx::barrier`. A component
//! call (a transformer forward, a sampler step loop) is one `Team::run`, so the cost of starting
//! threads is paid once per call rather than once per matrix multiply.
//!
//! Shared intermediate buffers are `SharedBuf`s under one discipline: between two barriers, each
//! element is either only read (by any threads) or written by exactly one thread. Without the
//! `threads` feature the team is the calling thread alone and barriers are no-ops, so the same
//! code is the single-threaded build.

mod shared;
mod team;

pub use shared::{PerThread, SharedBuf};
pub use team::{Ctx, Team};
