//! Qwen3-TTS-12Hz-0.6B-Base on the CPU, for native hosts and WebAssembly, built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! The model speaks a text in the voice of a short clip. The clip goes through the speaker
//! encoder (`speaker`: log-mel features into an ECAPA-TDNN) and, when the clip's transcript is
//! given, the speech tokenizer's encoder (`codec::encoder`: Mimi) into codes. The talker
//! (`blocks::lm`, Qwen3) reads the prompt (`prompt`) and draws a frame's first code
//! (`sampling`); the code predictor (`predictor`) draws its other 15; the speech tokenizer's
//! decoder (`codec::decoder`) turns frames into 24 kHz samples. `pipeline` lists the stages in
//! order; `session` runs them.

pub mod codec;
pub mod model;
pub mod pipeline;
mod predictor;
pub mod prompt;
pub mod sampling;
pub mod session;
pub mod speaker;

pub use bundle::Error;
pub use codec::encoder::Encoded;
pub use media::wav;
pub use model::{FRAME_SAMPLES, SAMPLE_RATE};
