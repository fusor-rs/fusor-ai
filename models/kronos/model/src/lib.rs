//! NeoQuasar's Kronos-base on the CPU, for native hosts and WebAssembly, built on fusor-ai's
//! kernels, thread team, blocks and bundle format.
//!
//! Kronos forecasts a series of candlesticks (open, high, low, close, volume and amount, one row
//! per interval) as a language model forecasts text: the context's columns are normalized
//! (`series`); the tokenizer turns each row into a token of two halves of 10 bits each
//! (`tokenizer`); the predictor reads the tokens with each row's time (`predictor`), a Qwen2
//! stack (`blocks::lm::Decoder`), and draws the next token's coarse half, then its fine half
//! given the coarse (`sampling`); the drawn tokens are decoded back to rows. `pipeline` lists
//! the stages in order; `session` runs them.

pub mod model;
pub mod pipeline;
pub mod predictor;
pub mod sampling;
pub mod series;
pub mod session;
pub mod tokenizer;

pub use bundle::Error;
