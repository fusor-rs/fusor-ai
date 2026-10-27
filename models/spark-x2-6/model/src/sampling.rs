//! Drawing a reply's next token (`blocks::lm::sampling`), the model card's settings the defaults
//! (`PRESET`).

pub use blocks::lm::sampling::{Constraint, Sampler, Sampling};

use crate::model::VOCABULARY;

/// The model card's `generation_config.json`: a temperature of 1, `top_p` 0.95, every token kept
/// (`top_k` −1) and no penalty.
pub const PRESET: Sampling = Sampling {
    temperature: 1.0,
    top_p: 0.95,
    top_k: VOCABULARY,
    presence_penalty: 0.0,
};
