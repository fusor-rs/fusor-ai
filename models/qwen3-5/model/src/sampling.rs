//! Drawing a reply's next token (`blocks::lm::sampling`), the model card's settings the defaults
//! (`preset`).

pub use blocks::lm::sampling::{Constraint, Sampler, Sampling};

use crate::session::Message;

/// The model card's `top_k`, in every mode.
const PRESET_TOP_K: usize = 20;

/// What a conversation holds: text only, or images or videos too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Content {
    Text,
    Vision,
}

/// The model card's settings for `message`, in a conversation of `content`.
pub fn preset(message: &Message, content: Content) -> Sampling {
    let (temperature, top_p, presence_penalty) = match (message.thinking, content) {
        (false, Content::Text) => (1.0, 1.0, 2.0),
        (false, Content::Vision) => (0.7, 0.8, 1.5),
        (true, Content::Text) => (1.0, 0.95, 1.5),
        (true, Content::Vision) => (0.6, 0.95, 0.0),
    };
    Sampling {
        temperature,
        top_p,
        top_k: PRESET_TOP_K,
        presence_penalty,
    }
}
