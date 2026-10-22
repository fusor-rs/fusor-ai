//! A prompt as CLIP's `TOKENS` tokens: the start token, the first `TOKENS − 2` tokens of its
//! text (CLIP's byte-level BPE, the bundle's `tokenizer.json`), the end token, then the end token
//! again to fill them, as the reference pipeline's `CLIPTokenizer(padding="max_length",
//! truncation=True)` gives them.

use bundle::Tokenizer;

use crate::Error;
use crate::model::{END, START, TOKENS};

/// A prompt's tokens, and how many of its text's tokens did not fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub ids: Vec<u32>,
    pub dropped: usize,
}

pub fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Tokens, Error> {
    let encoded = tokenizer.encode(text)?;
    let room = TOKENS - 2;
    let kept = encoded.len().min(room);
    let mut ids = Vec::with_capacity(TOKENS);
    ids.push(START);
    ids.extend_from_slice(&encoded[..kept]);
    ids.resize(TOKENS, END);
    Ok(Tokens {
        ids,
        dropped: encoded.len() - kept,
    })
}
