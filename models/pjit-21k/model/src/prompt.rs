//! A caption as Qwen3.5-4B's tokens, as P-JiT's text encoder writes it: the chat template's user
//! turn (the caption stripped of surrounding whitespace, the template's `trim`) and the reply's
//! opener with thinking off, `<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n
//! <think>\n\n</think>\n\n`, its first `TEXT_TOKENS` tokens (the reference truncates the end).

use bundle::Tokenizer;
use bundle::python::strip;

use crate::Error;
use crate::model::TEXT_TOKENS;

const USER: &str = "<|im_start|>user\n";
const OPENER: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub struct Tokens {
    pub ids: Vec<u32>,
    /// Tokens past `TEXT_TOKENS`, which the encoder does not read.
    pub dropped: usize,
}

pub fn tokenize(tokenizer: &Tokenizer, caption: &str) -> Result<Tokens, Error> {
    let templated = format!("{USER}{}{OPENER}", strip(caption));
    let mut ids = tokenizer.encode(&templated)?;
    let dropped = ids.len().saturating_sub(TEXT_TOKENS);
    ids.truncate(TEXT_TOKENS);
    Ok(Tokens { ids, dropped })
}
