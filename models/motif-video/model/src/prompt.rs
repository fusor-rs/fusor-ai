//! A prompt as T5Gemma2's tokens: `<bos>`, then the first `TEXT_TOKENS − 1` of the text's ids
//! (`tokenizer.json` without its padding and truncation, which the compiler drops), as the
//! pipeline's `tokenizer(…, max_length=512, truncation=True)` gives them before its padding.

use bundle::Tokenizer;

use crate::Error;
use crate::model::{BEGIN, TEXT_TOKENS};

/// The ids of `text`'s tokens, `BEGIN` first, at most `TEXT_TOKENS` in all.
pub fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    let encoded = tokenizer.encode(text)?;
    let mut ids = vec![BEGIN];
    ids.extend(encoded.into_iter().take(TEXT_TOKENS - 1));
    Ok(ids)
}
