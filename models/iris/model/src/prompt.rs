//! A prompt as Qwen3-VL reads it for Iris: the template's system turn and user header, the
//! caption's tokens (at most `TEXT_ROWS` minus the suffix's, the rest dropped), the assistant
//! header; Iris keeps the states of the rows from the caption on (`Qwen3VLTextEncoder._run`).

use bundle::Tokenizer;

use crate::Error;
use crate::model::TEXT_ROWS;

pub(crate) const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, \
    shape, size, texture, quantity, text, spatial relationships of the objects and \
    background:<|im_end|>\n<|im_start|>user\n";
pub(crate) const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The template's tokens around the first tokens of `caption` that leave room for the suffix.
pub(crate) fn tokenize(tokenizer: &Tokenizer, caption: &str) -> Result<Vec<u32>, Error> {
    let mut ids = tokenizer.encode(PREFIX)?;
    let suffix = tokenizer.encode(SUFFIX)?;
    let caption = tokenizer.encode(caption)?;
    let budget = TEXT_ROWS - suffix.len();
    ids.extend(caption.iter().take(budget));
    ids.extend(&suffix);
    Ok(ids)
}
