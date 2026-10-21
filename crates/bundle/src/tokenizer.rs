use crate::{Error, Store};

/// The bundle part holding a model's Hugging Face `tokenizer.json`.
pub const TOKENIZER_PART: &str = "tokenizer.json";

/// The tokenizer a bundle ships as its `TOKENIZER_PART` blob.
pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// Takes the tokenizer out of a complete store.
    pub fn take(store: &mut Store) -> Result<Self, Error> {
        Self::parse(&store.take_blob(TOKENIZER_PART)?)
    }

    pub fn parse(json: &[u8]) -> Result<Self, Error> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(json)
            .map_err(|error| Error::Format(format!("tokenizer: {error}")))?;
        Ok(Self(tokenizer))
    }

    /// The ids of `text`, without special tokens added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .0
            .encode(text, false)
            .map_err(|error| Error::Format(format!("tokenize: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}
