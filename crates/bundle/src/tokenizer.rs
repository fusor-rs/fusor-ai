use crate::{Error, Store};

/// The bundle part holding a model's Hugging Face `tokenizer.json`.
pub const TOKENIZER_PART: &str = "tokenizer.json";

/// The tokenizer a bundle ships as its `TOKENIZER_PART` blob.
pub struct Tokenizer(tokenizers::Tokenizer);

/// A text's ids, and the characters of the text each one covers (Python's string indices, as
/// `return_offsets_mapping` gives them).
pub struct Encoded {
    pub ids: Vec<u32>,
    pub spans: Vec<(usize, usize)>,
}

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

    /// The ids of `text`, as `encode`, with their spans of it.
    pub fn encode_spans(&self, text: &str) -> Result<Encoded, Error> {
        let encoding = self
            .0
            .encode_char_offsets(text, false)
            .map_err(|error| Error::Format(format!("tokenize: {error}")))?;
        Ok(Encoded {
            ids: encoding.get_ids().to_vec(),
            spans: encoding.get_offsets().to_vec(),
        })
    }

    /// The text of every special token.
    pub fn specials(&self) -> Vec<String> {
        let added = self.0.get_added_tokens_decoder().into_values();
        added
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect()
    }

    /// The text of `ids`, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.0
            .decode(ids, false)
            .map_err(|error| Error::Format(format!("detokenize: {error}")))
    }
}
