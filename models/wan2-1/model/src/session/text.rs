//! The text engine: a prompt's tokens through umT5-XXL.

use blocks::t5;
use bundle::{Load, Store, Tokenizer};
use team::Team;

use crate::Error;
use crate::model::{TEXT, TEXT_SHAPE, TEXT_TOKENS};
use crate::prompt::tokenize;

pub struct TextEngine {
    encoder: t5::Encoder,
    tokenizer: Tokenizer,
    team: Team,
    buffers: t5::Buffers,
}

impl bundle::Engine for TextEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let encoder = t5::Encoder::load(store, TEXT, TEXT_SHAPE)?;
        let tokenizer = Tokenizer::take(store)?;
        let team = Team::global();
        let buffers = TEXT_SHAPE.buffers(TEXT_TOKENS, team.threads());
        Ok(Self {
            encoder,
            tokenizer,
            team,
            buffers,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {}
}

impl TextEngine {
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        tokenize(&self.tokenizer, text)
    }

    /// umT5-XXL's last hidden states of `tokens` (1 to `TEXT_TOKENS`, each below the vocabulary),
    /// rows of `TEXT_SHAPE.hidden`.
    pub fn encode_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let vocabulary = TEXT_SHAPE.vocabulary;
        if !(1..=TEXT_TOKENS).contains(&tokens.len())
            || tokens.iter().any(|&token| token as usize >= vocabulary)
        {
            return Err(Error::Request(format!(
                "{} tokens: 1 to {TEXT_TOKENS}, each below {vocabulary}",
                tokens.len()
            )));
        }
        let Self {
            encoder,
            team,
            buffers,
            ..
        } = self;
        team.run(|ctx| encoder.forward(ctx, buffers, tokens));
        Ok(buffers.states.read()[..tokens.len() * TEXT_SHAPE.hidden].to_vec())
    }

    /// The states of `text`'s tokens.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        let tokens = self.tokens(text)?;
        self.encode_tokens(&tokens)
    }
}
