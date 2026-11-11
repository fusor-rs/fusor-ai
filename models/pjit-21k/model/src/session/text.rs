//! The text engine: a caption's tokens through Qwen3.5-4B's first layers, keeping the residual
//! stream after each of `TEXT_LAYERS`.

use blocks::hybrid::{self, Rows};
use blocks::rotary::Position;
use blocks::transformer::Mask;
use bundle::{Store, Tokenizer};
use kernels::Table;
use team::{Ctx, SharedBuf, Team};

use crate::Error;
use crate::model::{HIDDEN, LANGUAGE, Language, TEXT_LAYERS, TEXT_TOKENS, TEXT_WIDTH, VOCABULARY};
use crate::prompt::{Tokens, tokenize};

pub struct TextEngine {
    language: Language,
    tokenizer: Tokenizer,
    team: Team,
    state: hybrid::State,
    buffers: hybrid::Buffers,
    /// The states of the last call: `TEXT_TOKENS` rows of `TEXT_WIDTH`.
    states: SharedBuf,
}

impl bundle::Engine for TextEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let language = Language::load(store)?;
        let tokenizer = Tokenizer::take(store)?;
        let team = Team::global();
        Ok(Self {
            language,
            tokenizer,
            state: hybrid::State::new(&LANGUAGE, TEXT_TOKENS),
            buffers: LANGUAGE.buffers(TEXT_TOKENS, team.threads()),
            states: SharedBuf::new(TEXT_TOKENS * TEXT_WIDTH),
            team,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {}
}

impl TextEngine {
    pub fn tokens(&self, caption: &str) -> Result<Tokens, Error> {
        tokenize(&self.tokenizer, caption)
    }

    /// The states P-JiT reads of `ids` (1 to `TEXT_TOKENS`, each below the vocabulary): per row,
    /// the residual stream after each of `TEXT_LAYERS`, `[rows, 3, HIDDEN]`.
    pub fn encode_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, Error> {
        let rows = ids.len();
        if !(1..=TEXT_TOKENS).contains(&rows) || ids.iter().any(|&id| id as usize >= VOCABULARY) {
            return Err(Error::Request(format!(
                "{rows} tokens: 1 to {TEXT_TOKENS}, each below {VOCABULARY}"
            )));
        }
        let Self {
            language,
            team,
            state,
            buffers,
            states,
            ..
        } = self;
        state.clear();
        let positions: Vec<Position> = (0..rows).map(Position::uniform).collect();
        let span = Rows {
            first: 0,
            positions: &positions,
            mask: Mask::Causal,
        };
        team.run(|ctx| {
            embed(ctx, &language.embedding, buffers, ids);
            let mut previous = 0;
            for (slot, &layer) in TEXT_LAYERS.iter().enumerate() {
                let layers = previous..layer;
                language
                    .stack
                    .forward_layers(ctx, state, buffers, (&span, layers));
                keep(ctx, buffers, rows, (slot, states));
                previous = layer;
            }
        });
        Ok(self.states.read()[..rows * TEXT_WIDTH].to_vec())
    }

    /// The states of `caption`'s tokens.
    pub fn encode(&mut self, caption: &str) -> Result<Vec<f32>, Error> {
        let tokens = self.tokens(caption)?;
        self.encode_tokens(&tokens.ids)
    }
}

/// `buffers.stack.hidden` = the embeddings of `ids`; then a barrier.
fn embed(ctx: &Ctx, embedding: &Table, buffers: &hybrid::Buffers, ids: &[u32]) {
    let matrix = embedding.matrix();
    for row in ctx.split(ids.len(), 1) {
        let target = buffers.stack.hidden.part(row * HIDDEN..(row + 1) * HIDDEN);
        matrix.dequantize_row(ids[row] as usize, target);
    }
    ctx.barrier();
}

/// The residual stream of `rows` rows as layer `slot`'s states; then a barrier.
fn keep(ctx: &Ctx, buffers: &hybrid::Buffers, rows: usize, (slot, states): (usize, &SharedBuf)) {
    let hidden = buffers.stack.hidden.read();
    for row in ctx.split(rows, 1) {
        let at = (row * TEXT_LAYERS.len() + slot) * HIDDEN;
        let source = &hidden[row * HIDDEN..(row + 1) * HIDDEN];
        states.part(at..at + HIDDEN).copy_from_slice(source);
    }
    ctx.barrier();
}
