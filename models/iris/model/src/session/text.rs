//! The text engine: a prompt's tokens through Qwen3-VL's language model, keeping the residual
//! stream after each of the layers Iris reads.

use blocks::transformer::{Buffers, KvCache, Mask, Span, Transformer};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::Table;
use team::{Ctx, SharedBuf, Team};

use crate::Error;
use crate::model::{HIDDEN, TEXT, TEXT_LAYERS, TEXT_ROWS, TEXT_SHAPE, VOCABULARY};
use crate::prompt::{PREFIX, tokenize};

pub struct TextEngine {
    embedding: Table,
    stack: Transformer,
    tokenizer: Tokenizer,
    prefix: usize,
    team: Team,
    cache: KvCache,
    buffers: Buffers,
    states: SharedBuf,
}

impl bundle::Engine for TextEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let embedding = Scope::new(store, TEXT).load("embed", [VOCABULARY, HIDDEN])?;
        let stack = Transformer::load(store, TEXT, TEXT_SHAPE)?;
        let tokenizer = Tokenizer::take(store)?;
        let prefix = tokenizer.encode(PREFIX)?.len();
        let team = Team::global();
        let rows = prefix + TEXT_ROWS;
        Ok(Self {
            embedding,
            stack,
            prefix,
            cache: KvCache::new(TEXT_SHAPE, rows),
            buffers: TEXT_SHAPE.buffers(rows, team.threads()),
            states: SharedBuf::new(TEXT_ROWS * TEXT_LAYERS.len() * HIDDEN),
            tokenizer,
            team,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {}
}

impl TextEngine {
    pub fn tokens(&self, caption: &str) -> Result<Vec<u32>, Error> {
        tokenize(&self.tokenizer, caption)
    }

    /// The states Iris reads of `ids` (`tokens`' ids): for each row from the caption on, the
    /// residual stream after each of `TEXT_LAYERS`, `[rows, 12, HIDDEN]`.
    pub fn encode_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, Error> {
        let rows = ids.len();
        if rows <= self.prefix
            || rows > self.prefix + TEXT_ROWS
            || ids.iter().any(|&id| id as usize >= VOCABULARY)
        {
            return Err(Error::Request(format!(
                "{rows} tokens: {} to {}, each below {VOCABULARY}",
                self.prefix + 1,
                self.prefix + TEXT_ROWS
            )));
        }
        let Self {
            embedding,
            stack,
            prefix,
            team,
            cache,
            buffers,
            states,
            ..
        } = self;
        team.run(|ctx| {
            embed(ctx, embedding, buffers, ids);
            let span = Span {
                rows,
                first: 0,
                mask: Mask::Causal,
            };
            let mut previous = 0;
            for (slot, &layer) in TEXT_LAYERS.iter().enumerate() {
                stack.forward_layers(ctx, cache, buffers, (span, previous..layer));
                keep(ctx, buffers, (*prefix, rows), (slot, states));
                previous = layer;
            }
        });
        let kept = (rows - self.prefix) * TEXT_LAYERS.len() * HIDDEN;
        Ok(self.states.read()[..kept].to_vec())
    }

    /// The states of `caption`'s prompt.
    pub fn encode(&mut self, caption: &str) -> Result<Vec<f32>, Error> {
        let ids = self.tokens(caption)?;
        self.encode_tokens(&ids)
    }
}

/// `buffers.hidden` = the embeddings of `ids`; then a barrier.
fn embed(ctx: &Ctx, embedding: &Table, buffers: &Buffers, ids: &[u32]) {
    let matrix = embedding.matrix();
    for row in ctx.split(ids.len(), 1) {
        let target = buffers.hidden.part(row * HIDDEN..(row + 1) * HIDDEN);
        matrix.dequantize_row(ids[row] as usize, target);
    }
    ctx.barrier();
}

/// The residual stream of rows `prefix..rows` as layer `slot`'s states; then a barrier.
fn keep(
    ctx: &Ctx,
    buffers: &Buffers,
    (prefix, rows): (usize, usize),
    (slot, states): (usize, &SharedBuf),
) {
    let hidden = buffers.hidden.read();
    for row in ctx.split(rows - prefix, 1) {
        let at = (row * TEXT_LAYERS.len() + slot) * HIDDEN;
        let source = &hidden[(prefix + row) * HIDDEN..(prefix + row + 1) * HIDDEN];
        states.part(at..at + HIDDEN).copy_from_slice(source);
    }
    ctx.barrier();
}
