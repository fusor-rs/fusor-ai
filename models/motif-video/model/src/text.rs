//! T5Gemma2's text encoder: token embeddings scaled by √hidden (the end-of-image token's its
//! own), the `interleaved` stack read whole under `Mask::Full`, and a final RMSNorm.
//!
//! Tensors under `TEXT`: `embedding`, `end_of_image`, `norm` and the stack's.

use blocks::hybrid::Rows;
use blocks::interleaved::{Interleaved, State};
use blocks::rotary::Position;
use blocks::transformer::{Buffers, Mask};
use bundle::{Load, Scope, Store};
use kernels::ops::rms_norm;
use kernels::{Table, Vector};
use team::{Ctx, SharedBuf};

use crate::Error;
use crate::model::{
    END_OF_IMAGE, NORM_EPS, TEXT, TEXT_HIDDEN, TEXT_SHAPE, TEXT_TOKENS, VOCABULARY,
};

pub(crate) struct TextEncoder {
    embedding: Table,
    end_of_image: Vector,
    stack: Interleaved,
    norm: Vector,
    positions: Vec<Position>,
}

impl TextEncoder {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let stack = Interleaved::load(store, TEXT, TEXT_SHAPE)?;
        let mut scope = Scope::new(store, TEXT);
        Ok(Self {
            embedding: scope.load("embedding", [VOCABULARY, TEXT_HIDDEN])?,
            end_of_image: scope.load("end_of_image", TEXT_HIDDEN)?,
            stack,
            norm: scope.load("norm", TEXT_HIDDEN)?,
            positions: (0..TEXT_TOKENS).map(Position::uniform).collect(),
        })
    }

    /// The last hidden states of `tokens` (at most `TEXT_TOKENS`, each below the vocabulary)
    /// into `states` (rows of `TEXT_HIDDEN`).
    pub(crate) fn encode(
        &self,
        ctx: &Ctx,
        (state, buffers): (&State, &Buffers),
        tokens: &[u32],
        states: &SharedBuf,
    ) {
        let embedding = self.embedding.matrix();
        let scale = (TEXT_HIDDEN as f32).sqrt();
        for row in ctx.split(tokens.len(), 1) {
            let target = buffers
                .hidden
                .part(row * TEXT_HIDDEN..(row + 1) * TEXT_HIDDEN);
            if tokens[row] == END_OF_IMAGE {
                target.copy_from_slice(self.end_of_image.values());
                continue;
            }
            embedding.dequantize_row(tokens[row] as usize, target);
            for value in target {
                *value *= scale;
            }
        }
        ctx.barrier();
        let rows = Rows {
            first: 0,
            positions: &self.positions[..tokens.len()],
            mask: Mask::Full,
        };
        self.stack.forward(ctx, state, buffers, &rows);
        for row in ctx.split(tokens.len(), 1) {
            let at = row * TEXT_HIDDEN..(row + 1) * TEXT_HIDDEN;
            let last = &buffers.hidden.read()[at.clone()];
            rms_norm(last, self.norm.values(), NORM_EPS, states.part(at));
        }
        ctx.barrier();
    }
}
