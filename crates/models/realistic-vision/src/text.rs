//! CLIP's text encoder: token and learned position embeddings, the causal transformer
//! (`blocks::transformer`, the CLIP architecture) and a final LayerNorm. Its last hidden states,
//! one row per token, are the UNet's context.
//!
//! Tensors under a prefix `p`: `p.embedding` (`[VOCABULARY, hidden]`), `p.positions`
//! (`[TOKENS, hidden]`), the stack's `p.layers.{i}.…` and `p.norm`.

use blocks::norm::LayerNorm;
use blocks::transformer::{Buffers, KvCache, Mask, Span, Transformer};
use bundle::{Load, Scope, Store};
use kernels::Table;
use team::Team;

use crate::Error;
use crate::model::{TEXT_SHAPE, TOKENS, VOCABULARY};

const HIDDEN: usize = TEXT_SHAPE.hidden;

pub(crate) struct TextEncoder {
    embedding: Table,
    positions: Table,
    body: Transformer,
    norm: LayerNorm,
}

/// Activations of one `encode`.
pub(crate) struct Activations {
    stack: Buffers,
    cache: KvCache,
}

impl Activations {
    pub(crate) fn new(threads: usize) -> Self {
        Self {
            stack: TEXT_SHAPE.buffers(TOKENS, threads),
            cache: KvCache::new(TEXT_SHAPE, TOKENS),
        }
    }
}

impl TextEncoder {
    /// The last hidden states of `tokens` (`TOKENS` of them), `TOKENS × hidden` into `out`.
    pub(crate) fn encode(
        &self,
        team: &Team,
        activations: &mut Activations,
        tokens: &[u32],
        out: &mut [f32],
    ) {
        let hidden = activations.stack.hidden.get_mut();
        let mut position = [0.0; HIDDEN];
        let rows = hidden.as_chunks_mut::<HIDDEN>().0.iter_mut().zip(tokens);
        for (row, (embedded, &token)) in rows.enumerate() {
            self.embedding
                .matrix()
                .dequantize_row(token as usize, embedded);
            self.positions.matrix().dequantize_row(row, &mut position);
            for (value, shift) in embedded.iter_mut().zip(position) {
                *value += shift;
            }
        }
        let span = Span {
            rows: TOKENS,
            first: 0,
            mask: Mask::Causal,
        };
        let Activations { stack, cache } = activations;
        team.run(|ctx| self.body.forward(ctx, cache, stack, span));
        let states = &stack.hidden.get_mut()[..TOKENS * HIDDEN];
        self.norm.apply(states, TEXT_SHAPE.norm_eps, out);
    }
}

impl Load for TextEncoder {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let body = Transformer::load(store, prefix, TEXT_SHAPE)?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            embedding: scope.load("embedding", [VOCABULARY, HIDDEN])?,
            positions: scope.load("positions", [TOKENS, HIDDEN])?,
            body,
            norm: scope.load("norm", HIDDEN)?,
        })
    }
}
