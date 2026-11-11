//! Iris's text adapter (`LayerwiseTextEmbedder` and `y_pos_embedding`): each token's 12 layer
//! states attend to each other in two plain DiT layers (`blocks::dit`, `Kind::Plain`) and pool
//! into one state (`Σ_l w_l·h_l + b`); a linear and two refiner layers (`transformer`,
//! `Rotation::Partial(0)`) where the prompt's rows see each other and a padding row sees them and
//! itself (a second call on the prompt's cache); a final RMSNorm and the learned positions. The
//! rows after the prompt are zeros, so every padding row comes out of the layers and the refiner
//! as the same state: one padding row is computed and stands for them all.
//!
//! Tensors under a prefix `p`: `p.layers.blocks.{0,1}.*`, `p.pool.{weight, bias}`, `p.input`,
//! `p.refiner.layers.{0,1}.*`, `p.norm`; the positions are the model's `dit.positions`.
//! `condition` runs inside a team run.

use blocks::dit::{self as layerwise, Sequences};
use blocks::transformer::{self, KvCache, Mask, Span, Transformer};
use bundle::{Load, Scope, Store};
use kernels::conv::{Patches, project_rows};
use kernels::ops::rms_norm;
use kernels::{Linear, Table, Vector};
use team::{Ctx, PerThread, SharedBuf};

use crate::Error;
use crate::model::{ADAPTER_SHAPE, HIDDEN, NORM_EPS, REFINER_SHAPE, TEXT_LAYERS, TEXT_ROWS};

/// Tokens whose layer states the layer-wise blocks run on at a time.
const CHUNK_TOKENS: usize = 64;

pub(crate) struct Adapter {
    layers: layerwise::Dit,
    pool: Vector,
    pool_bias: Vector,
    input: Linear,
    refiner: Transformer,
    norm: Vector,
    positions: Table,
}

pub(crate) struct AdapterBuffers {
    layers: layerwise::Buffers,
    pooled: SharedBuf,
    refiner: transformer::Buffers,
    padding: transformer::Buffers,
    cache: KvCache,
    patches: PerThread<Patches>,
    row: PerThread<Vec<f32>>,
}

impl Adapter {
    pub(crate) fn load(store: &mut Store, prefix: &str, positions: &str) -> Result<Self, Error> {
        let mut scope = Scope::new(store, prefix);
        let layers = scope.load("layers", ADAPTER_SHAPE)?;
        let pool = scope.load("pool.weight", TEXT_LAYERS.len())?;
        let pool_bias = scope.load("pool.bias", 1)?;
        let input = scope.load("input", [HIDDEN; 2])?;
        let refiner = scope.load("refiner", REFINER_SHAPE)?;
        let norm = scope.load("norm", HIDDEN)?;
        Ok(Self {
            layers,
            pool,
            pool_bias,
            input,
            refiner,
            norm,
            positions: Table::load(store, positions, [TEXT_ROWS, HIDDEN])?,
        })
    }

    pub(crate) fn buffers(threads: usize) -> AdapterBuffers {
        let layer_rows = CHUNK_TOKENS * TEXT_LAYERS.len();
        AdapterBuffers {
            layers: ADAPTER_SHAPE.buffers(layer_rows, threads),
            pooled: SharedBuf::new(TEXT_ROWS * HIDDEN),
            refiner: REFINER_SHAPE.buffers(TEXT_ROWS, threads),
            padding: REFINER_SHAPE.buffers(1, threads),
            cache: KvCache::new(REFINER_SHAPE, TEXT_ROWS),
            patches: PerThread::new(threads, Patches::default),
            row: PerThread::new(threads, Vec::new),
        }
    }

    /// `out` (`TEXT_ROWS × HIDDEN`) = the adapted text of `states` (`rows × 12 × HIDDEN`,
    /// `rows ≥ 1`).
    pub(crate) fn condition(
        &self,
        ctx: &Ctx,
        buffers: &AdapterBuffers,
        (states, rows): (&[f32], usize),
        out: &SharedBuf,
    ) {
        let kept = (rows + 1).min(TEXT_ROWS);
        for first in (0..kept).step_by(CHUNK_TOKENS) {
            let tokens = CHUNK_TOKENS.min(kept - first);
            self.pool_tokens(ctx, buffers, states, (first, tokens));
        }
        self.refine(ctx, buffers, (kept, rows));
        self.finish(ctx, buffers, rows, out);
    }

    /// `pooled[first..first + tokens]` = each token's layer states through the layer-wise blocks
    /// and the pool; a token past the prompt has zero states. Ends with a barrier.
    fn pool_tokens(
        &self,
        ctx: &Ctx,
        buffers: &AdapterBuffers,
        states: &[f32],
        (first, tokens): (usize, usize),
    ) {
        let layers = TEXT_LAYERS.len();
        let hidden = &buffers.layers.hidden;
        for row in ctx.split(tokens * layers, 1) {
            let target = hidden.part(row * HIDDEN..(row + 1) * HIDDEN);
            let source = (first * layers + row) * HIDDEN;
            match states.get(source..source + HIDDEN) {
                Some(row_states) => target.copy_from_slice(row_states),
                None => target.fill(0.0),
            }
        }
        ctx.barrier();
        let sequences = Sequences {
            rows: tokens * layers,
            length: layers,
        };
        self.layers.forward(ctx, &buffers.layers, sequences);
        let hidden = hidden.read();
        for token in ctx.split(tokens, 1) {
            let pooled = buffers
                .pooled
                .part((first + token) * HIDDEN..(first + token + 1) * HIDDEN);
            pooled.fill(self.pool_bias.values()[0]);
            for (layer, weight) in self.pool.values().iter().enumerate() {
                let state = &hidden[(token * layers + layer) * HIDDEN..][..HIDDEN];
                for (sum, value) in pooled.iter_mut().zip(state) {
                    *sum += weight * value;
                }
            }
        }
        ctx.barrier();
    }

    /// The refiner's hidden rows = the first `rows` pooled tokens through `input` and the
    /// refiner's layers, and the padding row's hidden row = pooled token `rows`, when `kept`
    /// has one, through them too: it sees the prompt's keys and its own.
    fn refine(&self, ctx: &Ctx, buffers: &AdapterBuffers, (kept, rows): (usize, usize)) {
        let (prompt, padding) = (&buffers.refiner.hidden, &buffers.padding.hidden);
        let patches = buffers.patches.get(ctx);
        let pooled = (buffers.pooled.read(), kept);
        project_rows(ctx, &self.input, pooled, patches, |row, col, value| {
            if row < rows {
                prompt.set(row * HIDDEN + col, value);
            } else {
                padding.set(col, value);
            }
        });
        ctx.barrier();
        let prompt_span = Span {
            rows,
            first: 0,
            mask: Mask::Full,
        };
        self.refiner
            .forward(ctx, &buffers.cache, &buffers.refiner, prompt_span);
        if kept > rows {
            let padding_span = Span {
                rows: 1,
                first: rows,
                mask: Mask::Causal,
            };
            self.refiner
                .forward(ctx, &buffers.cache, &buffers.padding, padding_span);
        }
    }

    /// `out[t]` = the final norm of the refiner's row `t` (the padding row for every row from
    /// the prompt's `rows` on) plus the learned position `t`; then a barrier.
    fn finish(&self, ctx: &Ctx, buffers: &AdapterBuffers, rows: usize, out: &SharedBuf) {
        let (prompt, padding) = (buffers.refiner.hidden.read(), buffers.padding.hidden.read());
        let positions = self.positions.matrix();
        let position = buffers.row.get(ctx);
        position.resize(HIDDEN, 0.0);
        for row in ctx.split(TEXT_ROWS, 1) {
            let target = out.part(row * HIDDEN..(row + 1) * HIDDEN);
            let state = if row < rows {
                &prompt[row * HIDDEN..(row + 1) * HIDDEN]
            } else {
                padding
            };
            rms_norm(state, self.norm.values(), NORM_EPS, target);
            positions.dequantize_row(row, position);
            for (value, added) in target.iter_mut().zip(position.iter()) {
                *value += added;
            }
        }
        ctx.barrier();
    }
}
