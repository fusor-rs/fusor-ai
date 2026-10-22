//! The flow LM: the voice and the text through the transformer, then one latent per frame from
//! the previous one: the transformer at the next position, its final norm `t`, the end-of-speech
//! logit `eos · t` and the one-step flow head, `latent = noise + head(t, noise)`.

use std::ops::Range;

use blocks::adaln_mlp;
use blocks::transformer::{Buffers, KvCache, Mask, Span};
use kernels::QuantizedRows;
use team::{PerThread, SharedBuf, Team};

use crate::model::{FLOW_LM_SHAPE, FlowLm, HIDDEN, LATENT, OUT_NORM_EPS};

/// Rows per transformer call while prompting.
const PROMPT_ROWS: usize = 64;

/// One frame's sample.
#[derive(Debug, Clone)]
pub struct FlowStep {
    /// The latent, in the model's normalized space (the next step's input).
    pub latent: Vec<f32>,
    /// End of speech when above `session::EOS_THRESHOLD`.
    pub eos_logit: f32,
}

/// The transformer's KV cache, the activations of prompts and steps, and the next position.
pub(crate) struct LmState {
    cache: KvCache,
    buffers: Buffers,
    flow: adaln_mlp::Buffers,
    /// `t`, the final norm of the step's hidden state, per thread.
    normed: PerThread<Vec<f32>>,
    eos_logit: SharedBuf,
    /// The position of the next step.
    position: usize,
}

impl LmState {
    /// Room for `capacity` positions.
    pub(crate) fn new(lm: &FlowLm, capacity: usize, threads: usize) -> Self {
        Self {
            cache: KvCache::new(FLOW_LM_SHAPE, capacity),
            buffers: FLOW_LM_SHAPE.buffers(PROMPT_ROWS, threads),
            flow: adaln_mlp::Buffers::new(&lm.flow, threads),
            normed: PerThread::new(threads, || vec![0.0; HIDDEN]),
            eos_logit: SharedBuf::new(1),
            position: 0,
        }
    }
}

impl FlowLm {
    /// Runs the embedded `positions` through the transformer, `embed(i, row)` writing the
    /// embedding of the `i`-th; the next step follows them.
    pub(crate) fn prompt(
        &self,
        team: &Team,
        state: &mut LmState,
        positions: Range<usize>,
        embed: impl Fn(usize, &mut [f32]),
    ) {
        for first in positions.clone().step_by(PROMPT_ROWS) {
            let rows = (positions.end - first).min(PROMPT_ROWS);
            let hidden = &mut state.buffers.hidden.get_mut()[..rows * HIDDEN];
            for (offset, row) in hidden.as_chunks_mut::<HIDDEN>().0.iter_mut().enumerate() {
                embed(first - positions.start + offset, row);
            }
            let span = Span {
                rows,
                first,
                mask: Mask::Causal,
            };
            let state = &*state;
            team.run(|ctx| self.body.forward(ctx, &state.cache, &state.buffers, span));
        }
        state.position = positions.end;
    }

    /// One frame at the next position from the previous latent (or BOS) and the noise.
    pub(crate) fn step(
        &self,
        team: &Team,
        state: &mut LmState,
        input: &[f32],
        noise: &[f32],
    ) -> FlowStep {
        let hidden = &mut state.buffers.hidden.get_mut()[..HIDDEN];
        self.input
            .apply(input, &mut QuantizedRows::default(), hidden);
        let span = Span {
            rows: 1,
            first: state.position,
            mask: Mask::Causal,
        };
        state.position += 1;
        let state = &*state;
        team.run(|ctx| {
            self.body.forward(ctx, &state.cache, &state.buffers, span);
            let normed = state.normed.get(ctx);
            let hidden = &state.buffers.hidden.read()[..HIDDEN];
            self.norm.apply(hidden, OUT_NORM_EPS, normed);
            if ctx.is_first() {
                let mut logit = [0.0];
                let quant = &mut QuantizedRows::default();
                self.eos.apply(normed, quant, &mut logit);
                state.eos_logit.set(0, logit[0]);
            }
            self.flow.forward(ctx, &state.flow, normed, noise);
        });
        let velocity = &state.flow.output.read()[..LATENT];
        let latent = noise
            .iter()
            .zip(velocity)
            .map(|(start, velocity)| start + velocity)
            .collect();
        FlowStep {
            latent,
            eos_logit: state.eos_logit.get(0),
        }
    }
}
