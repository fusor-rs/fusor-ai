//! The flow LM: the voice and the text through the transformer, then one latent per frame from
//! the previous one: the transformer at the next position, its final norm `t`, the end-of-speech
//! logit `eos · t` and the one-step flow head, `latent = noise + head(t, noise)`.

use blocks::adaln_mlp;
use blocks::kyutai::{self, KvCache, Mask, Span};
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

/// The transformer's KV cache and the activations of prompts and steps.
pub(crate) struct LmState {
    cache: KvCache,
    buffers: kyutai::Buffers,
    flow: adaln_mlp::Buffers,
    /// `t`, the final norm of the step's hidden state, per thread.
    normed: PerThread<Vec<f32>>,
    eos_logit: SharedBuf,
}

impl LmState {
    /// Room for `capacity` positions.
    pub(crate) fn new(lm: &FlowLm, capacity: usize, threads: usize) -> Self {
        Self {
            cache: KvCache::new(FLOW_LM_SHAPE, capacity),
            buffers: kyutai::Buffers::new(FLOW_LM_SHAPE, PROMPT_ROWS, threads),
            flow: adaln_mlp::Buffers::new(&lm.flow, threads),
            normed: PerThread::new(threads, || vec![0.0; HIDDEN]),
            eos_logit: SharedBuf::new(1),
        }
    }
}

impl FlowLm {
    /// Runs `count` embedded positions from `first` through the transformer; `embed(i, row)`
    /// writes the embedding of the `i`-th.
    pub(crate) fn prompt(
        &self,
        team: &Team,
        state: &mut LmState,
        (first, count): (usize, usize),
        embed: impl Fn(usize, &mut [f32]),
    ) {
        for start in (0..count).step_by(PROMPT_ROWS) {
            let rows = (count - start).min(PROMPT_ROWS);
            let hidden = &mut state.buffers.hidden.get_mut()[..rows * HIDDEN];
            for (offset, row) in hidden.as_chunks_mut::<HIDDEN>().0.iter_mut().enumerate() {
                embed(start + offset, row);
            }
            let span = Span {
                rows,
                first: first + start,
                mask: Mask::Causal,
            };
            let state = &*state;
            team.run(|ctx| self.body.forward(ctx, &state.cache, &state.buffers, span));
        }
    }

    /// One frame at `position` from the previous latent (or BOS) and the noise.
    pub(crate) fn step(
        &self,
        team: &Team,
        state: &mut LmState,
        (input, position): (&[f32], usize),
        noise: &[f32],
    ) -> FlowStep {
        let hidden = &mut state.buffers.hidden.get_mut()[..HIDDEN];
        self.input
            .apply(input, 1, &mut QuantizedRows::default(), hidden);
        let span = Span {
            rows: 1,
            first: position,
            mask: Mask::Causal,
        };
        let state = &*state;
        team.run(|ctx| {
            self.body.forward(ctx, &state.cache, &state.buffers, span);
            let normed = state.normed.get(ctx);
            let hidden = &state.buffers.hidden.read()[..HIDDEN];
            self.norm.apply(hidden, OUT_NORM_EPS, normed);
            if ctx.is_first() {
                let mut logit = [0.0];
                let quant = &mut QuantizedRows::default();
                self.eos.apply(normed, 1, quant, &mut logit);
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
