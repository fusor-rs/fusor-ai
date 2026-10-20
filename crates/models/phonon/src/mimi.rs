//! Mimi's decoder, streaming: per latent frame, the latent projected to `HIDDEN` channels, the
//! depthwise transposed conv to `UPSAMPLE` positions at 200 Hz (each from this frame's and the
//! previous frame's projection), the transformer over a sliding window of `MIMI_WINDOW`
//! positions, then SEANet to `FRAME_SAMPLES` samples at 24 kHz.

use blocks::kyutai::{self, KvCache, Mask, Span};
use blocks::seanet;
use kernels::QuantizedRows;
use team::Team;

use crate::model::{HIDDEN, LATENT, MIMI_TRANSFORMER_SHAPE, MIMI_WINDOW, Mimi, UPSAMPLE};

/// One stream's decoding state.
pub(crate) struct MimiState {
    cache: KvCache,
    buffers: kyutai::Buffers,
    decoder: seanet::State,
    /// The previous frame's projected latent (zeros before the first).
    previous: Vec<f32>,
    /// The next 200 Hz position.
    position: usize,
}

impl MimiState {
    pub(crate) fn new(mimi: &Mimi, threads: usize) -> Self {
        Self {
            cache: KvCache::new(MIMI_TRANSFORMER_SHAPE, MIMI_WINDOW + UPSAMPLE),
            buffers: kyutai::Buffers::new(MIMI_TRANSFORMER_SHAPE, UPSAMPLE, threads),
            decoder: seanet::State::new(&mimi.decoder, UPSAMPLE, threads),
            previous: vec![0.0; HIDDEN],
            position: 0,
        }
    }

    /// Starts a new stream (the KV ring needs no clearing: positions restart at 0 and every
    /// visible slot is written before it is read).
    pub(crate) fn reset(&mut self) {
        self.decoder.reset();
        self.previous.fill(0.0);
        self.position = 0;
    }
}

impl Mimi {
    /// Decodes `latents` (whole frames of `LATENT`), appending their samples to `out`.
    pub(crate) fn decode(
        &self,
        team: &Team,
        state: &mut MimiState,
        latents: &[f32],
        out: &mut Vec<f32>,
    ) {
        for latent in latents.as_chunks::<LATENT>().0 {
            self.upsample(state, latent);
            let span = Span {
                rows: UPSAMPLE,
                first: state.position,
                mask: Mask::Window(MIMI_WINDOW),
            };
            let shared = &*state;
            team.run(|ctx| {
                self.transformer
                    .forward(ctx, &shared.cache, &shared.buffers, span);
                self.decoder
                    .forward(ctx, &shared.decoder, &shared.buffers.hidden, UPSAMPLE);
            });
            state.position += UPSAMPLE;
            let samples = self.decoder.output_rows(UPSAMPLE);
            out.extend_from_slice(&state.decoder.output.read()[..samples]);
        }
    }

    /// The transformer's input rows for one frame: `UPSAMPLE` positions of
    /// `kernel[c][j] · q[c] + kernel[c][UPSAMPLE + j] · previous[c]`.
    fn upsample(&self, state: &mut MimiState, latent: &[f32; LATENT]) {
        let kernel = self.upsample.values().expect("checked at load");
        let mut projected = vec![0.0; HIDDEN];
        self.latent
            .apply(latent, 1, &mut QuantizedRows::default(), &mut projected);
        let rows = &mut state.buffers.hidden.get_mut()[..UPSAMPLE * HIDDEN];
        for (channel, taps) in kernel.as_chunks::<{ 2 * UPSAMPLE }>().0.iter().enumerate() {
            for (j, row) in rows.as_chunks_mut::<HIDDEN>().0.iter_mut().enumerate() {
                row[channel] =
                    taps[j] * projected[channel] + taps[UPSAMPLE + j] * state.previous[channel];
            }
        }
        state.previous = projected;
    }
}
