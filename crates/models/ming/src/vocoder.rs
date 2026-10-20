//! The VAE decoder as a streaming vocoder (plan §1.6): latents in, 44.1 kHz samples out, equal
//! to the reference's one-shot `AudioVAE.decode` of the whole utterance.
//!
//! Per latent frame `i`: `h_i = fc1(latent_i)`, linear upsampling ×4 (`align_corners=False`)
//! gives positions `4i + j` from `h_{i−1}, h_i, h_{i+1}` with edge clamping, so a frame's last two
//! positions wait for the next frame; the Qwen2 backbone attends over a 64-position window;
//! `head.out` gives `log|S|` and phase for 1765 bins, and `blocks::istft` turns each position into
//! 882 new samples (`n_fft` 3528, "same" padding), streaming exactly what the one-shot decode
//! gives.

use blocks::istft::{FftScratch, Istft, OverlapAdd};
use blocks::qwen2::{Buffers, KvCache, Mask, Qwen2, Span};
use bundle::{Load, Store};
use kernels::{Linear, QuantizedRows, Tensor, matmul};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::model::{HIDDEN, QWEN2_SHAPE};

const HOP: usize = 882;
const UPSAMPLE: usize = 4;
/// Samples per latent frame (80 ms at 44.1 kHz).
pub const FRAME_SAMPLES: usize = UPSAMPLE * HOP;
const WINDOW: usize = 64;
/// Ring slots: the window plus one call's positions, rounded up.
const RING: usize = 128;
const MAX_MAGNITUDE: f32 = 100.0;
/// `(weight of h_{i−1}, weight of h_i)` for positions `4i + 0, 4i + 1` …
const EARLY: [(f32, f32); 2] = [(0.375, 0.625), (0.125, 0.875)];
/// … and for positions `4(i−1) + 2, 4(i−1) + 3`, which needed `h_i`.
const LATE: [(f32, f32); 2] = [(0.875, 0.125), (0.625, 0.375)];
/// A frame after the first completes the previous frame's last two positions and starts its own.
const LATE_THEN_EARLY: [(f32, f32); 4] = [LATE[0], LATE[1], EARLY[0], EARLY[1]];

pub(crate) struct Vocoder {
    fc1: Linear,
    body: Qwen2,
    head: Linear,
    istft: Istft,
}

/// One utterance's decoding state.
pub(crate) struct VocoderState {
    cache: KvCache,
    buffers: Buffers,
    spectrum: SharedBuf,
    frames: SharedBuf,
    fft_scratch: PerThread<FftScratch>,
    quant: PerThread<(Vec<f32>, QuantizedRows)>,
    previous: Option<Vec<f32>>,
    n_fft: usize,
    overlap: OverlapAdd,
}

impl VocoderState {
    pub(crate) fn new(vocoder: &Vocoder, threads: usize) -> Self {
        let istft = &vocoder.istft;
        Self {
            cache: KvCache::new(QWEN2_SHAPE, RING),
            buffers: Buffers::new(QWEN2_SHAPE, UPSAMPLE, threads),
            spectrum: SharedBuf::new(UPSAMPLE * 2 * istft.bins()),
            frames: SharedBuf::new(UPSAMPLE * istft.n_fft()),
            fft_scratch: PerThread::new(threads, || istft.scratch()),
            quant: PerThread::new(threads, Default::default),
            previous: None,
            n_fft: istft.n_fft(),
            overlap: OverlapAdd::new(istft.n_fft(), HOP),
        }
    }

    /// Forgets the utterance (the KV ring needs no clearing: positions restart at 0 and every
    /// slot is written before it is read).
    pub(crate) fn reset(&mut self) {
        self.previous = None;
        self.overlap = OverlapAdd::new(self.n_fft, HOP);
    }
}

impl Vocoder {
    /// Adds one latent frame; appends the samples that became final to `out`.
    pub(crate) fn push(
        &self,
        team: &Team,
        state: &mut VocoderState,
        latent: &[f32],
        out: &mut Vec<f32>,
    ) {
        let mut current = vec![0.0; HIDDEN];
        let mut quant = QuantizedRows::default();
        self.fc1.apply(latent, 1, &mut quant, &mut current);
        // The first frame clamps h_{−1} to h_0 and has no earlier positions to complete.
        let (previous, weights) = match state.previous.take() {
            None => (current.clone(), &EARLY[..]),
            Some(previous) => (previous, &LATE_THEN_EARLY[..]),
        };
        let position = |row: usize, col: usize| {
            let (previous_weight, current_weight) = weights[row];
            previous_weight * previous[col] + current_weight * current[col]
        };
        self.decode(team, state, position, weights.len());
        state.previous = Some(current);
        state.overlap.emit_ready(out);
    }

    /// Flushes the last frame's two positions (h_T clamped to h_{T−1}) and the tail.
    pub(crate) fn finish(&self, team: &Team, state: &mut VocoderState, out: &mut Vec<f32>) {
        let Some(last) = state.previous.take() else {
            return;
        };
        self.decode(team, state, |_, col| last[col], LATE.len());
        state.overlap.finish(out);
    }

    /// Runs `rows` upsampled positions (`position(row, col)` gives their values) through the
    /// backbone, the head and the inverse STFT, then overlap-adds them.
    fn decode(
        &self,
        team: &Team,
        state: &mut VocoderState,
        position: impl Fn(usize, usize) -> f32,
        rows: usize,
    ) {
        let hidden = state.buffers.hidden.get_mut();
        for (index, value) in hidden[..rows * HIDDEN].iter_mut().enumerate() {
            *value = position(index / HIDDEN, index % HIDDEN);
        }
        let span = Span {
            rows,
            first: state.overlap.frames(),
            mask: Mask::Window(WINDOW),
        };
        let shared = &*state;
        team.run(|ctx| {
            self.body.forward(ctx, &shared.cache, &shared.buffers, span);
            self.spectrum(ctx, shared, rows);
            ctx.barrier();
            self.frames(ctx, shared, rows);
        });
        let n_fft = self.istft.n_fft();
        for frame in state.frames.read()[..rows * n_fft].chunks_exact(n_fft) {
            state.overlap.add(frame, self.istft.window());
        }
    }

    /// `spectrum[row, cols] = head(norm(hidden[row]))` for this thread's columns.
    fn spectrum(&self, ctx: &Ctx, state: &VocoderState, rows: usize) {
        let width = 2 * self.istft.bins();
        let (normed, quant) = state.quant.get(ctx);
        normed.resize(rows * HIDDEN, 0.0);
        for (row, out) in normed.as_chunks_mut::<HIDDEN>().0.iter_mut().enumerate() {
            self.body.final_norm(&state.buffers, row, out);
        }
        let matrix = self.head.matrix();
        let input = quant.input(&matrix, normed, rows);
        let bias = self.head.bias().expect("head.out has a bias");
        matmul(
            &matrix,
            &input,
            rows,
            ctx.split(width, 2),
            |row, col, value| {
                state.spectrum.set(row * width + col, value + bias[col]);
            },
        );
    }

    /// The time frames of this thread's rows of `spectrum`.
    fn frames(&self, ctx: &Ctx, state: &VocoderState, rows: usize) {
        let (bins, n_fft) = (self.istft.bins(), self.istft.n_fft());
        let spectrum = state.spectrum.read();
        for row in ctx.split(rows, 1) {
            let polar = spectrum[row * 2 * bins..(row + 1) * 2 * bins].split_at(bins);
            let frame = state.frames.part(row * n_fft..(row + 1) * n_fft);
            let scratch = state.fft_scratch.get(ctx);
            self.istft.frame(polar, MAX_MAGNITUDE, scratch, frame);
        }
    }
}

/// `p.fc1`, the Qwen2 backbone under `p`, `p.head` and the iSTFT window `p.window`.
impl Load for Vocoder {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let window = Tensor::load(store, &format!("{prefix}.window"), ())?;
        let istft = Istft::new(window.values()?.to_vec());
        if istft.n_fft() != UPSAMPLE * HOP {
            return Err(bundle::Error::Format(format!(
                "an iSTFT window of {}",
                istft.n_fft()
            )));
        }
        Ok(Self {
            fc1: Linear::load(store, &format!("{prefix}.fc1"), ())?,
            body: Qwen2::load(store, prefix, QWEN2_SHAPE)?,
            head: Linear::load(store, &format!("{prefix}.head"), ())?,
            istft,
        })
    }
}
