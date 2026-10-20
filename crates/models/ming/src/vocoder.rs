//! The VAE decoder as a streaming vocoder (plan §1.6): latents in, 44.1 kHz samples out, equal
//! to the reference's one-shot `AudioVAE.decode` of the whole utterance.
//!
//! Per latent frame `i`: `h_i = fc1(latent_i)`, linear upsampling ×4 (`align_corners=False`)
//! gives positions `4i + j` from `h_{i−1}, h_i, h_{i+1}` with edge clamping, so a frame's last two
//! positions wait for the next frame; the Qwen2 backbone attends over a 64-position window;
//! `head.out` gives `log|S|` and phase for 1765 bins; `irfft(n = 3528)` (1/n scaling) times the
//! window is overlap-added at hop 882 and divided by the summed squared window, dropping
//! `(3528 − 882) / 2` samples at both ends ("same" padding). A sample is final once the position
//! covering its start has been added, which is what `push` emits.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner};

use kernels::{Linear, QuantizedRows, Tensor, matmul};
use nn::qwen2::{Buffers, KvCache, Mask, Qwen2, Span};
use team::{PerThread, SharedBuf, Team};

use crate::model::{HIDDEN, QWEN2_SHAPE};

const HOP: usize = 882;
const N_FFT: usize = 4 * HOP;
const BINS: usize = N_FFT / 2 + 1;
const PAD: usize = (N_FFT - HOP) / 2;
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

pub struct Vocoder {
    pub fc1: Linear,
    pub body: Qwen2,
    pub head: Linear,
    pub window: Tensor,
}

/// One utterance's decoding state.
pub struct VocoderState {
    cache: KvCache,
    buffers: Buffers,
    spectrum: SharedBuf,
    frames: SharedBuf,
    fft: Arc<dyn ComplexToReal<f32>>,
    fft_scratch: PerThread<FftScratch>,
    quant: PerThread<(Vec<f32>, QuantizedRows)>,
    previous: Option<Vec<f32>>,
    overlap: Overlap,
}

/// Overlap-add sums and squared-window sums; index 0 is padded sample `base`.
#[derive(Default)]
struct Overlap {
    positions: usize,
    sum: Vec<f32>,
    envelope: Vec<f32>,
    base: usize,
}

struct FftScratch {
    input: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    output: Vec<f32>,
}

impl VocoderState {
    pub fn new(threads: usize) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_inverse(N_FFT);
        let scratch_len = fft.get_scratch_len();
        let fft_scratch = PerThread::new(threads, || FftScratch {
            input: vec![Complex::default(); BINS],
            scratch: vec![Complex::default(); scratch_len],
            output: vec![0.0; N_FFT],
        });
        Self {
            cache: KvCache::new(QWEN2_SHAPE, RING),
            buffers: Buffers::new(QWEN2_SHAPE, UPSAMPLE, threads),
            spectrum: SharedBuf::new(UPSAMPLE * 2 * BINS),
            frames: SharedBuf::new(UPSAMPLE * N_FFT),
            fft,
            fft_scratch,
            quant: PerThread::new(threads, Default::default),
            previous: None,
            overlap: Overlap::default(),
        }
    }

    /// Forgets the utterance (the KV ring needs no clearing: positions restart at 0 and every
    /// slot is written before it is read).
    pub fn reset(&mut self) {
        self.previous = None;
        self.overlap = Overlap::default();
    }
}

impl Vocoder {
    /// Adds one latent frame; appends the samples that became final to `out`.
    pub fn push(&self, team: &Team, state: &mut VocoderState, latent: &[f32], out: &mut Vec<f32>) {
        let mut h = vec![0.0; HIDDEN];
        self.fc1
            .apply(latent, 1, &mut QuantizedRows::default(), &mut h);
        // The first frame clamps h_{−1} to h_0 and has no earlier positions to complete.
        let (previous, weights) = match state.previous.take() {
            None => (h.clone(), &EARLY[..]),
            Some(previous) => (previous, &LATE_THEN_EARLY[..]),
        };
        self.decode(
            team,
            state,
            |r, c| weights[r].0 * previous[c] + weights[r].1 * h[c],
            weights.len(),
        );
        state.previous = Some(h);
        let limit = state.overlap.positions * HOP;
        state.overlap.emit(limit, out);
    }

    /// Flushes the last frame's two positions (h_T clamped to h_{T−1}) and the tail.
    pub fn finish(&self, team: &Team, state: &mut VocoderState, out: &mut Vec<f32>) {
        let Some(last) = state.previous.take() else {
            return;
        };
        self.decode(team, state, |_, c| last[c], LATE.len());
        let limit = (state.overlap.positions - 1) * HOP + N_FFT - PAD;
        state.overlap.emit(limit, out);
    }

    /// Runs `rows` upsampled positions (`row(r, c)` gives their values) through the backbone,
    /// the head and the iFFT, then overlap-adds them.
    fn decode(
        &self,
        team: &Team,
        state: &mut VocoderState,
        row: impl Fn(usize, usize) -> f32,
        rows: usize,
    ) {
        for (i, v) in state.buffers.hidden.get_mut()[..rows * HIDDEN]
            .iter_mut()
            .enumerate()
        {
            *v = row(i / HIDDEN, i % HIDDEN);
        }
        let span = Span {
            rows,
            first: state.overlap.positions,
            mask: Mask::Window(WINDOW),
        };
        let state_ref = &*state;
        team.run(|ctx| {
            self.body
                .forward(ctx, &state_ref.cache, &state_ref.buffers, span);
            let (normed, quant) = state_ref.quant.get(ctx);
            normed.resize(rows * HIDDEN, 0.0);
            for r in 0..rows {
                self.body.final_norm(
                    &state_ref.buffers,
                    r,
                    &mut normed[r * HIDDEN..(r + 1) * HIDDEN],
                );
            }
            let w = self.head.matrix();
            let input = quant.input(&w, normed, rows);
            let bias = self.head.bias().expect("head.out has a bias");
            matmul(&w, &input, rows, ctx.split(2 * BINS, 2), |m, c, v| {
                state_ref.spectrum.set(m * 2 * BINS + c, v + bias[c])
            });
            ctx.barrier();
            for r in ctx.split(rows, 1) {
                self.inverse_fft(state_ref, state_ref.fft_scratch.get(ctx), r);
            }
        });
        let window = self.window.values().expect("window is f32");
        for frame in state.frames.read()[..rows * N_FFT].as_chunks::<N_FFT>().0 {
            state.overlap.add(frame, window);
        }
    }

    fn inverse_fft(&self, state: &VocoderState, scratch: &mut FftScratch, row: usize) {
        let spectrum = &state.spectrum.read()[row * 2 * BINS..(row + 1) * 2 * BINS];
        let (log_magnitude, phase) = spectrum.split_at(BINS);
        for ((c, &m), &p) in scratch.input.iter_mut().zip(log_magnitude).zip(phase) {
            let magnitude = m.exp().min(MAX_MAGNITUDE);
            *c = Complex::new(magnitude * p.cos(), magnitude * p.sin());
        }
        // The inverse real FFT ignores the imaginary parts of the DC and Nyquist bins, like
        // torch's irfft.
        scratch.input[0].im = 0.0;
        scratch.input[BINS - 1].im = 0.0;
        state
            .fft
            .process_with_scratch(
                &mut scratch.input,
                &mut scratch.output,
                &mut scratch.scratch,
            )
            .expect("buffer sizes come from the plan");
        let window = self.window.values().expect("window is f32");
        let out = state.frames.part(row * N_FFT..(row + 1) * N_FFT);
        let scale = 1.0 / N_FFT as f32;
        for ((o, &x), &w) in out.iter_mut().zip(&scratch.output).zip(window) {
            *o = x * scale * w;
        }
    }
}

impl Overlap {
    /// Adds one windowed frame at the next position's padded offset.
    fn add(&mut self, frame: &[f32], window: &[f32]) {
        let start = self.positions * HOP - self.base;
        let end = start + N_FFT;
        if self.sum.len() < end {
            self.sum.resize(end, 0.0);
            self.envelope.resize(end, 0.0);
        }
        for ((t, &x), &w) in frame.iter().enumerate().zip(window) {
            self.sum[start + t] += x;
            self.envelope[start + t] += w * w;
        }
        self.positions += 1;
    }

    /// Appends padded samples `[base, limit)` that lie past the leading padding, normalized by
    /// the envelope, and drops them from the buffers.
    fn emit(&mut self, limit: usize, out: &mut Vec<f32>) {
        let take = limit - self.base;
        let skip = PAD.saturating_sub(self.base).min(take);
        out.extend((skip..take).map(|i| self.sum[i] / self.envelope[i]));
        self.sum.drain(..take);
        self.envelope.drain(..take);
        self.base = limit;
    }
}
