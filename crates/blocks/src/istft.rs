//! The inverse short-time Fourier transform of a neural vocoder head, streamed frame by frame.
//!
//! `Istft::frame` turns one position's log-magnitude and phase (`n_fft / 2 + 1` bins each)
//! into a windowed time frame (`irfft` with `1/n` scaling, as `torch.fft.irfft`). `OverlapAdd`
//! adds frames at `hop` intervals and divides by the summed squared window, dropping
//! `(n_fft − hop) / 2` samples at both ends ("same" padding, as Vocos' `ISTFT`). A sample is
//! final once the frame starting at or before it is the last that can reach it, so `emit_ready`
//! streams exactly what a one-shot transform of the whole sequence would give.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner};

pub struct Istft {
    window: Vec<f32>,
    fft: Arc<dyn ComplexToReal<f32>>,
}

/// One thread's buffers for `Istft::frame`.
pub struct FftScratch {
    input: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    output: Vec<f32>,
}

impl Istft {
    /// The transform for frames of `window.len()` samples.
    pub fn new(window: Vec<f32>) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_inverse(window.len());
        Self { window, fft }
    }

    pub fn n_fft(&self) -> usize {
        self.window.len()
    }

    pub fn bins(&self) -> usize {
        self.n_fft() / 2 + 1
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    pub fn scratch(&self) -> FftScratch {
        FftScratch {
            input: vec![Complex::default(); self.bins()],
            scratch: vec![Complex::default(); self.fft.get_scratch_len()],
            output: vec![0.0; self.n_fft()],
        }
    }

    /// `out = irfft(m · e^{iφ}) · window`, with `m = min(e^{log_magnitude}, max_magnitude)`.
    pub fn frame(
        &self,
        polar: (&[f32], &[f32]),
        max_magnitude: f32,
        scratch: &mut FftScratch,
        out: &mut [f32],
    ) {
        let (log_magnitude, phase) = polar;
        let bins = scratch.input.iter_mut().zip(log_magnitude).zip(phase);
        for ((bin, &level), &angle) in bins {
            let magnitude = level.exp().min(max_magnitude);
            *bin = Complex::new(magnitude * angle.cos(), magnitude * angle.sin());
        }
        // The inverse real FFT ignores the imaginary parts of the DC and Nyquist bins, like
        // torch's irfft.
        let last = self.bins() - 1;
        scratch.input[0].im = 0.0;
        scratch.input[last].im = 0.0;
        self.fft
            .process_with_scratch(
                &mut scratch.input,
                &mut scratch.output,
                &mut scratch.scratch,
            )
            .expect("buffer sizes come from the plan");
        let scale = 1.0 / self.n_fft() as f32;
        let samples = out.iter_mut().zip(&scratch.output).zip(&self.window);
        for ((sample, &value), &weight) in samples {
            *sample = value * scale * weight;
        }
    }
}

/// Overlap-add sums and squared-window sums of a stream; index 0 is padded sample `base`.
pub struct OverlapAdd {
    n_fft: usize,
    hop: usize,
    frames: usize,
    sum: Vec<f32>,
    envelope: Vec<f32>,
    base: usize,
}

impl OverlapAdd {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Self {
            n_fft,
            hop,
            frames: 0,
            sum: Vec::new(),
            envelope: Vec::new(),
            base: 0,
        }
    }

    /// Frames added so far (the next frame's position).
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Adds one windowed frame at the next position.
    pub fn add(&mut self, frame: &[f32], window: &[f32]) {
        let start = self.frames * self.hop - self.base;
        let end = start + self.n_fft;
        if self.sum.len() < end {
            self.sum.resize(end, 0.0);
            self.envelope.resize(end, 0.0);
        }
        for ((offset, &value), &weight) in frame.iter().enumerate().zip(window) {
            self.sum[start + offset] += value;
            self.envelope[start + offset] += weight * weight;
        }
        self.frames += 1;
    }

    /// Appends the samples no later frame can change.
    pub fn emit_ready(&mut self, out: &mut Vec<f32>) {
        self.emit(self.frames * self.hop, out);
    }

    /// Appends the rest, up to the trailing padding.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        let Some(last) = self.frames.checked_sub(1) else {
            return;
        };
        self.emit(last * self.hop + self.n_fft - self.pad(), out);
    }

    fn pad(&self) -> usize {
        (self.n_fft - self.hop) / 2
    }

    /// Appends padded samples `[base, limit)` past the leading padding, normalized by the
    /// envelope, and drops them from the buffers.
    fn emit(&mut self, limit: usize, out: &mut Vec<f32>) {
        let take = limit - self.base;
        let skip = self.pad().saturating_sub(self.base).min(take);
        out.extend((skip..take).map(|i| self.sum[i] / self.envelope[i]));
        self.sum.drain(..take);
        self.envelope.drain(..take);
        self.base = limit;
    }
}
