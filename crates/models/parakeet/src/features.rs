//! The log-mel frontend: pre-emphasis, a 512-point FFT every 10 ms of a 25 ms Hann window
//! (centred on the hop, zeros around the recording), the power spectrum through the Slaney mel
//! filterbank, `ln(· + 2⁻²⁴)`, and each mel bin normalized over the recording by its mean and
//! unbiased standard deviation. A recording of `n` samples has `n / HOP` frames.

use std::ops::Range;
use std::sync::Arc;

use kernels::Tensor;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::Error;
use crate::model::{BINS, FFT, HOP, MELS};

const PREEMPHASIS: f32 = 0.97;
/// Added to every mel energy before the logarithm (2⁻²⁴).
const LOG_GUARD: f32 = 5.960_464_5e-8;
/// Added to each bin's standard deviation before dividing by it.
const DEVIATION_GUARD: f32 = 1e-5;
/// The fewest samples with two frames, the least a deviation needs.
pub const MIN_SAMPLES: usize = 2 * HOP;

pub(crate) struct Frontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    filters: Vec<f32>,
    /// The bins where each mel filter is not zero.
    supports: Vec<Range<usize>>,
}

impl Frontend {
    /// From the window (`FFT` values) and the filterbank (`[MELS, BINS]`).
    pub(crate) fn new(window: &Tensor, mel: &Tensor) -> Result<Self, Error> {
        let filters = mel.values()?.to_vec();
        let supports = filters
            .as_chunks::<BINS>()
            .0
            .iter()
            .map(|filter| {
                // The lowest filters are narrower than a bin and can be all zeros: no support.
                let first = filter.iter().position(|&weight| weight != 0.0);
                let last = filter.iter().rposition(|&weight| weight != 0.0);
                first
                    .zip(last)
                    .map_or(0..0, |(first, last)| first..last + 1)
            })
            .collect();
        Ok(Self {
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(FFT),
            window: window.values()?.to_vec(),
            filters,
            supports,
        })
    }

    /// The normalized features of `pcm` (16 kHz), `frames × MELS` row-major, into `out`.
    /// Returns the frames.
    pub(crate) fn compute(&self, pcm: &[f32], out: &mut Vec<f32>) -> Result<usize, Error> {
        if pcm.len() < MIN_SAMPLES {
            return Err(Error::Request(format!(
                "{} samples is too short: at least {MIN_SAMPLES}",
                pcm.len()
            )));
        }
        let frames = pcm.len() / HOP;
        out.clear();
        out.resize(frames * MELS, 0.0);
        let padded = emphasized(pcm);
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        for (frame, row) in out.as_chunks_mut::<MELS>().0.iter_mut().enumerate() {
            let samples = &padded[frame * HOP..frame * HOP + FFT];
            for (sample, (value, weight)) in input.iter_mut().zip(samples.iter().zip(&self.window))
            {
                *sample = value * weight;
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("the buffers come from the plan");
            self.log_mel(&spectrum, row);
        }
        normalize(out, frames);
        Ok(frames)
    }

    fn log_mel(&self, spectrum: &[Complex<f32>], row: &mut [f32]) {
        let filters = self
            .filters
            .as_chunks::<BINS>()
            .0
            .iter()
            .zip(&self.supports);
        for (energy, (filter, support)) in row.iter_mut().zip(filters) {
            let bins = spectrum[support.clone()]
                .iter()
                .zip(&filter[support.clone()]);
            let mel: f32 = bins.map(|(bin, weight)| bin.norm_sqr() * weight).sum();
            *energy = (mel + LOG_GUARD).ln();
        }
    }
}

/// The pre-emphasized recording with `FFT / 2` zeros on each side, so frame `t`'s window
/// starts at sample `t · HOP`.
fn emphasized(pcm: &[f32]) -> Vec<f32> {
    let mut padded = vec![0.0; pcm.len() + FFT];
    let signal = &mut padded[FFT / 2..FFT / 2 + pcm.len()];
    signal[0] = pcm[0];
    for (index, sample) in signal.iter_mut().enumerate().skip(1) {
        *sample = pcm[index] - PREEMPHASIS * pcm[index - 1];
    }
    padded
}

/// Each mel bin minus its mean over the frames, over its unbiased standard deviation.
fn normalize(features: &mut [f32], frames: usize) {
    for mel in 0..MELS {
        let values = || {
            features
                .iter()
                .skip(mel)
                .step_by(MELS)
                .map(|&value| f64::from(value))
        };
        let mean = values().sum::<f64>() / frames as f64;
        let variance = values().map(|value| (value - mean).powi(2)).sum::<f64>();
        let deviation = (variance / (frames - 1) as f64).sqrt() as f32;
        let mean = mean as f32;
        for value in features.iter_mut().skip(mel).step_by(MELS) {
            *value = (*value - mean) / (deviation + DEVIATION_GUARD);
        }
    }
}
