//! Mel power spectrograms as audio models' reference preprocessing computes them (`torch.stft`
//! with `center=False`, the Slaney filterbank of librosa and transformers): every `hop` samples,
//! a periodic Hann window of `fft` samples, the power of its real FFT, through `mels` triangular
//! filters spaced evenly on the Slaney mel scale from 0 Hz to the Nyquist frequency, each scaled
//! to unit area (`2 / (high − low)`). The filterbank is computed in f64 and rounded to f32, as
//! the references compute it.

use std::f64::consts::TAU;
use std::sync::Arc;

use kernels::dot_f32;
use realfft::{RealFftPlanner, RealToComplex};

/// The Slaney scale's corner between its linear and logarithmic parts.
const LINEAR_HZ: f64 = 1000.0;
/// Hz per mel in the linear part.
const HZ_PER_MEL: f64 = 200.0 / 3.0;
const LINEAR_MEL: f64 = LINEAR_HZ / HZ_PER_MEL;
/// The logarithmic part: frequencies grow 6.4-fold over 27 mels.
const OCTAVE_RATIO: f64 = 6.4;
const OCTAVE_MELS: f64 = 27.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub sample_rate: u32,
    /// Samples of a window, and of its FFT.
    pub fft: usize,
    /// Samples from one window's start to the next's.
    pub hop: usize,
    pub mels: usize,
}

impl Shape {
    /// Frequency bins of the FFT.
    pub fn bins(&self) -> usize {
        self.fft / 2 + 1
    }

    /// Whole windows in `samples` samples.
    pub fn frames(&self, samples: usize) -> usize {
        samples
            .checked_sub(self.fft)
            .map_or(0, |rest| rest / self.hop + 1)
    }
}

pub struct MelSpectrogram {
    shape: Shape,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// `mels × bins` row-major.
    filters: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new(shape: Shape) -> Self {
        let window = (0..shape.fft)
            .map(|index| (0.5 - 0.5 * (TAU * index as f64 / shape.fft as f64).cos()) as f32)
            .collect();
        Self {
            shape,
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(shape.fft),
            window,
            filters: slaney_filters(&shape),
        }
    }

    /// The mel powers of every whole window of `samples`, `frames × mels` row-major, into `out`.
    /// Returns the frames.
    pub fn power(&self, samples: &[f32], out: &mut Vec<f32>) -> usize {
        let Shape { fft, hop, mels, .. } = self.shape;
        let frames = self.shape.frames(samples.len());
        out.clear();
        out.resize(frames * mels, 0.0);
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0; self.shape.bins()];
        for (frame, row) in out.chunks_exact_mut(mels).enumerate() {
            let windowed = samples[frame * hop..][..fft].iter().zip(&self.window);
            for (slot, (sample, weight)) in input.iter_mut().zip(windowed) {
                *slot = sample * weight;
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("the buffers come from the plan");
            for (bin_power, bin) in power.iter_mut().zip(&spectrum) {
                *bin_power = bin.norm_sqr();
            }
            let filters = self.filters.chunks_exact(power.len());
            for (mel, filter) in row.iter_mut().zip(filters) {
                *mel = dot_f32(filter, &power);
            }
        }
        frames
    }
}

/// `mels × bins` row-major.
fn slaney_filters(shape: &Shape) -> Vec<f32> {
    let nyquist = f64::from(shape.sample_rate) / 2.0;
    let edges: Vec<f64> = linspace(hz_to_mel(0.0), hz_to_mel(nyquist), shape.mels + 2)
        .into_iter()
        .map(mel_to_hz)
        .collect();
    let frequencies = linspace(0.0, nyquist, shape.bins());
    let mut filters = Vec::with_capacity(shape.mels * frequencies.len());
    for edge in edges.windows(3) {
        let [low, centre, high] = [edge[0], edge[1], edge[2]];
        let area = 2.0 / (high - low);
        filters.extend(frequencies.iter().map(|&frequency| {
            let rising = (frequency - low) / (centre - low);
            let falling = (high - frequency) / (high - centre);
            (rising.min(falling).max(0.0) * area) as f32
        }));
    }
    filters
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < LINEAR_HZ {
        hz / HZ_PER_MEL
    } else {
        LINEAR_MEL + (hz / LINEAR_HZ).ln() * (OCTAVE_MELS / OCTAVE_RATIO.ln())
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < LINEAR_MEL {
        mel * HZ_PER_MEL
    } else {
        LINEAR_HZ * ((mel - LINEAR_MEL) * (OCTAVE_RATIO.ln() / OCTAVE_MELS)).exp()
    }
}

/// `count` values from `start` to `stop` evenly, as numpy's `linspace`.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let step = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count)
        .map(|index| start + index as f64 * step)
        .collect();
    values[count - 1] = stop;
    values
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::{MelSpectrogram, Shape};

    /// Below 1 kHz the Slaney scale is linear, so 3 filters to 1 kHz have edges every 250 Hz and
    /// area weights `2 / 500`. A cosine on bin 2 (250 Hz) of a 16-point FFT has power
    /// `(16 / 4)² = 16` there through a periodic Hann window and `(16 / 8)² = 4` on bins 1 and 3
    /// (125 and 375 Hz), where the first filter weighs half its peak and the second half of
    /// its own on bin 3: mels `0.002 · 4 + 0.004 · 16 + 0.002 · 4 = 0.08` and `0.002 · 4`.
    #[test]
    fn filters_a_bin_centred_cosine_through_slaney_triangles() {
        let shape = Shape {
            sample_rate: 2000,
            fft: 16,
            hop: 4,
            mels: 3,
        };
        let samples: Vec<f32> = (0..16 + 2 * 4 + 3)
            .map(|index| (TAU * 2.0 * index as f32 / 16.0).cos())
            .collect();
        let mut out = Vec::new();
        let frames = MelSpectrogram::new(shape).power(&samples, &mut out);
        assert_eq!(frames, 3);
        for (got, want) in out.iter().zip([0.08, 0.008, 0.0].repeat(frames)) {
            assert!((got - want).abs() < 1e-6, "{out:?}");
        }
    }
}
