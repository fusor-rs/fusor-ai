//! The frontend's constants, as the reference computes them: a symmetric Hann window of
//! `WINDOW` samples centred in `FFT`, and the Slaney-normalized mel filterbank (linear to
//! 1 kHz, logarithmic above), in f64, then f32.

use std::f64::consts::PI;

use parakeet::SAMPLE_RATE;
use parakeet::model::{BINS, FFT, MELS, WINDOW};

/// The Slaney scale's corner between its linear and logarithmic parts.
const LINEAR_HZ: f64 = 1000.0;
const LINEAR_MEL: f64 = 15.0;
/// Hz per mel in the linear part.
const HZ_PER_MEL: f64 = 200.0 / 3.0;
/// The logarithmic part: frequencies grow 6.4-fold over 27 mels.
const OCTAVE_RATIO: f64 = 6.4;
const OCTAVE_MELS: f64 = 27.0;

pub(crate) fn window() -> Vec<f32> {
    let margin = (FFT - WINDOW) / 2;
    let mut window = vec![0.0; FFT];
    for (index, value) in window[margin..margin + WINDOW].iter_mut().enumerate() {
        let phase = 2.0 * PI * index as f64 / (WINDOW - 1) as f64;
        *value = (0.5 - 0.5 * phase.cos()) as f32;
    }
    window
}

/// `[MELS, BINS]` row-major.
pub(crate) fn mel() -> Vec<f32> {
    let nyquist = f64::from(SAMPLE_RATE) / 2.0;
    let edges: Vec<f64> = linspace(hz_to_mel(0.0), hz_to_mel(nyquist), MELS + 2)
        .into_iter()
        .map(mel_to_hz)
        .collect();
    let frequencies = linspace(0.0, nyquist, BINS);
    let mut filters = Vec::with_capacity(MELS * BINS);
    for mel in 0..MELS {
        let [low, centre, high] = [edges[mel], edges[mel + 1], edges[mel + 2]];
        let norm = 2.0 / (high - low);
        filters.extend(frequencies.iter().map(|&frequency| {
            let rising = (frequency - low) / (centre - low);
            let falling = (high - frequency) / (high - centre);
            (rising.min(falling).max(0.0) * norm) as f32
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
