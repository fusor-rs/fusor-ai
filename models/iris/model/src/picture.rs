//! Depth as a picture (`colorize`): near is bright, scaled to the prediction's own 2nd to 98th
//! percentile range, through matplotlib's `inferno`.

use crate::inferno::INFERNO;
use crate::model::{IMAGE_CHANNELS, RGBA_CHANNELS};

const LOW_PERCENTILE: f64 = 2.0;
const HIGH_PERCENTILE: f64 = 98.0;
/// The smallest range the prediction is scaled by.
const MIN_RANGE: f64 = 1e-6;

/// RGB bytes, one pixel per value of `depth`.
pub fn colorize(depth: &[f32]) -> Vec<u8> {
    let mut sorted = depth.to_vec();
    sorted.sort_by(f32::total_cmp);
    let low = percentile(&sorted, LOW_PERCENTILE);
    let high = percentile(&sorted, HIGH_PERCENTILE);
    let range = (high - low).max(MIN_RANGE);
    let last = INFERNO.len() - 1;
    depth
        .iter()
        .flat_map(|&value| {
            let near = ((high - f64::from(value)) / range).clamp(0.0, 1.0);
            INFERNO[((near * INFERNO.len() as f64) as usize).min(last)]
        })
        .collect()
}

/// RGB bytes as opaque RGBA.
pub(crate) fn opaque(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / IMAGE_CHANNELS * RGBA_CHANNELS);
    for pixel in rgb.as_chunks::<IMAGE_CHANNELS>().0 {
        rgba.extend(pixel);
        rgba.push(u8::MAX);
    }
    rgba
}

/// NumPy's linear percentile `percent` of the ascending `sorted`: the interpolation between the
/// two values around `percent / 100 · (len − 1)`, from the nearer one.
fn percentile(sorted: &[f32], percent: f64) -> f64 {
    let index = percent / 100.0 * (sorted.len() - 1) as f64;
    let below = index.floor() as usize;
    let above = (below + 1).min(sorted.len() - 1);
    let fraction = index - below as f64;
    let (first, second) = (f64::from(sorted[below]), f64::from(sorted[above]));
    let span = second - first;
    if fraction < 0.5 {
        first + span * fraction
    } else {
        second - span * (1.0 - fraction)
    }
}
