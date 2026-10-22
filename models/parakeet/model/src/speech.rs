//! The speech head over the subsampler's frames, and the speech regions its probabilities mark.
//!
//! `forward` runs inside a team run, after the subsampler: the projection by output columns, the
//! context convolution by output columns, then the logits by frames.

use blocks::conformer::subsampling::Frames;
use kernels::ops::{sigmoid, silu};
use kernels::{COLUMN_ALIGN, dot_f32};
use serde::Serialize;
use team::{Ctx, SharedBuf};

use crate::model::{Dense, FRAME_SECONDS, SPEECH_HIDDEN, SPEECH_KERNEL, SpeechHead};

/// A frame is speech from this probability on.
const THRESHOLD: f32 = 0.5;
/// Shorter runs of speech are dropped.
const MIN_SPEECH_SECONDS: f64 = 0.1;
/// Shorter gaps between two runs of speech are bridged.
const MIN_GAP_SECONDS: f64 = 0.1;

/// A stretch of a recording, in seconds from its start.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

/// The head's activations for up to `frames` subsampled frames.
pub(crate) struct Buffers {
    projected: SharedBuf,
    context: SharedBuf,
    pub(crate) probabilities: SharedBuf,
}

impl Buffers {
    pub(crate) fn new(frames: usize) -> Self {
        Self {
            projected: SharedBuf::new(frames * SPEECH_HIDDEN),
            context: SharedBuf::new(frames * SPEECH_HIDDEN),
            probabilities: SharedBuf::new(frames),
        }
    }
}

impl SpeechHead {
    /// The speech probability of every frame of `hidden` (`frames.all` rows) into
    /// `buffers.probabilities`.
    pub(crate) fn forward(&self, ctx: &Ctx, hidden: &SharedBuf, frames: Frames, buffers: &Buffers) {
        let rows = frames.all;
        let input = &hidden.read()[..rows * self.projection.inputs()];
        convolve(ctx, &self.projection, input, 1, &buffers.projected);
        ctx.barrier();
        let input = &buffers.projected.read()[..rows * SPEECH_HIDDEN];
        convolve(ctx, &self.context, input, SPEECH_KERNEL, &buffers.context);
        ctx.barrier();
        let out = &self.out;
        let context = buffers.context.read();
        for row in ctx.split(rows, 1) {
            let frame = &context[row * SPEECH_HIDDEN..][..SPEECH_HIDDEN];
            let logit = out.bias.values()[0] + dot_f32(&out.weight, frame);
            buffers.probabilities.set(row, sigmoid(logit));
        }
        ctx.barrier();
    }
}

impl Dense {
    fn inputs(&self) -> usize {
        self.weight.len() / self.bias.values().len()
    }
}

/// `out = silu(dense(window))` for this thread's output columns, the window of each row its
/// `taps` neighbouring rows of `input`, centred (zeros beyond the rows).
fn convolve(ctx: &Ctx, dense: &Dense, input: &[f32], taps: usize, out: &SharedBuf) {
    let (inputs, outputs) = (dense.inputs(), dense.bias.values().len());
    let width = inputs / taps;
    let rows = input.len() / width;
    for col in ctx.split(outputs, COLUMN_ALIGN) {
        let filter = &dense.weight[col * inputs..(col + 1) * inputs];
        for row in 0..rows {
            let mut sum = dense.bias.values()[col];
            for tap in 0..taps {
                let Some(source) = (row + tap).checked_sub(taps / 2).filter(|&at| at < rows) else {
                    continue;
                };
                let values = &input[source * width..(source + 1) * width];
                sum += dot_f32(&filter[tap * width..(tap + 1) * width], values);
            }
            out.set(row * outputs + col, silu(sum));
        }
    }
}

/// The runs of frames whose probability reaches `THRESHOLD`, gaps shorter than
/// `MIN_GAP_SECONDS` bridged and runs shorter than `MIN_SPEECH_SECONDS` dropped.
pub(crate) fn regions(probabilities: &[f32]) -> Vec<Span> {
    let mut runs: Vec<Span> = Vec::new();
    let mut start = None;
    for (frame, &probability) in probabilities.iter().chain([&0.0]).enumerate() {
        let seconds = frame as f64 * FRAME_SECONDS;
        match (start, probability >= THRESHOLD) {
            (None, true) => start = Some(seconds),
            (Some(begun), false) => {
                start = None;
                match runs.last_mut() {
                    Some(last) if begun - last.end < MIN_GAP_SECONDS => last.end = seconds,
                    _ => runs.push(Span {
                        start: begun,
                        end: seconds,
                    }),
                }
            }
            _ => {}
        }
    }
    runs.retain(|run| run.end - run.start >= MIN_SPEECH_SECONDS);
    runs
}

#[cfg(test)]
mod tests {
    use super::{Span, regions};

    /// Frames are 80 ms: the one-frame gap is bridged, the two-frame gap is not, and the
    /// one-frame run after it is dropped.
    #[test]
    fn marks_runs_of_speech_bridging_short_gaps() {
        let probabilities = [0.9, 0.9, 0.1, 0.9, 0.2, 0.2, 0.9, 0.1];
        let runs = regions(&probabilities);
        assert_eq!(runs.len(), 1);
        let Span { start, end } = runs[0];
        assert_eq!(start, 0.0);
        assert!((end - 0.32).abs() < 1e-12, "{end}");
    }
}
