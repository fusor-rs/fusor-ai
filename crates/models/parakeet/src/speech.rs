//! The speech head over the subsampler's frames, and the speech regions its probabilities mark.
//!
//! `forward` runs inside a team run, after the subsampler: the projection by output columns, the
//! context convolution by output columns, then the logits by frames.

use blocks::conformer::subsampling::Frames;
use kernels::ops::{exp, silu};
use kernels::{COLUMN_ALIGN, Linear, MatrixStorage, dot_f32};
use serde::Serialize;
use team::{Ctx, SharedBuf};

use crate::model::{FRAME_SECONDS, SPEECH_HIDDEN, SPEECH_KERNEL, SpeechHead};

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
        let input = &hidden.read()[..rows * self.projection.matrix().cols];
        convolve(ctx, &self.projection, (input, 1), rows, &buffers.projected);
        ctx.barrier();
        let input = &buffers.projected.read()[..rows * SPEECH_HIDDEN];
        convolve(
            ctx,
            &self.context,
            (input, SPEECH_KERNEL),
            rows,
            &buffers.context,
        );
        ctx.barrier();
        let weights = f32_weights(&self.out);
        let bias = self.out.bias().map_or(0.0, |bias| bias[0]);
        let context = buffers.context.read();
        for row in ctx.split(rows, 1) {
            let logit = bias + dot_f32(weights, &context[row * SPEECH_HIDDEN..][..SPEECH_HIDDEN]);
            buffers.probabilities.set(row, 1.0 / (1.0 + exp(-logit)));
        }
        ctx.barrier();
    }
}

/// `out = silu(linear(window))` for this thread's output columns, the window of each row its
/// `taps` neighbouring rows of `input`, centred (zeros beyond the rows).
fn convolve(
    ctx: &Ctx,
    linear: &Linear,
    (input, taps): (&[f32], usize),
    rows: usize,
    out: &SharedBuf,
) {
    let matrix = linear.matrix();
    let weights = f32_weights(linear);
    let width = matrix.cols / taps;
    let bias = linear.bias();
    for col in ctx.split(matrix.rows, COLUMN_ALIGN) {
        let filter = &weights[col * matrix.cols..(col + 1) * matrix.cols];
        for row in 0..rows {
            let mut sum = bias.map_or(0.0, |bias| bias[col]);
            for tap in 0..taps {
                let Some(source) = (row + tap).checked_sub(taps / 2).filter(|&at| at < rows) else {
                    continue;
                };
                let values = &input[source * width..(source + 1) * width];
                sum += dot_f32(&filter[tap * width..(tap + 1) * width], values);
            }
            out.set(row * matrix.rows + col, silu(sum));
        }
    }
}

fn f32_weights(linear: &Linear) -> &[f32] {
    match linear.matrix().storage {
        MatrixStorage::F32(values) => values,
        MatrixStorage::Quant { .. } => unreachable!("the speech head is f32 (checked at load)"),
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
