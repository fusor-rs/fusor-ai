//! The subsampler's dimensions: its shape, and the rows and frequencies of every stage.

use std::ops::Range;

/// The dimensions of one subsampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Mel bins of the input features.
    pub mels: usize,
    /// Channels of the convolutions.
    pub channels: usize,
    /// The side of the square windows, their stride, and how many stages there are.
    pub kernel: usize,
    pub stride: usize,
    pub stages: usize,
    /// The width of an encoder frame.
    pub hidden: usize,
}

impl Shape {
    /// Mel frames per encoder frame.
    pub const fn factor(&self) -> usize {
        self.stride.pow(self.stages as u32)
    }

    /// Outputs of a stage over `inputs` rows (or frequencies).
    pub(super) fn stage_outputs(&self, inputs: usize) -> usize {
        let padding = self.kernel / 2;
        (inputs + 2 * padding - self.kernel) / self.stride + 1
    }

    /// The input rows a stage reads for its output rows `rows`, within `available` input rows.
    pub(super) fn reads(&self, rows: &Range<usize>, available: usize) -> Range<usize> {
        let padding = self.kernel / 2;
        let first = (self.stride * rows.start).saturating_sub(padding);
        let last = self.stride * (rows.end - 1) + self.kernel - padding;
        first..last.min(available)
    }

    /// The frequencies after every stage, the first the mel bins.
    pub(super) fn widths(&self) -> Vec<usize> {
        let mut width = self.mels;
        let mut widths = vec![width];
        for _ in 0..self.stages {
            width = self.stage_outputs(width);
            widths.push(width);
        }
        widths
    }

    /// Values of one subsampled frame before the last linear: the last stage's frequencies of
    /// every channel.
    pub(super) fn stacked_width(&self) -> usize {
        self.widths().last().expect("the mel bins") * self.channels
    }
}

/// Rows of a stage's output: all of them, and the leading ones that are valid. For the last
/// stage, the encoder frames of `mel_frames` mel frames: the subsampler's output rows, of which
/// the first `valid` are the encoder's frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frames {
    pub all: usize,
    pub valid: usize,
}

impl Frames {
    pub fn of(shape: &Shape, mel_frames: usize) -> Self {
        *lengths(shape, mel_frames)
            .last()
            .expect("a subsampler has stages")
    }
}

/// Every stage's output rows for `mel_frames` mel frames.
pub(super) fn lengths(shape: &Shape, mel_frames: usize) -> Vec<Frames> {
    let mut length = Frames {
        all: mel_frames + 1,
        valid: mel_frames,
    };
    let stages = (0..shape.stages).map(|_| {
        length = Frames {
            all: shape.stage_outputs(length.all),
            valid: shape.stage_outputs(length.valid),
        };
        length
    });
    stages.collect()
}
