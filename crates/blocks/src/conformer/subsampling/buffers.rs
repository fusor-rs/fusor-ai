//! The activations of one `Subsampling::forward`, and each thread's working set.

use std::ops::Range;

use kernels::QuantizedRows;
use team::{PerThread, SharedBuf};

use super::{Frames, Shape};

/// Activations of one `forward` over up to `mel_frames` mel frames, each thread computing
/// `block_frames` output frames at a time (its working set).
pub struct Buffers {
    pub(super) mel_frames: usize,
    pub(super) block_frames: usize,
    /// The last stage's rows, `frames × frequencies · channels`.
    pub(super) stacked: SharedBuf,
    /// The subsampled frames, `frames × hidden`.
    pub hidden: SharedBuf,
    pub(super) scratch: PerThread<Scratch>,
}

#[derive(Default)]
pub(super) struct Scratch {
    /// The rows of each stage a block reads, and computes.
    pub(super) reads: Vec<Range<usize>>,
    pub(super) stages: Vec<Vec<f32>>,
    pub(super) pointwise: Vec<f32>,
    pub(super) quant: QuantizedRows,
}

impl Buffers {
    pub fn new(shape: Shape, mel_frames: usize, block_frames: usize, threads: usize) -> Self {
        let frames = Frames::of(&shape, mel_frames).all;
        Self {
            mel_frames,
            block_frames,
            stacked: SharedBuf::new(frames * shape.stacked_width()),
            hidden: SharedBuf::new(frames * shape.hidden),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    pub fn mel_frames(&self) -> usize {
        self.mel_frames
    }
}
