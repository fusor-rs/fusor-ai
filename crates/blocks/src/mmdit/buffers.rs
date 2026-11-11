//! The activations of one multimodal diffusion transformer call.

use team::{PerThread, SharedBuf};

use super::{Capacity, Shape};
use crate::attention;
use crate::row_work::RowWork;

/// Activations for calls of up to a `Capacity`'s rows.
pub struct Buffers {
    pub(super) capacity: Capacity,
    /// The calls' rows, rows of `hidden` values: in and out.
    pub hidden: SharedBuf,
    pub(super) normed: SharedBuf,
    pub(super) attention: SharedBuf,
    pub(super) qkv: SharedBuf,
    /// A dual-stream block's MLP activations as rows of `feed_forward`; a single-stream block's
    /// attention output, then its MLP activations, as rows of `hidden + feed_forward`.
    pub(super) inner: SharedBuf,
    pub(super) cross_query: SharedBuf,
    pub(super) crossed: SharedBuf,
    pub(super) cross_context: SharedBuf,
    pub(super) scratch: PerThread<attention::Scratch>,
    pub(super) rows: RowWork,
}

impl Shape {
    /// Activations for calls of up to `capacity` rows.
    pub fn buffers(&self, capacity: Capacity, threads: usize) -> Buffers {
        let new = SharedBuf::new;
        let hidden = self.hidden;
        Buffers {
            capacity,
            hidden: new(capacity.rows * hidden),
            normed: new(capacity.rows * hidden),
            attention: new(capacity.rows * hidden),
            qkv: new(capacity.rows * 3 * hidden),
            inner: new(capacity.rows * (hidden + self.feed_forward)),
            cross_query: new(capacity.stream * hidden),
            crossed: new(capacity.stream * hidden),
            cross_context: new(capacity.text * 2 * hidden),
            scratch: PerThread::new(threads, attention::Scratch::default),
            rows: RowWork::new(threads),
        }
    }
}
