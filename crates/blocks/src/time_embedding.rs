//! A timestep's embedding, as Wan2.1's and P-JiT's diffusion transformers compute it: the
//! timestep's sinusoids through a SiLU MLP of two linears.

use kernels::Linear;
use kernels::ops::silu;
use team::{Ctx, SharedBuf};

use crate::row_work::RowWork;
use crate::{SINUSOID_BASE, timestep_sinusoid_f64};

/// The buffers a timestep passes through on its way to the embedding.
pub(crate) struct TimeBuffers {
    frequencies: usize,
    sinusoid: SharedBuf,
    hidden: SharedBuf,
    /// The time embedding.
    pub(crate) embedding: SharedBuf,
    /// The time embedding's SiLU.
    pub(crate) activated: SharedBuf,
}

impl TimeBuffers {
    /// Buffers for `frequencies` sinusoids and an embedding of `hidden` values.
    pub(crate) fn new(frequencies: usize, hidden: usize) -> Self {
        let new = SharedBuf::new;
        Self {
            frequencies,
            sinusoid: new(frequencies),
            hidden: new(hidden),
            embedding: new(hidden),
            activated: new(hidden),
        }
    }

    /// `embedding` and `activated` for `timestep` through the two `layers`; a barrier after each
    /// layer.
    pub(crate) fn embed(&self, ctx: &Ctx, rows: &RowWork, layers: &[Linear; 2], timestep: f32) {
        if ctx.is_first() {
            let frequencies = self.sinusoid.part(0..self.frequencies);
            timestep_sinusoid_f64(timestep, SINUSOID_BASE, frequencies);
        }
        ctx.barrier();
        rows.project_synced(ctx, &layers[0], (&self.sinusoid, 1), |_, col, value| {
            self.hidden.set(col, silu(value));
        });
        rows.project_synced(ctx, &layers[1], (&self.hidden, 1), |_, col, value| {
            self.embedding.set(col, value);
            self.activated.set(col, silu(value));
        });
    }
}
