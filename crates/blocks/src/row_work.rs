//! The per-thread scratch a diffusion transformer's row work shares: projecting rows through a
//! linear and RMS-normalizing runs of them.

use kernels::conv::{Patches, project_rows};
use kernels::ops::rms_norm;
use kernels::{Linear, Vector};
use team::{Ctx, PerThread, SharedBuf};

/// One thread's row on its way through a norm, and a token's rotary angles.
#[derive(Default)]
pub(crate) struct RowScratch {
    pub(crate) row: Vec<f32>,
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
}

pub(crate) struct RowWork {
    patches: PerThread<Patches>,
    scratch: PerThread<RowScratch>,
}

impl RowWork {
    pub(crate) fn new(threads: usize) -> Self {
        Self {
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, RowScratch::default),
        }
    }

    /// This thread's scratch.
    pub(crate) fn scratch(&self, ctx: &Ctx) -> &mut RowScratch {
        self.scratch.get(ctx)
    }

    /// `store(row, col, value)` of `linear` over the first `rows` rows of `input`; no barrier.
    pub(crate) fn project(
        &self,
        ctx: &Ctx,
        linear: &Linear,
        (input, rows): (&[f32], usize),
        store: impl FnMut(usize, usize, f32),
    ) {
        project_rows(ctx, linear, (input, rows), self.patches.get(ctx), store);
    }

    /// As `project`, then a barrier.
    pub(crate) fn project_synced(
        &self,
        ctx: &Ctx,
        linear: &Linear,
        (input, rows): (&SharedBuf, usize),
        store: impl FnMut(usize, usize, f32),
    ) {
        self.project(ctx, linear, (input.read(), rows), store);
        ctx.barrier();
    }

    /// This thread's rows of `target` (`rows` rows of `stride` values), their `weight`-wide run at
    /// `offset` RMS-normalized in place.
    pub(crate) fn rms_rows(
        &self,
        ctx: &Ctx,
        target: &SharedBuf,
        (rows, stride, offset): (usize, usize, usize),
        (weight, eps): (&Vector, f32),
    ) {
        let width = weight.values().len();
        let scratch = &mut self.scratch(ctx).row;
        scratch.resize(width, 0.0);
        for row in ctx.split(rows, 1) {
            let start = row * stride + offset;
            let run = target.part(start..start + width);
            rms_norm(run, weight.values(), eps, scratch);
            run.copy_from_slice(scratch);
        }
    }
}
