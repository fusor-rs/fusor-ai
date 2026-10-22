//! The two planes of one `decode`: which holds the current activations, and the steps that
//! write the other.

use kernels::conv::{Conv2d, Grid};
use team::{Ctx, SharedBuf};

use super::Buffers;
use crate::resnet::{Planes, Resnet, Shared};

/// The planes of one `decode`: which holds the current activations.
pub(super) struct Walk<'call> {
    pub(super) ctx: &'call Ctx<'call>,
    pub(super) buffers: &'call Buffers,
    pub(super) shared: Shared<'call>,
    pub(super) current: usize,
    pub(super) grid: Grid,
}

impl Walk<'_> {
    pub(super) fn plane(&self) -> &SharedBuf {
        &self.buffers.planes[self.current]
    }

    /// The plane the next step writes.
    fn other(&self) -> &SharedBuf {
        &self.buffers.planes[1 - self.current]
    }

    /// `other = conv(input)`, then a barrier; `other` becomes the current plane.
    pub(super) fn convolve(&mut self, conv: &Conv2d, input: &SharedBuf) {
        let (ctx, grid, outputs) = (self.ctx, self.grid, conv.outputs());
        let patches = self.buffers.patches.get(ctx);
        let output = self.other();
        conv.forward(
            ctx,
            (input.read(), grid),
            patches,
            |pixel, channel, value| {
                output.set(pixel * outputs + channel, value);
            },
        );
        ctx.barrier();
        self.current = 1 - self.current;
    }

    /// `other = resnet(current)`, normalizing `current` in place; `other` becomes current.
    pub(super) fn resnet(&mut self, resnet: &Resnet) {
        let input = self.plane();
        let planes = Planes {
            input,
            output: self.other(),
            normed: input,
            inner: &self.buffers.inner,
        };
        resnet.forward(self.ctx, &planes, self.grid, &self.shared);
        self.current = 1 - self.current;
    }

    /// `other = conv(current)` at twice the resolution, then a barrier; `other` becomes
    /// current.
    pub(super) fn upsample(&mut self, conv: &Conv2d) {
        let (ctx, grid, outputs) = (self.ctx, self.grid, conv.outputs());
        let input = self.plane().read();
        let patches = self.buffers.patches.get(ctx);
        let output = self.other();
        conv.forward_upsampled(ctx, (input, grid), patches, |pixel, channel, value| {
            output.set(pixel * outputs + channel, value);
        });
        ctx.barrier();
        self.current = 1 - self.current;
        self.grid = grid.doubled();
    }
}
