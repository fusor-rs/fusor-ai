//! One thread's view of a `Unet::forward`: the buffers its blocks share, and the steps the walk
//! is made of (a residual block's planes, a transformer with its context layers, a
//! convolution, the up path's concatenation).

use kernels::conv::{Conv2d, Grid};
use team::{Ctx, SharedBuf};

use super::transformer::{self, Text, Transformer};
use super::{Buffers, Context};
use crate::resnet::{Planes, Shared, Time};

/// Everything one thread's `forward` reads.
pub(super) struct Pass<'call> {
    pub(super) ctx: &'call Ctx<'call>,
    pub(super) buffers: &'call Buffers,
    pub(super) context: &'call Context,
    pub(super) shared: Shared<'call>,
    pub(super) planes: transformer::Planes<'call>,
}

impl<'call> Pass<'call> {
    pub(super) fn new(
        ctx: &'call Ctx<'call>,
        buffers: &'call Buffers,
        context: &'call Context,
    ) -> Self {
        Self {
            ctx,
            buffers,
            context,
            shared: Shared {
                statistics: &buffers.statistics,
                time: Some(Time {
                    embedding: &buffers.activated,
                    projected: &buffers.projected,
                }),
                patches: &buffers.patches,
            },
            planes: transformer::Planes {
                normed: &buffers.normed,
                tokens: &buffers.tokens,
                qkv: &buffers.qkv,
                attention: &buffers.attention,
                inner: &buffers.feed_forward,
                statistics: &buffers.statistics,
                patches: &buffers.patches,
                attention_scratch: &buffers.attention_scratch,
            },
        }
    }

    pub(super) fn resnet(
        &self,
        input: &'call SharedBuf,
        output: &'call SharedBuf,
        normed: &'call SharedBuf,
    ) -> Planes<'call> {
        Planes {
            input,
            output,
            normed,
            inner: &self.buffers.inner,
        }
    }

    /// `x` through `transformer` in place, with the next of the context's layers.
    pub(super) fn transform(
        &self,
        transformer: &Transformer,
        x: (&SharedBuf, Grid),
        layer: &mut usize,
    ) {
        let text = Text {
            layers: &self.context.layers[*layer..*layer + transformer.depth()],
            tokens: self.context.tokens,
        };
        transformer.forward(self.ctx, &self.planes, x, &text);
        *layer += transformer.depth();
    }

    /// `output = conv(input)`, then a barrier.
    pub(super) fn convolve(
        &self,
        conv: &Conv2d,
        (input, grid): (&SharedBuf, Grid),
        output: &SharedBuf,
    ) {
        let outputs = conv.outputs();
        let patches = self.buffers.patches.get(self.ctx);
        conv.forward(
            self.ctx,
            (input.read(), grid),
            patches,
            |pixel, channel, value| {
                output.set(pixel * outputs + channel, value);
            },
        );
        self.ctx.barrier();
    }

    /// `wide` = each pixel's `hidden` channels then its `skip` channels, then a barrier.
    pub(super) fn concatenate(
        &self,
        (hidden, skip): (&SharedBuf, &SharedBuf),
        (left, right): (usize, usize),
        grid: Grid,
    ) {
        let width = left + right;
        let (hidden, skip) = (hidden.read(), skip.read());
        for pixel in self.ctx.split(grid.pixels(), 1) {
            let row = self.buffers.wide.part(pixel * width..(pixel + 1) * width);
            row[..left].copy_from_slice(&hidden[pixel * left..(pixel + 1) * left]);
            row[left..].copy_from_slice(&skip[pixel * right..(pixel + 1) * right]);
        }
        self.ctx.barrier();
    }
}
