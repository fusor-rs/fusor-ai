//! The activations of the convolutional networks and the layers that compute them inside one
//! team run: pixel-major rows of a `Grid`, `channels` values each, in a shared buffer.
//!
//! Each layer is one phase: its threads write disjoint output pixels (or, for few pixels,
//! disjoint channels of every pixel) of its own buffer and read only buffers of earlier phases;
//! a barrier ends it.

use kernels::conv::{Conv2d, Depthwise2d, Grid, Patches};
use team::{Ctx, PerThread, SharedBuf};

/// A view of activations: `channels` values per pixel of `grid`.
#[derive(Clone, Copy)]
pub(crate) struct Plane<'buf> {
    pub(crate) values: &'buf SharedBuf,
    pub(crate) grid: Grid,
    pub(crate) channels: usize,
}

impl Plane<'_> {
    pub(crate) fn at(&self, pixel: usize, channel: usize) -> f32 {
        self.values.get(pixel * self.channels + channel)
    }
}

/// One thread's context and scratch for running layers.
#[derive(Clone, Copy)]
pub(crate) struct Layers<'run> {
    pub(crate) ctx: &'run Ctx<'run>,
    pub(crate) patches: &'run PerThread<Patches>,
    pub(crate) scratch: &'run PerThread<Vec<f32>>,
}

/// Per-thread scratch for the networks' layers.
pub struct Scratch {
    pub(crate) patches: PerThread<Patches>,
    pub(crate) values: PerThread<Vec<f32>>,
}

impl Scratch {
    pub(crate) fn new(threads: usize) -> Self {
        Self {
            patches: PerThread::new(threads, Patches::default),
            values: PerThread::new(threads, Vec::new),
        }
    }

    pub(crate) fn layers<'run>(&'run self, ctx: &'run Ctx<'run>) -> Layers<'run> {
        Layers {
            ctx,
            patches: &self.patches,
            scratch: &self.values,
        }
    }
}

impl Layers<'_> {
    /// `into` = `finish(pixel, channel, value)` of the convolution of `input`.
    pub(crate) fn conv<'out>(
        &self,
        conv: &Conv2d,
        input: Plane,
        into: &'out SharedBuf,
        finish: impl Fn(usize, usize, f32) -> f32,
    ) -> Plane<'out> {
        let output = Plane {
            values: into,
            grid: conv.output_grid(input.grid),
            channels: conv.outputs(),
        };
        let source = (input.values.read(), input.grid);
        let patches = self.patches.get(self.ctx);
        conv.forward(self.ctx, source, patches, |pixel, channel, value| {
            into.set(
                pixel * output.channels + channel,
                finish(pixel, channel, value),
            );
        });
        self.ctx.barrier();
        output
    }

    /// `into` = `finish(pixel, channel, value)` of the depthwise convolution of `input`.
    pub(crate) fn depthwise<'out>(
        &self,
        depthwise: &Depthwise2d,
        input: Plane,
        into: &'out SharedBuf,
        finish: impl Fn(usize, usize, f32) -> f32,
    ) -> Plane<'out> {
        let channels = depthwise.channels();
        let output = Plane {
            values: into,
            grid: depthwise.output_grid(input.grid),
            channels,
        };
        let source = (input.values.read(), input.grid);
        let scratch = self.scratch.get(self.ctx);
        depthwise.forward(self.ctx, source, scratch, |pixel, values| {
            let row = into.part(pixel * channels..(pixel + 1) * channels);
            for (channel, (out, &value)) in row.iter_mut().zip(values).enumerate() {
                *out = finish(pixel, channel, value);
            }
        });
        self.ctx.barrier();
        output
    }
}
