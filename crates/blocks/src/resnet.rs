//! The residual block of latent-diffusion UNets and their autoencoders (LDM's `ResnetBlock`):
//! `shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x))) + time)))`. The norms are GroupNorms,
//! the convolutions `kernel`-wide at stride 1, the shortcut a pointwise linear where the
//! channels change (else `x` itself), and `time` a projection of the activated time embedding,
//! added per channel, in blocks that have one.
//!
//! `forward` runs inside a team run: the shortcut, the time projection and the first norm, the
//! first convolution, the second norm in place, then the second convolution adds into the
//! output, with barriers between dependent phases.
//!
//! Tensors under a prefix `p`: `p.norm1`, `p.conv1`, `p.norm2` and `p.conv2`, and where the
//! block has them `p.time` and `p.shortcut` (linears).

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{self, Conv2d, Grid, Patches, project_rows};
use kernels::ops::silu;
use team::{Ctx, PerThread, SharedBuf};

use crate::norm::{GroupNorm, Groups};

/// The dimensions of one block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shape {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) kernel: usize,
    pub(crate) groups: usize,
    pub(crate) eps: f32,
    /// The width of the time embedding, in blocks that add it.
    pub(crate) time: Option<usize>,
}

pub(crate) struct Resnet {
    norm1: GroupNorm,
    conv1: Conv2d,
    time: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Linear>,
}

/// The pixel-major buffers of one `forward`. `normed` may be `input`, which it then overwrites,
/// unless `output` is `input` too; `output` is never `normed`.
pub(crate) struct Planes<'buf> {
    pub(crate) input: &'buf SharedBuf,
    pub(crate) output: &'buf SharedBuf,
    pub(crate) normed: &'buf SharedBuf,
    /// `conv1`'s output, then its norm.
    pub(crate) inner: &'buf SharedBuf,
}

/// What the blocks of one call share.
pub(crate) struct Shared<'buf> {
    /// The GroupNorms' statistics.
    pub(crate) statistics: &'buf SharedBuf,
    /// The time embedding, in calls whose blocks add it.
    pub(crate) time: Option<Time<'buf>>,
    pub(crate) patches: &'buf PerThread<Patches>,
}

/// `silu` of the time embedding (one row), and a block's projection of it (one value per
/// output channel).
#[derive(Clone, Copy)]
pub(crate) struct Time<'buf> {
    pub(crate) embedding: &'buf SharedBuf,
    pub(crate) projected: &'buf SharedBuf,
}

impl Resnet {
    pub(crate) fn inputs(&self) -> usize {
        self.conv1.inputs()
    }

    pub(crate) fn outputs(&self) -> usize {
        self.conv2.outputs()
    }

    /// `planes.output` = the block over `planes.input`, pixels of `grid`.
    pub(crate) fn forward(&self, ctx: &Ctx, planes: &Planes, grid: Grid, shared: &Shared) {
        let patches = shared.patches.get(ctx);
        let outputs = self.outputs();
        self.residual(ctx, planes, grid, patches);
        let time = self.time.as_ref().zip(shared.time);
        if let Some((linear, time)) = time {
            let embedding = (time.embedding.read(), 1);
            project_rows(ctx, linear, embedding, patches, |_, channel, value| {
                time.projected.set(channel, value);
            });
        }
        let normed = (planes.normed, shared.statistics);
        self.norm1.apply(ctx, (planes.input, grid), normed, silu);
        ctx.barrier();
        let projected = time.map(|(_, time)| time.projected.read());
        let input = (planes.normed.read(), grid);
        self.conv1
            .forward(ctx, input, patches, |pixel, channel, value| {
                let shift = projected.map_or(0.0, |projected| projected[channel]);
                planes.inner.set(pixel * outputs + channel, value + shift);
            });
        ctx.barrier();
        let inner = (planes.inner, shared.statistics);
        self.norm2.apply(ctx, (planes.inner, grid), inner, silu);
        ctx.barrier();
        let input = (planes.inner.read(), grid);
        self.conv2
            .forward(ctx, input, patches, |pixel, channel, value| {
                let at = pixel * outputs + channel;
                planes.output.set(at, planes.output.get(at) + value);
            });
        ctx.barrier();
    }

    /// `output = shortcut(input)`, or `input` itself; the barriers before the second
    /// convolution order these writes before it adds to them.
    fn residual(&self, ctx: &Ctx, planes: &Planes, grid: Grid, patches: &mut Patches) {
        let outputs = self.outputs();
        let input = planes.input.read();
        match &self.shortcut {
            Some(shortcut) => {
                let rows = (input, grid.pixels());
                project_rows(ctx, shortcut, rows, patches, |pixel, channel, value| {
                    planes.output.set(pixel * outputs + channel, value);
                });
            }
            None if !std::ptr::eq(planes.input, planes.output) => {
                let pixels = ctx.split(grid.pixels(), 1);
                let range = pixels.start * outputs..pixels.end * outputs;
                planes
                    .output
                    .part(range.clone())
                    .copy_from_slice(&input[range]);
            }
            None => {}
        }
    }
}

impl Load for Resnet {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            inputs,
            outputs,
            kernel,
            ..
        } = shape;
        let norm = |channels| Groups {
            channels,
            groups: shape.groups,
            eps: shape.eps,
        };
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", norm(inputs))?,
            conv1: scope.load("conv1", conv::Shape::same(inputs, outputs, kernel))?,
            time: (shape.time)
                .map(|width| scope.load("time", [outputs, width]))
                .transpose()?,
            norm2: scope.load("norm2", norm(outputs))?,
            conv2: scope.load("conv2", conv::Shape::same(outputs, outputs, kernel))?,
            shortcut: (inputs != outputs)
                .then(|| scope.load("shortcut", [outputs, inputs]))
                .transpose()?,
        })
    }
}
