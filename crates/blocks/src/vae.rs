//! The decoder of latent diffusion's KL autoencoder (`AutoencoderKL` of Stable Diffusion 1.x,
//! 2.x and XL): latents to an image. A pointwise linear (`post_quant_conv`, into which a compiler
//! folds the latents' scaling factor), `conv_in`, a middle block (a residual block, a single-head
//! self-attention over the pixels added back, a residual block), then the levels from the last
//! to the first, `blocks + 1` residual blocks each, every level but the first doubling the
//! resolution after them (nearest, then a convolution); last a GroupNorm, SiLU and `conv_out`.
//! The residual blocks have no time embedding.
//!
//! `decode` runs inside a team run. Activations move between two pixel-major planes, each step
//! (a convolution or a residual block) reading one and writing the other; a residual block
//! normalizes its input in place, as nothing reads it after, so no step needs a third buffer of
//! its input's size. `Shape::steps` lists the steps' outputs in that order, which sizes the
//! planes.
//!
//! Tensors under a prefix `p`: `p.post_quant` (a linear), `p.conv_in`, `p.middle.{first,
//! attention, second}`, `p.up.{l}.resnets.{i}` and, but for the first level, `p.up.{l}.resample`,
//! `p.norm_out` and `p.conv_out`; the attention's `norm` (a GroupNorm), `qkv` (the fused
//! projection) and `out` (linears); `resnet` lists a residual block's tensors.

mod attention;
mod buffers;
mod walk;

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{self, Conv2d, Grid, project_rows};
use kernels::ops::silu;
use team::Ctx;

use crate::norm::{GroupNorm, Groups};
use crate::resnet::{self, Resnet, Shared};
use attention::Attention;
pub use buffers::Buffers;
use walk::Walk;

/// The dimensions of one decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Channels of a latent pixel.
    pub latent: usize,
    /// Channels at each level, the first at the image's resolution.
    pub channels: &'static [usize],
    /// Residual blocks per level, less one.
    pub blocks: usize,
    pub kernel: usize,
    pub groups: usize,
    pub eps: f32,
    /// Channels of an image pixel.
    pub outputs: usize,
}

impl Shape {
    fn levels(&self) -> usize {
        self.channels.len()
    }

    fn bottom(&self) -> usize {
        self.channels[self.levels() - 1]
    }

    fn norm(&self, channels: usize) -> Groups {
        Groups {
            channels,
            groups: self.groups,
            eps: self.eps,
        }
    }

    fn resnet(&self, inputs: usize, outputs: usize) -> resnet::Shape {
        resnet::Shape {
            inputs,
            outputs,
            kernel: self.kernel,
            groups: self.groups,
            eps: self.eps,
            time: None,
        }
    }

    /// Doublings of the latents' resolution between the image and the latents.
    pub fn scale(&self) -> u32 {
        self.levels() as u32 - 1
    }
}

struct Level {
    resnets: Vec<Resnet>,
    resample: Option<Conv2d>,
}

pub struct Decoder {
    shape: Shape,
    post_quant: Linear,
    conv_in: Conv2d,
    first: Resnet,
    attention: Attention,
    second: Resnet,
    /// By level, the first at the image's resolution: `decode` walks them last first.
    up: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    /// The image's grid for latents of `grid`.
    pub fn output_grid(&self, grid: Grid) -> Grid {
        (0..self.shape.scale()).fold(grid, |grid, _| grid.doubled())
    }

    /// `buffers.image` = the image of the latents in `buffers.latents`, one image of `grid`.
    ///
    /// # Panics
    /// When the buffers do not hold the latents.
    pub fn decode(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        assert!(
            grid.images == 1 && grid.area() <= buffers.pixels,
            "the buffers hold one image's latents"
        );
        let mut walk = Walk {
            ctx,
            buffers,
            shared: Shared {
                statistics: &buffers.statistics,
                time: None,
                patches: &buffers.patches,
            },
            current: 1,
            grid,
        };
        let latent = self.shape.latent;
        let latents = (buffers.latents.read(), grid.pixels());
        let patches = buffers.patches.get(ctx);
        project_rows(
            ctx,
            &self.post_quant,
            latents,
            patches,
            |pixel, channel, value| {
                buffers.quantized.set(pixel * latent + channel, value);
            },
        );
        ctx.barrier();
        walk.convolve(&self.conv_in, &buffers.quantized);
        walk.resnet(&self.first);
        self.attend(&walk);
        walk.resnet(&self.second);
        for level in self.up.iter().rev() {
            for resnet in &level.resnets {
                walk.resnet(resnet);
            }
            if let Some(upsample) = &level.resample {
                walk.upsample(upsample);
            }
        }
        self.out(&walk);
    }

    /// `image = conv_out(silu(norm_out(current)))`.
    fn out(&self, walk: &Walk) {
        let Walk {
            ctx, buffers, grid, ..
        } = walk;
        let x = walk.plane();
        self.norm_out
            .apply(ctx, (x, *grid), (x, &buffers.statistics), silu);
        ctx.barrier();
        let outputs = self.shape.outputs;
        let patches = buffers.patches.get(ctx);
        self.conv_out
            .forward(ctx, (x.read(), *grid), patches, |pixel, channel, value| {
                buffers.image.set(pixel * outputs + channel, value);
            });
        ctx.barrier();
    }
}

impl Load for Decoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let levels = shape.levels();
        let up = (0..levels)
            .map(|level| {
                let channels = shape.channels[level];
                let above = shape.channels[(level + 1).min(levels - 1)];
                let resnets = (0..=shape.blocks).map(|block| {
                    let inputs = if block == 0 { above } else { channels };
                    let name = format!("{prefix}.up.{level}.resnets.{block}");
                    Resnet::load(store, &name, shape.resnet(inputs, channels))
                });
                let resnets = resnets.collect::<Result<Vec<_>, _>>()?;
                let resample = conv::Shape::same(channels, channels, shape.kernel);
                let name = format!("{prefix}.up.{level}.resample");
                let resample = (level > 0)
                    .then(|| Conv2d::load(store, &name, resample))
                    .transpose()?;
                Ok(Level { resnets, resample })
            })
            .collect::<Result<Vec<_>, bundle::Error>>()?;
        let bottom = shape.bottom();
        let middle = |store: &mut Store, part: &str| {
            let name = format!("{prefix}.middle.{part}");
            Resnet::load(store, &name, shape.resnet(bottom, bottom))
        };
        let first = middle(store, "first")?;
        let second = middle(store, "second")?;
        let attention = Attention::load(store, &format!("{prefix}.middle.attention"), shape)?;
        let image = shape.channels[0];
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            post_quant: scope.load("post_quant", [shape.latent; 2])?,
            conv_in: scope.load(
                "conv_in",
                conv::Shape::same(shape.latent, bottom, shape.kernel),
            )?,
            first,
            attention,
            second,
            up,
            norm_out: scope.load("norm_out", shape.norm(image))?,
            conv_out: scope.load(
                "conv_out",
                conv::Shape::same(image, shape.outputs, shape.kernel),
            )?,
        })
    }
}
