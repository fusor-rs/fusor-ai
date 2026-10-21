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

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{self, Conv2d, Grid, Patches, project_rows};
use kernels::ops::silu;
use team::{Ctx, PerThread, SharedBuf};

use crate::attention::{self, Call, Operand, attend};
use crate::norm::{GroupNorm, Groups};
use crate::resnet::{self, Planes, Resnet, Shared};

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

    /// The channels and the doublings of the latents' resolution of each step's output, in the
    /// order `decode` runs them: `conv_in`, the middle block's residual blocks, then each
    /// level's residual blocks and its resample. The steps alternate between the two planes.
    fn steps(&self) -> Vec<(usize, u32)> {
        let bottom = (self.bottom(), 0);
        let mut steps = Vec::from([bottom, bottom, bottom]);
        for (scale, level) in (0..self.levels()).rev().enumerate() {
            let scale = scale as u32;
            let channels = self.channels[level];
            steps.extend(std::iter::repeat_n((channels, scale), self.blocks + 1));
            if level > 0 {
                steps.push((channels, scale + 1));
            }
        }
        steps
    }

    /// Activations for decoding one image of up to `pixels` latent pixels.
    pub fn buffers(&self, pixels: usize, threads: usize) -> Buffers {
        let size = |(channels, scale): (usize, u32)| channels * (pixels << (2 * scale));
        let mut planes = [0; 2];
        for (step, output) in self.steps().into_iter().enumerate() {
            planes[step % 2] = planes[step % 2].max(size(output));
        }
        let resnets = (0..self.levels()).map(|level| {
            let scale = self.scale() - level as u32;
            size((self.channels[level], scale))
        });
        let new = SharedBuf::new;
        Buffers {
            pixels,
            latents: new(pixels * self.latent),
            quantized: new(pixels * self.latent),
            planes: planes.map(new),
            inner: new(resnets.fold(0, usize::max)),
            qkv: new(3 * pixels * self.bottom()),
            attention: new(pixels * self.bottom()),
            statistics: new(2 * self.groups),
            image: new(self.outputs * (pixels << (2 * self.scale()))),
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, attention::Scratch::default),
        }
    }
}

/// Activations of one `decode`.
pub struct Buffers {
    pixels: usize,
    /// The latents in, `pixels × latent`, pixel-major.
    pub latents: SharedBuf,
    /// The latents through `post_quant`.
    quantized: SharedBuf,
    planes: [SharedBuf; 2],
    inner: SharedBuf,
    qkv: SharedBuf,
    attention: SharedBuf,
    statistics: SharedBuf,
    /// The image out, pixel-major, `outputs` channels of every pixel.
    pub image: SharedBuf,
    patches: PerThread<Patches>,
    scratch: PerThread<attention::Scratch>,
}

struct Attention {
    norm: GroupNorm,
    qkv: Linear,
    out: Linear,
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

/// The planes of one `decode`: which holds the current activations.
struct Walk<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    shared: Shared<'call>,
    current: usize,
    grid: Grid,
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

    /// The middle block's attention, added to the current plane.
    fn attend(&self, walk: &Walk) {
        let Walk {
            ctx, buffers, grid, ..
        } = walk;
        let (x, channels) = (walk.plane(), self.shape.bottom());
        let normed = &buffers.inner;
        let identity = std::convert::identity;
        self.attention
            .norm
            .apply(ctx, (x, *grid), (normed, &buffers.statistics), identity);
        ctx.barrier();
        let width = 3 * channels;
        let rows = (normed.read(), grid.pixels());
        let patches = buffers.patches.get(ctx);
        project_rows(
            ctx,
            &self.attention.qkv,
            rows,
            patches,
            |pixel, col, value| {
                buffers.qkv.set(pixel * width + col, value);
            },
        );
        ctx.barrier();
        let part = |offset| Operand {
            values: buffers.qkv.read(),
            stride: width,
            offset,
        };
        let call = Call {
            heads: 1,
            head_dim: channels,
            images: grid.images,
            queries: grid.area(),
            keys: grid.area(),
            query: part(0),
            key: part(channels),
            value: part(2 * channels),
        };
        attend(ctx, &call, &buffers.attention, buffers.scratch.get(ctx));
        ctx.barrier();
        let rows = (buffers.attention.read(), grid.pixels());
        let patches = buffers.patches.get(ctx);
        project_rows(
            ctx,
            &self.attention.out,
            rows,
            patches,
            |pixel, col, value| {
                let at = pixel * channels + col;
                x.set(at, x.get(at) + value);
            },
        );
        ctx.barrier();
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

impl Walk<'_> {
    fn plane(&self) -> &SharedBuf {
        &self.buffers.planes[self.current]
    }

    /// The plane the next step writes.
    fn other(&self) -> &SharedBuf {
        &self.buffers.planes[1 - self.current]
    }

    /// `other = conv(input)`, then a barrier; `other` becomes the current plane.
    fn convolve(&mut self, conv: &Conv2d, input: &SharedBuf) {
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
    fn resnet(&mut self, resnet: &Resnet) {
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
    fn upsample(&mut self, conv: &Conv2d) {
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

impl Load for Attention {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let channels = shape.bottom();
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm: scope.load("norm", shape.norm(channels))?,
            qkv: scope.load("qkv", [3 * channels, channels])?,
            out: scope.load("out", [channels; 2])?,
        })
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
