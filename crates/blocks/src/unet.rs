//! The UNet of latent diffusion (`UNet2DConditionModel` of Stable Diffusion 1.x and 2.x): the
//! noise in latents at a diffusion timestep, conditioned on a context (the text). A sinusoidal
//! embedding of the timestep through a two-layer MLP conditions every residual block. Levels of
//! `blocks` residual blocks, each followed by a spatial transformer at the levels that have them,
//! halve the resolution on the way down (a stride-2 convolution) and double it on the way up
//! (nearest, then a convolution); every down block's output is concatenated (channels after the
//! up path's) to the input of an up block, the last first. A middle block (a residual block, a
//! transformer, a residual block) sits at the bottom.
//!
//! Latents are pixel-major (`kernels::conv::Grid`), several per call (classifier-free guidance's
//! two branches), each image attending to its own context rows. Their height and width must be
//! multiples of `2^(levels − 1)`.
//!
//! `forward` runs inside a team run, one phase per norm, convolution, projection and attention
//! with barriers between. `encode` computes every cross-attention's keys and values of a context
//! once, before the `forward`s over it.
//!
//! Tensors under a prefix `p`: `p.time_in` and `p.time_out` (linears), `p.conv_in`,
//! `p.down.{l}.{resnets.{i}, transformers.{i}, resample}` (the last level has no resample),
//! `p.middle.{first, transformer, second}`, `p.up.{l}.{resnets.{i}, transformers.{i}, resample}`
//! (the first level has no resample), `p.norm_out` and `p.conv_out`; `resnet` lists a residual
//! block's tensors, `unet::transformer` a transformer's.

mod buffers;
mod level;
mod pass;
mod transformer;
mod walk;

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{self, Conv2d, Grid};
use team::{Ctx, SharedBuf};

use crate::norm::{GroupNorm, Groups};
use crate::resnet::{self, Resnet};
pub use buffers::Buffers;
use level::{Level, Middle, Stage};
use pass::Pass;
use transformer::Transformer;

/// The dimensions of one UNet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Channels of a latent pixel.
    pub latent: usize,
    /// Channels at each level, the first at the latents' resolution.
    pub channels: &'static [usize],
    /// The transformer layers after each residual block of a level, 0 where it has none.
    pub depths: &'static [usize],
    /// The transformer layers of the middle block.
    pub middle_depth: usize,
    /// Residual blocks per level on the way down; the way up has one more.
    pub blocks: usize,
    pub kernel: usize,
    pub heads: usize,
    /// The feed-forwards' inner width, in multiples of their channels.
    pub feed_forward: usize,
    /// Width of a context row.
    pub context: usize,
    /// The time embedding's width, in multiples of the first level's channels.
    pub time: usize,
    pub groups: usize,
    /// Epsilon of the residual blocks' norms, the LayerNorms and the output norm.
    pub norm_eps: f32,
    /// Epsilon of the transformers' GroupNorms.
    pub spatial_eps: f32,
}

impl Shape {
    fn levels(&self) -> usize {
        self.channels.len()
    }

    /// Channels at `level`, clamped to the levels there are.
    fn level_channels(&self, level: isize) -> usize {
        let last = self.levels() - 1;
        self.channels[level.clamp(0, last as isize) as usize]
    }

    fn time_width(&self) -> usize {
        self.time * self.channels[0]
    }

    fn resnet(&self, inputs: usize, outputs: usize) -> resnet::Shape {
        resnet::Shape {
            inputs,
            outputs,
            kernel: self.kernel,
            groups: self.groups,
            eps: self.norm_eps,
            time: Some(self.time_width()),
        }
    }

    fn transformer(&self, channels: usize, depth: usize) -> transformer::Shape {
        transformer::Shape {
            channels,
            depth,
            heads: self.heads,
            inner: self.feed_forward * channels,
            context: self.context,
            groups: self.groups,
            group_eps: self.spatial_eps,
            layer_eps: self.norm_eps,
        }
    }
}

pub struct Unet {
    shape: Shape,
    time_in: Linear,
    time_out: Linear,
    conv_in: Conv2d,
    down: Vec<Level>,
    middle: Middle,
    /// By level, the first at the latents' resolution: the up path walks them last first.
    up: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

/// Every cross-attention's keys and values of one context (`Unet::encode`): per transformer
/// layer, in the order `forward` runs them, `tokens` rows per image.
pub struct Context {
    tokens: usize,
    layers: Vec<SharedBuf>,
}

impl Unet {
    /// The buffers of a context of `tokens` rows per image for `images` images.
    pub fn context(&self, images: usize, tokens: usize) -> Context {
        let layers = self.transformers().flat_map(|transformer| {
            let channels = transformer.channels();
            (0..transformer.depth()).map(move |_| SharedBuf::new(images * tokens * 2 * channels))
        });
        Context {
            tokens,
            layers: layers.collect(),
        }
    }

    /// Every transformer layer's keys and values of `text`, `context.tokens` rows of the
    /// context's width per image, for this thread's share.
    pub fn encode(&self, ctx: &Ctx, buffers: &Buffers, text: &[f32], context: &Context) {
        let rows = text.len() / self.shape.context;
        let mut at = 0;
        for transformer in self.transformers() {
            let layers = &context.layers[at..at + transformer.depth()];
            transformer.encode(ctx, (text, rows), layers, buffers.patches.get(ctx));
            at += transformer.depth();
        }
        ctx.barrier();
    }

    /// The transformers in the order `forward` runs them.
    fn transformers(&self) -> impl Iterator<Item = &Transformer> {
        let down = self.down.iter().flat_map(|level| &level.transformers);
        let up = self.up.iter().rev().flat_map(|level| &level.transformers);
        down.chain([&self.middle.transformer]).chain(up)
    }

    /// `buffers.prediction` = the noise in `buffers.sample` (latents of `grid`) at `timestep`,
    /// each image conditioned on its rows of `context`.
    ///
    /// # Panics
    /// When the buffers do not hold `grid`, or its sides are not multiples of
    /// `2^(levels − 1)`.
    pub fn forward(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        context: &Context,
        (grid, timestep): (Grid, f32),
    ) {
        let multiple = 1 << (self.shape.levels() - 1);
        assert!(
            grid.images <= buffers.images && grid.area() <= buffers.pixels,
            "the buffers hold the latents"
        );
        assert!(
            grid.height.is_multiple_of(multiple) && grid.width.is_multiple_of(multiple),
            "the latents' sides are multiples of {multiple}"
        );
        let pass = Pass::new(ctx, buffers, context);
        self.embed_time(&pass, timestep);
        let mut layer = 0;
        let bottom = self.down_path(&pass, grid, &mut layer);
        self.middle(&pass, bottom, &mut layer);
        let (top, hidden) = self.up_path(&pass, bottom, &mut layer);
        self.out(&pass, top, hidden);
    }
}

impl Load for Unet {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let levels = shape.levels();
        if levels == 0 || shape.depths.len() != levels {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {levels} levels with {} transformer depths",
                shape.depths.len()
            )));
        }
        let path = |store: &mut Store, path: &str, stage: &dyn Fn(usize) -> Stage| {
            (0..levels)
                .map(|level| Level::load(store, &format!("{prefix}.{path}.{level}"), stage(level)))
                .collect::<Result<Vec<_>, _>>()
        };
        let down = path(store, "down", &|level| shape.down(level))?;
        let up = path(store, "up", &|level| shape.up(level))?;
        let bottom = shape.channels[levels - 1];
        let resnet = shape.resnet(bottom, bottom);
        let middle = Middle {
            first: Resnet::load(store, &format!("{prefix}.middle.first"), resnet)?,
            transformer: Transformer::load(
                store,
                &format!("{prefix}.middle.transformer"),
                shape.transformer(bottom, shape.middle_depth),
            )?,
            second: Resnet::load(store, &format!("{prefix}.middle.second"), resnet)?,
        };
        let (first, width) = (shape.channels[0], shape.time_width());
        let output = Groups {
            channels: first,
            groups: shape.groups,
            eps: shape.norm_eps,
        };
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            time_in: scope.load("time_in", [width, first])?,
            time_out: scope.load("time_out", [width, width])?,
            conv_in: scope.load(
                "conv_in",
                conv::Shape::same(shape.latent, first, shape.kernel),
            )?,
            down,
            middle,
            up,
            norm_out: scope.load("norm_out", output)?,
            conv_out: scope.load(
                "conv_out",
                conv::Shape::same(first, shape.latent, shape.kernel),
            )?,
        })
    }
}
