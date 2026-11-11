//! The encoder of a video's first frame: the decoder's mirror, with every convolution over the
//! frame alone.

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{self, CausalConv3d, Conv2d, Grid};
use team::Ctx;

use super::head::Head;
use super::middle::Middle;
use super::resnet::{Dimensions, Resnet};
use super::walk::{Walk, into};
use super::{Buffers, Shape};

/// One level of the encoder: its residual blocks, then the resample to the next level.
struct EncoderLevel {
    resnets: Vec<Resnet>,
    /// But on the last level, the strided convolution halving the height and width.
    resample: Option<Conv2d>,
}

impl EncoderLevel {
    fn forward(&self, walk: &mut Walk) {
        for resnet in &self.resnets {
            resnet.forward(walk);
        }
        if let Some(resample) = &self.resample {
            walk.halve_grid(resample);
        }
    }

    /// The `level`th level from the full resolution, at the channels of the decoder's level
    /// `levels − 1 − level`.
    fn load(
        store: &mut Store,
        prefix: &str,
        shape: &Shape,
        level: usize,
    ) -> Result<Self, bundle::Error> {
        let last = shape.levels() - 1;
        let channels = shape.channels[last - level];
        let resnets = (0..shape.blocks - 1)
            .map(|block| {
                let inputs = match (block, level) {
                    (0, 0) => shape.channels[last],
                    (0, _) => shape.channels[last - level + 1],
                    _ => channels,
                };
                let dimensions = Dimensions {
                    shape: *shape,
                    inputs,
                    outputs: channels,
                    frames: 1,
                };
                Resnet::load(store, &format!("{prefix}.resnets.{block}"), dimensions)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let halving = conv::Shape::halving(channels, channels, shape.encoder_resample());
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            resnets,
            resample: (level < last)
                .then(|| scope.load("resample", halving))
                .transpose()?,
        })
    }
}

pub struct Encoder {
    shape: Shape,
    conv_in: CausalConv3d,
    down: Vec<EncoderLevel>,
    middle: Middle,
    head: Head,
    quant: Linear,
}

impl Encoder {
    /// `buffers.latent` = the moments of a video's first frame (`buffers.frames`, one frame of
    /// `grid`): per latent pixel `latent` means, then `latent` log-variances.
    ///
    /// # Panics
    /// When the buffers do not hold the first frame.
    pub fn encode(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        let scale = self.shape.scale();
        assert!(
            grid.images == 1 && grid.area() <= buffers.capacity.pixels * scale * scale,
            "the buffers hold one first frame"
        );
        let Shape {
            latent,
            channels,
            outputs,
            ..
        } = self.shape;
        let frame = &buffers.frames;
        for pixel in ctx.split(grid.area(), 1) {
            let range = pixel * outputs..(pixel + 1) * outputs;
            let target = buffers.planes[1].part(range.clone());
            target.copy_from_slice(&frame.read()[range]);
        }
        ctx.barrier();
        let mut walk = Walk::new(ctx, buffers, (grid, 0), &[]);
        let last = channels.len() - 1;
        walk.causal(&self.conv_in, 1, into(&buffers.planes[0], channels[last]));
        for level in &self.down {
            level.forward(&mut walk);
        }
        self.middle.forward(&mut walk);
        let [_, out] = walk.free();
        self.head
            .forward(&mut walk, into(&buffers.planes[out], 2 * latent));
        let moments = into(&buffers.latent, 2 * latent);
        walk.project(&self.quant, &buffers.planes[out], moments);
    }
}

impl Load for Encoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            latent,
            channels,
            outputs,
            ..
        } = shape;
        let prefix = format!("{prefix}.encoder");
        let last = channels.len() - 1;
        let down = (0..shape.levels())
            .map(|level| {
                EncoderLevel::load(store, &format!("{prefix}.down.{level}"), &shape, level)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let middle = Middle::load(store, &prefix, (shape, 1))?;
        let head_shape = (channels[0], shape.causal((channels[0], 2 * latent), 1));
        let head = Head::load(store, &prefix, head_shape)?;
        let mut scope = Scope::new(store, &prefix);
        Ok(Self {
            shape,
            conv_in: scope.load("conv_in", shape.causal((outputs, channels[last]), 1))?,
            down,
            middle,
            head,
            quant: scope.load("quant", [2 * latent; 2])?,
        })
    }
}
