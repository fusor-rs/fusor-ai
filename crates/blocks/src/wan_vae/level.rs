//! One level of the decoder: its residual blocks, then the resample to the next level.

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::conv::{self, CausalConv3d, CausalShape, Conv2d};

use super::Shape;
use super::resnet::{Dimensions, Resnet};
use super::walk::Walk;

pub(super) struct Level {
    resnets: Vec<Resnet>,
    /// On temporal levels, the convolution doubling the frames.
    time: Option<CausalConv3d>,
    /// But on the last level, the convolution after the nearest ×2 upsampling.
    resample: Option<Conv2d>,
}

impl Level {
    /// The blocks over the current plane, then the resample of rows `kept` of their output.
    pub(super) fn forward(&self, walk: &mut Walk, kept: Range<usize>) {
        for resnet in &self.resnets {
            resnet.forward(walk);
        }
        if let Some(time) = &self.time {
            walk.double_frames(time);
        }
        if let Some(resample) = &self.resample {
            walk.double_grid(resample, kept);
        }
    }

    pub(super) fn load(
        store: &mut Store,
        prefix: &str,
        shape: &Shape,
        level: usize,
    ) -> Result<Self, bundle::Error> {
        let channels = shape.channels[level];
        let resnets = (0..shape.blocks)
            .map(|block| {
                let inputs = if block == 0 {
                    shape.level_inputs(level)
                } else {
                    channels
                };
                let name = format!("{prefix}.resnets.{block}");
                Resnet::load(
                    store,
                    &name,
                    Dimensions {
                        shape: *shape,
                        inputs,
                        outputs: channels,
                        frames: shape.kernel,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut scope = Scope::new(store, prefix);
        let doubling = CausalShape {
            inputs: channels,
            outputs: 2 * channels,
            window: shape.kernel,
            kernel: 1,
        };
        let halving = conv::Shape::same(channels, channels / 2, shape.kernel);
        Ok(Self {
            resnets,
            time: shape
                .has_time(level)
                .then(|| scope.load("time", doubling))
                .transpose()?,
            resample: (level + 1 < shape.levels())
                .then(|| scope.load("resample", halving))
                .transpose()?,
        })
    }
}
