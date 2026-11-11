//! The residual block: `shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x))))))`, the norms
//! per pixel, the convolutions causal 3 × 3 × 3, the shortcut a pointwise linear where the
//! channels change (else `x` itself).
//!
//! Tensors under a prefix `p`: the vectors `p.norm1` and `p.norm2`, the convolutions `p.conv1`
//! and `p.conv2`, and where the channels change the linear `p.shortcut`.

use bundle::{Load, Scope, Store};
use kernels::conv::CausalConv3d;
use kernels::ops::silu;
use kernels::{Linear, Vector};

use super::Shape;
use super::walk::{Walk, into};

/// The channels a block reads and writes, over the autoencoder's convolutions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Dimensions {
    pub(super) shape: Shape,
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    /// Frames each convolution reads: the decoder's `kernel`, the encoder's first frame 1.
    pub(super) frames: usize,
}

pub(super) struct Resnet {
    norm1: Vector,
    conv1: CausalConv3d,
    norm2: Vector,
    conv2: CausalConv3d,
    shortcut: Option<Linear>,
}

impl Resnet {
    /// The block over the current plane; its output becomes the current one.
    pub(super) fn forward(&self, walk: &mut Walk) {
        let planes = &walk.buffers.planes;
        let outputs = self.conv2.outputs();
        let input = walk.current;
        let [normed, inner] = walk.free();
        walk.norm((input, normed), &self.norm1, silu);
        walk.causal(&self.conv1, normed, into(&planes[inner], outputs));
        walk.norm((inner, inner), &self.norm2, silu);
        let (input, output) = (&planes[input], &planes[normed]);
        match &self.shortcut {
            Some(shortcut) => {
                walk.project(shortcut, input, into(output, outputs));
                walk.causal(&self.conv2, inner, |pixel, channel, value| {
                    let at = pixel * outputs + channel;
                    output.set(at, output.get(at) + value);
                });
            }
            None => walk.causal(&self.conv2, inner, |pixel, channel, value| {
                let at = pixel * outputs + channel;
                output.set(at, value + input.get(at));
            }),
        }
        walk.current = normed;
    }
}

impl Load for Resnet {
    type Shape = Dimensions;

    fn load(
        store: &mut Store,
        prefix: &str,
        dimensions: Dimensions,
    ) -> Result<Self, bundle::Error> {
        let Dimensions {
            shape,
            inputs,
            outputs,
            frames,
        } = dimensions;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", inputs)?,
            conv1: scope.load("conv1", shape.causal((inputs, outputs), frames))?,
            norm2: scope.load("norm2", outputs)?,
            conv2: scope.load("conv2", shape.causal((outputs, outputs), frames))?,
            shortcut: (inputs != outputs)
                .then(|| scope.load("shortcut", [outputs, inputs]))
                .transpose()?,
        })
    }
}
