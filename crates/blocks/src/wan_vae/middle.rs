//! The middle block between a decoder's convolution in and its levels, and an encoder's levels
//! and head: a residual block, the attention, a residual block, at the first level's channels.
//!
//! Tensors under a prefix `p`: `p.middle.{first, attention, second}`.

use bundle::{Load, Scope, Store};

use super::Shape;
use super::attention::Attention;
use super::resnet::{Dimensions, Resnet};
use super::walk::Walk;

pub(super) struct Middle {
    first: Resnet,
    attention: Attention,
    second: Resnet,
}

impl Middle {
    pub(super) fn forward(&self, walk: &mut Walk) {
        self.first.forward(walk);
        self.attention.forward(walk);
        self.second.forward(walk);
    }
}

impl Load for Middle {
    /// The shape and the frames each convolution reads.
    type Shape = (Shape, usize);

    fn load(
        store: &mut Store,
        prefix: &str,
        (shape, frames): (Shape, usize),
    ) -> Result<Self, bundle::Error> {
        let channels = shape.channels[0];
        let dimensions = Dimensions {
            shape,
            inputs: channels,
            outputs: channels,
            frames,
        };
        let middle_prefix = format!("{prefix}.middle");
        let mut scope = Scope::new(store, &middle_prefix);
        Ok(Self {
            first: scope.load("first", dimensions)?,
            attention: scope.load("attention", channels)?,
            second: scope.load("second", dimensions)?,
        })
    }
}
