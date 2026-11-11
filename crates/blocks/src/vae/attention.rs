//! The middle block's single-head self-attention over the pixels, added back.

use bundle::{Load, Scope, Store};

use super::walk::Walk;
use super::{Decoder, Shape};
use crate::norm::GroupNorm;
use crate::pixel_attention::{PixelAttention, Workspace};

pub(super) struct Attention {
    norm: GroupNorm,
    pixels: PixelAttention,
}

impl Decoder {
    /// The middle block's attention, added to the current plane.
    pub(super) fn attend(&self, walk: &Walk) {
        let Walk {
            ctx, buffers, grid, ..
        } = walk;
        let x = walk.plane();
        let normed = &buffers.inner;
        self.attention.norm.apply(
            ctx,
            (x, *grid),
            (normed, &buffers.statistics),
            std::convert::identity,
        );
        ctx.barrier();
        let workspace = Workspace {
            fused: &buffers.qkv,
            attended: &buffers.attention,
            patches: &buffers.patches,
            scratch: &buffers.scratch,
        };
        self.attention
            .pixels
            .forward(ctx, *grid, (normed, x), &workspace);
    }
}

impl Load for Attention {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let channels = shape.bottom();
        let pixels = PixelAttention::load(store, prefix, channels)?;
        let norm = Scope::new(store, prefix).load("norm", shape.norm(channels))?;
        Ok(Self { norm, pixels })
    }
}
