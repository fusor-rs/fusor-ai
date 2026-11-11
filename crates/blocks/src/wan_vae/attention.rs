//! The middle block's single-head self-attention over each frame's pixels, added back.

use bundle::{Load, Scope, Store};
use kernels::Vector;
use kernels::conv::Grid;

use super::walk::Walk;
use crate::pixel_attention::{PixelAttention, Workspace};

pub(super) struct Attention {
    norm: Vector,
    pixels: PixelAttention,
}

impl Attention {
    /// `x += out(attention(norm(x)))` over the current plane.
    pub(super) fn forward(&self, walk: &Walk) {
        let Walk {
            ctx,
            buffers,
            current,
            grid,
            frames,
            ..
        } = walk;
        let [normed, fused] = walk.free();
        walk.norm((*current, normed), &self.norm, std::convert::identity);
        let workspace = Workspace {
            fused: &buffers.planes[fused],
            attended: &buffers.planes[normed],
            patches: &buffers.patches,
            scratch: &buffers.scratch,
        };
        let frames_grid = Grid {
            images: *frames,
            ..*grid
        };
        self.pixels.forward(
            ctx,
            frames_grid,
            (&buffers.planes[normed], &buffers.planes[*current]),
            &workspace,
        );
    }
}

impl Load for Attention {
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, channels: usize) -> Result<Self, bundle::Error> {
        let pixels = PixelAttention::load(store, prefix, channels)?;
        let norm = Scope::new(store, prefix).load("norm", channels)?;
        Ok(Self { norm, pixels })
    }
}
