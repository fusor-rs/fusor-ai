//! The middle block's single-head self-attention over the pixels, added back.

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::project_rows;

use super::walk::Walk;
use super::{Decoder, Shape};
use crate::attention::{Call, Operand, attend};
use crate::norm::GroupNorm;

pub(super) struct Attention {
    norm: GroupNorm,
    qkv: Linear,
    out: Linear,
}

impl Decoder {
    /// The middle block's attention, added to the current plane.
    pub(super) fn attend(&self, walk: &Walk) {
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
