//! A single-head self-attention over each image's pixels, added back to them: the middle block
//! of an autoencoder's decoder, after its own norm.
//!
//! Tensors under a prefix `p`: `p.qkv` (the fused query, key and value projection) and `p.out`
//! (linears).

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{Grid, Patches, project_rows};
use kernels::ops::Exponentials;
use team::{Ctx, PerThread, SharedBuf};

use crate::attention::{Call, Operand, Scratch, attend};

pub(crate) struct PixelAttention {
    qkv: Linear,
    out: Linear,
}

/// The buffers one attention works in.
pub(crate) struct Workspace<'buf> {
    /// The fused projection of every pixel.
    pub(crate) fused: &'buf SharedBuf,
    /// The attention's output, pixel-major; it may be the input.
    pub(crate) attended: &'buf SharedBuf,
    pub(crate) patches: &'buf PerThread<Patches>,
    pub(crate) scratch: &'buf PerThread<Scratch>,
}

impl PixelAttention {
    /// `residual += out(attention(normed))` over the pixels of `grid`, then a barrier.
    pub(crate) fn forward(
        &self,
        ctx: &Ctx,
        grid: Grid,
        (normed, residual): (&SharedBuf, &SharedBuf),
        workspace: &Workspace,
    ) {
        let channels = self.out.matrix().rows;
        let width = 3 * channels;
        project_rows(
            ctx,
            &self.qkv,
            (normed.read(), grid.pixels()),
            workspace.patches.get(ctx),
            |pixel, col, value| {
                workspace.fused.set(pixel * width + col, value);
            },
        );
        ctx.barrier();
        let part = |offset| Operand {
            values: workspace.fused.read(),
            stride: width,
            offset,
        };
        let call = Call {
            heads: 1,
            kv_heads: 1,
            head_dim: channels,
            images: grid.images,
            queries: grid.area(),
            keys: grid.area(),
            query: part(0),
            key: part(channels),
            value: part(2 * channels),
            exponentials: Exponentials::Exact,
        };
        attend(ctx, &call, workspace.attended, workspace.scratch.get(ctx));
        ctx.barrier();
        project_rows(
            ctx,
            &self.out,
            (workspace.attended.read(), grid.pixels()),
            workspace.patches.get(ctx),
            |pixel, col, value| {
                let at = pixel * channels + col;
                residual.set(at, residual.get(at) + value);
            },
        );
        ctx.barrier();
    }
}

impl Load for PixelAttention {
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, channels: usize) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            qkv: scope.load("qkv", [3 * channels, channels])?,
            out: scope.load("out", [channels; 2])?,
        })
    }
}
