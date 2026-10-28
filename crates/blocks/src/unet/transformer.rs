//! The spatial transformer of latent-diffusion UNets (`Transformer2DModel` of
//! `BasicTransformerBlock`s): `x + proj_out(blocks(proj_in(norm(x))))`, an image's pixels its
//! tokens. Each block is pre-LayerNorm: self-attention, cross-attention to the image's context
//! (its text) and a GEGLU feed-forward, `down(hidden(x) · gelu(gate(x)))` with the exact GELU,
//! each added back. The attention projections have no bias; their output projections do.
//!
//! Tensors under a prefix `p`: `p.norm` (a GroupNorm), the linears `p.proj_in` and `p.proj_out`,
//! and per block `p.blocks.{j}.{norm1, norm2, norm3}` (LayerNorms) and the linears `qkv` (the
//! self-attention's fused projection), `out1`, `query`, `context` (the cross-attention's keys,
//! then values, of a context row), `out2`, `hidden`, `gate` and `down`.

use bundle::{Load, Scope, Store, load_each};
use kernels::Linear;
use kernels::conv::{Grid, Patches, project_rows};
use kernels::ops::{Exponentials, gelu_erf};
use team::{Ctx, PerThread, SharedBuf};

use crate::attention::{self, Call, Operand, attend};
use crate::norm::{GroupNorm, Groups, LayerNorm};

/// The dimensions of one spatial transformer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shape {
    pub(crate) channels: usize,
    pub(crate) depth: usize,
    pub(crate) heads: usize,
    /// Width of the feed-forward's inner activations.
    pub(crate) inner: usize,
    /// Width of a context row.
    pub(crate) context: usize,
    pub(crate) groups: usize,
    pub(crate) group_eps: f32,
    pub(crate) layer_eps: f32,
}

pub(crate) struct Transformer {
    shape: Shape,
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<Block>,
    proj_out: Linear,
}

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    out1: Linear,
    norm2: LayerNorm,
    query: Linear,
    context: Linear,
    out2: Linear,
    norm3: LayerNorm,
    hidden: Linear,
    gate: Linear,
    down: Linear,
}

/// The buffers of one `forward`, rows of the image's pixels.
pub(crate) struct Planes<'buf> {
    pub(crate) normed: &'buf SharedBuf,
    pub(crate) tokens: &'buf SharedBuf,
    pub(crate) qkv: &'buf SharedBuf,
    pub(crate) attention: &'buf SharedBuf,
    pub(crate) inner: &'buf SharedBuf,
    pub(crate) statistics: &'buf SharedBuf,
    pub(crate) patches: &'buf PerThread<Patches>,
    pub(crate) attention_scratch: &'buf PerThread<attention::Scratch>,
}

/// The context of one call: per block, the cross-attention keys and values of `tokens` context
/// rows per image (`encode`).
pub(crate) struct Text<'buf> {
    pub(crate) layers: &'buf [SharedBuf],
    pub(crate) tokens: usize,
}

/// Everything one thread's pass over a transformer reads.
struct Step<'call> {
    ctx: &'call Ctx<'call>,
    planes: &'call Planes<'call>,
    grid: Grid,
    shape: Shape,
}

impl Transformer {
    pub(crate) fn channels(&self) -> usize {
        self.shape.channels
    }

    /// Its layers: `BasicTransformerBlock`s.
    pub(crate) fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// `x` (pixel-major rows of `grid`) through the transformer, in place.
    pub(crate) fn forward(
        &self,
        ctx: &Ctx,
        planes: &Planes,
        (x, grid): (&SharedBuf, Grid),
        text: &Text,
    ) {
        let step = Step {
            ctx,
            planes,
            grid,
            shape: self.shape,
        };
        let normed = (planes.normed, planes.statistics);
        self.norm
            .apply(ctx, (x, grid), normed, std::convert::identity);
        ctx.barrier();
        step.project(&self.proj_in, planes.normed, |at, value| {
            planes.tokens.set(at, value)
        });
        ctx.barrier();
        for (block, context) in self.blocks.iter().zip(text.layers) {
            block.forward(&step, (context, text.tokens));
        }
        step.project(&self.proj_out, planes.tokens, |at, value| {
            x.set(at, x.get(at) + value)
        });
        ctx.barrier();
    }

    /// Each block's cross-attention keys and values of `rows` context rows into its buffer of
    /// `out`, for this thread's share.
    pub(crate) fn encode(
        &self,
        ctx: &Ctx,
        (text, rows): (&[f32], usize),
        out: &[SharedBuf],
        patches: &mut Patches,
    ) {
        let width = 2 * self.shape.channels;
        for (block, keys_values) in self.blocks.iter().zip(out) {
            project_rows(
                ctx,
                &block.context,
                (text, rows),
                patches,
                |row, col, value| {
                    keys_values.set(row * width + col, value);
                },
            );
        }
    }
}

impl Block {
    fn forward(&self, step: &Step, (context, tokens): (&SharedBuf, usize)) {
        let Step {
            ctx, planes, shape, ..
        } = step;
        let channels = shape.channels;
        let add = |at: usize, value: f32| planes.tokens.set(at, planes.tokens.get(at) + value);
        step.normalize(&self.norm1);
        step.project(&self.qkv, planes.normed, |at, value| {
            planes.qkv.set(at, value)
        });
        ctx.barrier();
        let qkv = planes.qkv.read();
        let part = |offset| Operand {
            values: qkv,
            stride: 3 * channels,
            offset,
        };
        let pixels = step.grid.area();
        step.attend(part(0), (part(channels), part(2 * channels)), pixels);
        step.project(&self.out1, planes.attention, add);
        ctx.barrier();
        step.normalize(&self.norm2);
        step.project(&self.query, planes.normed, |at, value| {
            planes.qkv.set(at, value)
        });
        ctx.barrier();
        let query = Operand {
            values: planes.qkv.read(),
            stride: channels,
            offset: 0,
        };
        let part = |offset| Operand {
            values: context.read(),
            stride: 2 * channels,
            offset,
        };
        step.attend(query, (part(0), part(channels)), tokens);
        step.project(&self.out2, planes.attention, add);
        ctx.barrier();
        step.normalize(&self.norm3);
        step.project(&self.hidden, planes.normed, |at, value| {
            planes.inner.set(at, value)
        });
        // The gate's products split over the threads exactly as the hidden ones (the same rows
        // and outputs), so each element is gated by the thread that wrote it.
        step.project(&self.gate, planes.normed, |at, value| {
            planes.inner.set(at, planes.inner.get(at) * gelu_erf(value));
        });
        ctx.barrier();
        step.project(&self.down, planes.inner, add);
        ctx.barrier();
    }
}

impl Step<'_> {
    /// `normed = norm(tokens)` for this thread's rows, then a barrier.
    fn normalize(&self, norm: &LayerNorm) {
        let channels = self.shape.channels;
        let rows = self.ctx.split(self.grid.pixels(), 1);
        let range = rows.start * channels..rows.end * channels;
        let tokens = &self.planes.tokens.read()[range.clone()];
        norm.apply(tokens, self.shape.layer_eps, self.planes.normed.part(range));
        self.ctx.barrier();
    }

    /// `store(row · outputs + col, input[row] · W[col] + b[col])` over the pixels.
    fn project(&self, linear: &Linear, input: &SharedBuf, mut store: impl FnMut(usize, f32)) {
        let outputs = linear.matrix().rows;
        let rows = (input.read(), self.grid.pixels());
        let patches = self.planes.patches.get(self.ctx);
        project_rows(self.ctx, linear, rows, patches, |row, col, value| {
            store(row * outputs + col, value);
        });
    }

    /// Attention of each pixel's query over `keys` key and value rows of its image into
    /// `attention`, then a barrier.
    fn attend(&self, query: Operand, (key, value): (Operand, Operand), keys: usize) {
        let Shape {
            channels, heads, ..
        } = self.shape;
        let call = Call {
            heads,
            kv_heads: heads,
            head_dim: channels / heads,
            images: self.grid.images,
            queries: self.grid.area(),
            keys,
            query,
            key,
            value,
            exponentials: Exponentials::Exact,
        };
        let scratch = self.planes.attention_scratch.get(self.ctx);
        attend(self.ctx, &call, self.planes.attention, scratch);
        self.ctx.barrier();
    }
}

impl Load for Transformer {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let channels = shape.channels;
        if shape.heads == 0 || !channels.is_multiple_of(shape.heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {channels} channels in {} heads",
                shape.heads
            )));
        }
        let blocks = load_each(
            store,
            &format!("{prefix}.blocks"),
            shape.depth,
            |store, name| Block::load(store, name, shape),
        )?;
        let groups = Groups {
            channels,
            groups: shape.groups,
            eps: shape.group_eps,
        };
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            norm: scope.load("norm", groups)?,
            proj_in: scope.load("proj_in", [channels; 2])?,
            blocks,
            proj_out: scope.load("proj_out", [channels; 2])?,
        })
    }
}

impl Load for Block {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            channels,
            inner,
            context,
            ..
        } = shape;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", channels)?,
            qkv: scope.load("qkv", [3 * channels, channels])?,
            out1: scope.load("out1", [channels; 2])?,
            norm2: scope.load("norm2", channels)?,
            query: scope.load("query", [channels; 2])?,
            context: scope.load("context", [2 * channels, context])?,
            out2: scope.load("out2", [channels; 2])?,
            norm3: scope.load("norm3", channels)?,
            hidden: scope.load("hidden", [inner, channels])?,
            gate: scope.load("gate", [inner, channels])?,
            down: scope.load("down", [channels, inner])?,
        })
    }
}
