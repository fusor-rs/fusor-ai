//! A vision tower whose layers interleave full attention among an image's patches with gated 2-D
//! convolutions over its patch grid (LFM3-VL's): each patch's embedding (a linear over its
//! pixels, RMS-normed), then pre-norm layers, each mixing the patches by its `Mixer` and followed
//! by a SiLU-gated MLP. An attention layer is Qwen3's (`layer`), grouped-query, its query and key
//! heads RMS-normed and rotated by the patch's row and column (`Rotation::Grid`), attending on the
//! int8 kernels as `vit` does; a conv layer's convolution is `interleaved`'s gated one over the
//! `kernel × kernel` patches around each.
//!
//! Tensors under a prefix `p`: `p.patch` (a linear), `p.patch_norm`, and
//! `p.layers.{i}.{input_norm, post_norm, gate, up, down}` with an attention layer's
//! `{qkv, q_norm, k_norm, o}` (q, k and v fused into `qkv`) or a conv layer's `{input, conv, out}`
//! (`hidden × kernel²` taps).
//!
//! `embed` and `encode` run inside a team run over the rows of `tower::Buffers`' encoder, one per
//! patch, row-major over the image's patch grid.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::{Exponentials, rms_norm};
use kernels::{COLUMN_ALIGN, Linear, Vector};
use team::Ctx;

use super::{Grid, Patches, tower};
use crate::interleaved::conv::{self, Reach};
use crate::layer::{
    self, Activation, FeedForward, Heads, Layer, LayerSpec, Norm, Pass, Rope, Source, Stack,
    Widths, attended_residual, feed_forward_residual, input, prepare,
};
use crate::rotary::{Rotary, Rotation};

/// How a layer mixes the patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    Attention,
    Conv,
}

/// The dimensions of one tower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    /// Each layer's mixer, in order.
    pub mixers: &'static [Mixer],
    pub heads: usize,
    pub kv_heads: usize,
    pub feed_forward: usize,
    /// Side of a conv layer's square of taps, odd.
    pub kernel: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    /// Values of one patch: its pixels, every channel.
    pub patch_values: usize,
    pub exponentials: Exponentials,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.hidden / self.heads,
        }
    }

    fn widths(&self) -> Widths {
        Widths {
            hidden: self.hidden,
            heads: self.heads(),
            mixer: self.heads().qkv().max(conv::input_width(self.hidden)),
            feed_forward: self.feed_forward,
        }
    }

    /// Activations for images of up to `patches` patches.
    pub fn buffers(&self, patches: usize, threads: usize) -> tower::Buffers {
        let encoder = layer::Buffers::new(&self.widths(), patches, threads);
        tower::Buffers::around(encoder, threads)
    }
}

/// A conv layer.
struct ConvLayer {
    norm1: Norm,
    conv: conv::Weights,
    norm2: Norm,
    feed_forward: FeedForward,
}

impl ConvLayer {
    fn load(store: &mut Store, prefix: &str, shape: &Shape) -> Result<Self, bundle::Error> {
        let (hidden, eps) = (shape.hidden, shape.norm_eps);
        let conv = conv::Weights::load(store, prefix, hidden, shape.kernel * shape.kernel)?;
        let mut scope = Scope::new(store, prefix);
        let mut norm = |name: &str| -> Result<Norm, bundle::Error> {
            Ok(Norm::Rms {
                weight: scope.load(name, hidden)?,
                eps,
            })
        };
        let (norm1, norm2) = (norm("input_norm")?, norm("post_norm")?);
        let feed_forward =
            FeedForward::gated(&mut scope, hidden, shape.feed_forward, Activation::Silu)?;
        Ok(Self {
            norm1,
            conv,
            norm2,
            feed_forward,
        })
    }

    fn forward(&self, pass: &Pass, grid: Grid) {
        let norm = Source::Normed(&self.norm1);
        prepare(pass, norm, &self.conv.input);
        pass.ctx.barrier();
        let normed = input(pass, norm, &self.conv.input);
        self.conv.mix(pass, &normed, &Reach::Grid(grid));
        attended_residual(pass, &self.conv.out, None);
        feed_forward_residual(pass, &self.norm2, &self.feed_forward, None);
    }
}

/// A layer: the attention layer of that index in the tower's stack, or a conv layer.
enum Mixing {
    Attention(usize),
    Conv(Box<ConvLayer>),
}

pub struct Tower {
    shape: Shape,
    patch: Linear,
    patch_norm: Vector,
    /// The attention layers, in order.
    attention: Stack,
    layers: Vec<Mixing>,
}

impl Tower {
    /// `encoder.hidden` = each patch's embedding, RMS-normed; then a barrier.
    pub fn embed(&self, ctx: &Ctx, buffers: &tower::Buffers, patches: &[f32]) {
        let width = self.shape.hidden;
        let input = buffers.quant.get(ctx).input(&self.patch.matrix(), patches);
        let hidden = &buffers.encoder.hidden;
        self.patch
            .project(&input, ctx.split(width, COLUMN_ALIGN), |row, col, value| {
                hidden.set(row * width + col, value);
            });
        ctx.barrier();
        let (weight, eps) = (self.patch_norm.values(), self.shape.norm_eps);
        let rows = patches.len() / self.shape.patch_values;
        buffers.norm_rows(ctx, rows, |row, out| rms_norm(row, weight, eps, out));
    }

    /// Every layer over the patches of `grid` in `encoder.hidden` (in and out).
    pub fn encode(&self, ctx: &Ctx, buffers: &tower::Buffers, grid: Grid) {
        let pass = Pass {
            ctx,
            buffers: &buffers.encoder,
            rows: grid.patches(),
            heads: self.shape.heads(),
        };
        let attention = Patches {
            columns: grid.columns,
            exponentials: self.shape.exponentials,
        };
        for layer in &self.layers {
            match layer {
                Mixing::Attention(index) => self.attention.forward_layer(&pass, *index, &attention),
                Mixing::Conv(conv) => conv.forward(&pass, grid),
            }
        }
    }
}

impl Load for Tower {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            heads,
            kv_heads,
            ..
        } = shape;
        if !hidden.is_multiple_of(heads) || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads over {kv_heads} KV heads for a width of {hidden}"
            )));
        }
        let widths = shape.widths();
        let mut attention_layers = Vec::new();
        let mut index = 0;
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.mixers.len(),
            |store, name| {
                let mixing = match shape.mixers[index] {
                    Mixer::Attention => {
                        let spec = &LayerSpec::QWEN3;
                        let layer = Layer::load(store, name, spec, &widths, shape.norm_eps)?;
                        attention_layers.push(layer);
                        Mixing::Attention(attention_layers.len() - 1)
                    }
                    Mixer::Conv => Mixing::Conv(Box::new(ConvLayer::load(store, name, &shape)?)),
                };
                index += 1;
                Ok(mixing)
            },
        )?;
        let head_dim = shape.heads().head_dim;
        let rotary = Rotary::new(Rotation::Grid, shape.rope_base, head_dim);
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            patch: scope.load("patch", [hidden, shape.patch_values])?,
            patch_norm: scope.load("patch_norm", hidden)?,
            attention: Stack::new(shape.heads(), Some(Rope::Half), rotary, attention_layers),
            layers,
            shape,
        })
    }
}
