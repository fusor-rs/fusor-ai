//! Vision transformer encoders over the patches of one image of any size, as SigLIP's NaViT
//! variant runs them: pre-LayerNorm layers, a fused QKV, full attention among all the image's
//! patches on the int8 kernels (`attention`) with 2-D rotary embeddings by patch row and column
//! (`Rotation::Grid`), and a GELU-tanh MLP; every linear has a bias.
//!
//! Tensors under a prefix `p`: `p.layers.{i}.{norm1, qkv, out, norm2, ff1, ff2}` (q, k and v
//! fused into `qkv`), the norms as `weight` and `bias` vectors.
//!
//! `forward` runs inside a team run over the rows of `Buffers::hidden`, one per patch, row-major
//! over the image's patch grid.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::gelu_tanh;
use team::Ctx;

use crate::attention::{self, Call, Operand};
pub use crate::layer::Buffers;
use crate::layer::{Attention, FeedForward, Heads, Layer, Norm, Pass, Rope, Stack, Widths};
use crate::norm::LayerNorm;
use crate::rotary::{Position, Rotary, Rotation};

/// The dimensions of one encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub feed_forward: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.heads,
            head_dim: self.hidden / self.heads,
        }
    }

    /// Activations for images of up to `patches` patches.
    pub fn buffers(&self, patches: usize, threads: usize) -> Buffers {
        let widths = Widths {
            hidden: self.hidden,
            heads: self.heads(),
            feed_forward: self.feed_forward,
        };
        Buffers::new(&widths, patches, threads)
    }
}

/// An image's patch grid: `rows × columns` patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub columns: usize,
}

impl Grid {
    pub fn patches(self) -> usize {
        self.rows * self.columns
    }
}

pub struct Vit {
    stack: Stack,
}

impl Vit {
    /// Every layer over the patches of `grid` in `buffers.hidden` (in and out).
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        let patches = grid.patches();
        let attention = Patches {
            columns: grid.columns,
        };
        self.stack.forward(ctx, buffers, patches, &attention);
    }
}

/// Attention among all the patches of one image, rotated by each patch's row and column.
struct Patches {
    columns: usize,
}

impl Attention for Patches {
    fn position(&self, row: usize) -> usize {
        row
    }

    fn rotary(&self, row: usize) -> Position {
        Position {
            temporal: 0,
            height: row / self.columns,
            width: row % self.columns,
        }
    }

    fn keep(&self, _layer: usize, _unit: usize, _position: usize, _head: &[f32]) {}

    /// On the int8 kernels (`attention`), split over (head, block of queries) units.
    fn attend(&self, pass: &Pass, _layer: usize) {
        let Heads {
            heads, head_dim, ..
        } = pass.heads;
        let (qkv_width, width) = (pass.heads.qkv(), pass.heads.width());
        let qkv = pass.buffers.qkv.read();
        let operand = |part: usize| Operand {
            values: qkv,
            stride: qkv_width,
            offset: part * width,
        };
        let call = Call {
            heads,
            head_dim,
            images: 1,
            queries: pass.rows,
            keys: pass.rows,
            query: operand(0),
            key: operand(1),
            value: operand(2),
        };
        let scratch = &mut pass.buffers.scratch.get(pass.ctx).attention;
        attention::attend(pass.ctx, &call, &pass.buffers.attention, scratch);
    }
}

impl Load for Vit {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape { hidden, heads, .. } = shape;
        if !hidden.is_multiple_of(heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads for a width of {hidden}"
            )));
        }
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| layer(store, name, shape),
        )?;
        let head_dim = shape.heads().head_dim;
        let rotary = Rotary::new(Rotation::Grid, shape.rope_base, head_dim);
        Ok(Self {
            stack: Stack::new(shape.heads(), Some(Rope::Half), rotary, layers),
        })
    }
}

fn layer(store: &mut Store, prefix: &str, shape: Shape) -> Result<Layer, bundle::Error> {
    let Shape {
        hidden,
        feed_forward,
        norm_eps: eps,
        ..
    } = shape;
    let mut scope = Scope::new(store, prefix);
    let layer_norm = |norm: LayerNorm| Norm::Layer { norm, eps };
    Ok(Layer {
        norm1: layer_norm(scope.load("norm1", hidden)?),
        qkv: scope.load("qkv", [shape.heads().qkv(), hidden])?,
        out: scope.load("out", [hidden, hidden])?,
        norm2: layer_norm(scope.load("norm2", hidden)?),
        feed_forward: FeedForward::Plain {
            up: scope.load("ff1", [feed_forward, hidden])?,
            down: scope.load("ff2", [hidden, feed_forward])?,
            activation: gelu_tanh,
        },
    })
}
