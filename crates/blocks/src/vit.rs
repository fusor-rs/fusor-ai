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

pub mod merger;
pub mod preprocessing;
pub mod tower;

use std::ops::Range;

use bundle::{Load, Store, load_each};
use kernels::ops::Exponentials;
use team::Ctx;

use crate::attention::{self, Call, Operand};
pub use crate::layer::Buffers;
use crate::layer::{Attention, Heads, Layer, LayerSpec, Pass, Stack, Widths};
use crate::rotary::{Position, Rotary, Rotation};
pub use preprocessing::Preprocessing;

/// The dimensions of one encoder, and the exponential its attention takes of the scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub feed_forward: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub exponentials: Exponentials,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.heads,
            head_dim: self.hidden / self.heads,
        }
    }

    fn widths(&self) -> Widths {
        Widths {
            hidden: self.hidden,
            heads: self.heads(),
            mixer: self.heads().qkv(),
            feed_forward: self.feed_forward,
        }
    }

    /// Activations for images of up to `patches` patches.
    pub fn buffers(&self, patches: usize, threads: usize) -> Buffers {
        Buffers::new(&self.widths(), patches, threads)
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

    /// The patches of the `token`th `merge × merge` block, the blocks row-major (the grid's
    /// sides are multiples of `merge`), and the patches of the block row-major too.
    pub fn merged_patches(self, token: usize, merge: usize) -> impl Iterator<Item = usize> {
        let blocks_across = self.columns / merge;
        let (top, left) = (token / blocks_across * merge, token % blocks_across * merge);
        (0..merge * merge).map(move |at| (top + at / merge) * self.columns + left + at % merge)
    }
}

/// Every row's rotary position in a language model's prompt of `rows` rows, with the merged
/// tokens of images at `images` (their rows, in order, and their grids of tokens): a text row's
/// on every axis, continuing from the last; an image token's at the image's start, down its row
/// and across its column; the text after an image from past its longer side. A video's frame
/// groups, separated by their timestamps' text, are images here (Qwen3-VL, Qwen3.5).
pub fn positions(rows: usize, images: &[(Range<usize>, Grid)]) -> Vec<Position> {
    let mut positions = Vec::with_capacity(rows);
    let mut images = images.iter().peekable();
    let mut next = 0;
    while positions.len() < rows {
        let row = positions.len();
        let Some((tokens, grid)) = images.next_if(|(tokens, _)| tokens.start == row) else {
            positions.push(Position::uniform(next));
            next += 1;
            continue;
        };
        positions.extend((0..tokens.len()).map(|token| Position {
            temporal: next,
            height: next + token / grid.columns,
            width: next + token % grid.columns,
        }));
        next += grid.rows.max(grid.columns);
    }
    positions
}

pub struct Vit {
    stack: Stack,
    exponentials: Exponentials,
}

impl Vit {
    /// Every layer over the patches of `grid` in `buffers.hidden` (in and out).
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        let patches = grid.patches();
        let attention = Patches {
            columns: grid.columns,
            exponentials: self.exponentials,
        };
        self.stack.forward(ctx, buffers, patches, &attention);
    }
}

/// Attention among all the patches of one image, rotated by each patch's row and column.
struct Patches {
    columns: usize,
    exponentials: Exponentials,
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
            exponentials: self.exponentials,
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
            |store, name| {
                Layer::load(
                    store,
                    name,
                    &LayerSpec::VIT,
                    &shape.widths(),
                    shape.norm_eps,
                )
            },
        )?;
        let head_dim = shape.heads().head_dim;
        let rotary = Rotary::new(Rotation::Grid, shape.rope_base, head_dim);
        Ok(Self {
            stack: Stack::new(shape.heads(), LayerSpec::VIT.rope, rotary, layers),
            exponentials: shape.exponentials,
        })
    }
}
