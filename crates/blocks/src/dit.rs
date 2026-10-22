//! DiT blocks: pre-RMSNorm layers (`layer`) with non-causal attention inside short sequences,
//! interleaved rotary embeddings (x-transformers) at positions `0..len`, and a GELU-tanh MLP.
//!
//! Tensors under a prefix `p`: `p.blocks.{i}.{norm1, qkv, out, norm2, ff1, ff2}` (q, k and v fused
//! into `qkv`, every linear with a bias), and for `FinalLayer`, `p.final_norm` and `p.final`.
//!
//! `forward` runs inside a team run over `Sequences`: consecutive sequences of equal length.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::attend;
use kernels::{Linear, NARROW_COLUMN_ALIGN};
use team::{Ctx, SharedBuf};

pub use crate::layer::Buffers;
use crate::layer::{
    Activation, Attention, FeedForward, Heads, Layer, Norm, Pass, Rope, Stack, Widths,
};
use crate::rotary::{Rotary, Rotation};

/// The dimensions of one DiT stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub depth: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.heads,
            head_dim: self.head_dim,
        }
    }

    /// Activations for passes of up to `rows` rows.
    pub fn buffers(&self, rows: usize, threads: usize) -> Buffers {
        let widths = Widths {
            hidden: self.hidden,
            heads: self.heads(),
            feed_forward: self.feed_forward,
        };
        Buffers::new(&widths, rows, threads)
    }
}

/// `rows` rows of `Buffers::hidden`, forming consecutive sequences of `length` tokens that
/// attend within themselves.
#[derive(Debug, Clone, Copy)]
pub struct Sequences {
    pub rows: usize,
    pub length: usize,
}

pub struct Dit {
    stack: Stack,
}

/// The output projection after the last block: RMSNorm, then a linear layer.
pub struct FinalLayer {
    norm: Norm,
    linear: Linear,
}

/// What a `FinalLayer` projects: rows of a stack of `shape` to `outputs` values each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    pub shape: Shape,
    pub outputs: usize,
}

impl Dit {
    /// All blocks over the rows of `buffers.hidden` that `sequences` describes.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, sequences: Sequences) {
        let attention = Within {
            length: sequences.length,
        };
        self.stack.forward(ctx, buffers, sequences.rows, &attention);
    }
}

impl FinalLayer {
    /// `out[i, cols] = final(norm(hidden[rows[i]]))` for this thread's output columns.
    pub fn project(&self, ctx: &Ctx, buffers: &Buffers, rows: &[usize], out: &SharedBuf) {
        let hidden = self.linear.matrix().cols;
        let scratch = buffers.scratch.get(ctx);
        let values = buffers.hidden.read();
        scratch.normed.resize(rows.len() * hidden, 0.0);
        for (normed, &row) in scratch.normed.chunks_exact_mut(hidden).zip(rows) {
            self.norm
                .apply(&values[row * hidden..(row + 1) * hidden], normed);
        }
        let outputs = self.linear.matrix().rows;
        let input = scratch.quant.input(&self.linear.matrix(), &scratch.normed);
        let cols = ctx.split(outputs, NARROW_COLUMN_ALIGN);
        self.linear.project(&input, cols, |row, col, value| {
            out.set(row * outputs + col, value);
        });
    }
}

/// Non-causal attention inside each sequence.
struct Within {
    length: usize,
}

impl Attention for Within {
    fn position(&self, row: usize) -> usize {
        row % self.length
    }

    fn keep(&self, _layer: usize, _unit: usize, _position: usize, _head: &[f32]) {}

    /// Split over (sequence, head) pairs.
    fn attend(&self, pass: &Pass, _layer: usize) {
        let Heads {
            heads, head_dim, ..
        } = pass.heads;
        let (qkv_width, width) = (pass.heads.qkv(), pass.heads.width());
        let length = self.length;
        let scratch = pass.buffers.scratch.get(pass.ctx);
        let qkv = pass.buffers.qkv.read();
        for unit in pass.ctx.split(pass.rows / length * heads, 1) {
            let (sequence, head) = (unit / heads, unit % heads);
            let first = sequence * length;
            let at = |row: usize, part: usize| {
                (first + row) * qkv_width + part * width + head * head_dim
            };
            for row in 0..length {
                let out = (first + row) * width + head * head_dim;
                attend(
                    &qkv[at(row, 0)..at(row, 0) + head_dim],
                    length,
                    |j| {
                        (
                            &qkv[at(j, 1)..at(j, 1) + head_dim],
                            &qkv[at(j, 2)..at(j, 2) + head_dim],
                        )
                    },
                    &mut scratch.scores,
                    pass.buffers.attention.part(out..out + head_dim),
                );
            }
        }
    }
}

impl Load for Dit {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            heads,
            head_dim,
            ..
        } = shape;
        if heads * head_dim != hidden {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads of {head_dim} for a width of {hidden}"
            )));
        }
        let rotary = Rotary::new(Rotation::Sequence, shape.rope_base, head_dim);
        let blocks = load_each(
            store,
            &format!("{prefix}.blocks"),
            shape.depth,
            |store, name| block(store, name, shape),
        )?;
        Ok(Self {
            stack: Stack::new(shape.heads(), Some(Rope::Interleaved), rotary, blocks),
        })
    }
}

fn block(store: &mut Store, prefix: &str, shape: Shape) -> Result<Layer, bundle::Error> {
    let Shape {
        hidden,
        feed_forward,
        norm_eps: eps,
        ..
    } = shape;
    let mut scope = Scope::new(store, prefix);
    let rms = |weight| Norm::Rms { weight, eps };
    Ok(Layer {
        norm1: rms(scope.load("norm1", hidden)?),
        qkv: scope.load("qkv", [shape.heads().qkv(), hidden])?,
        out: scope.load("out", [hidden, hidden])?,
        norm2: rms(scope.load("norm2", hidden)?),
        feed_forward: FeedForward::Plain {
            up: scope.load("ff1", [feed_forward, hidden])?,
            down: scope.load("ff2", [hidden, feed_forward])?,
            activation: Activation::GeluTanh,
        },
    })
}

impl Load for FinalLayer {
    type Shape = Projection;

    fn load(
        store: &mut Store,
        prefix: &str,
        projection: Projection,
    ) -> Result<Self, bundle::Error> {
        let Projection { shape, outputs } = projection;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm: Norm::Rms {
                weight: scope.load("final_norm", shape.hidden)?,
                eps: shape.norm_eps,
            },
            linear: scope.load("final", [outputs, shape.hidden])?,
        })
    }
}
