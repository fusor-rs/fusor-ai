//! Gated attention: grouped-query attention over the KV cache whose output is gated by a sigmoid
//! of a second head per query head. Each query and key head is RMS-normalized, then rotated in
//! its first `rotary_dim` elements (a partial rotary embedding, rotated by halves).
//!
//! Tensors under a layer's prefix `p`: `p.qkv` (the query heads, their gate heads, the key heads,
//! the value heads), `p.q_norm` and `p.k_norm` (`head_dim` wide), `p.o`.
//!
//! `mix` is two phases: the projection by heads, each head normed and rotated or through its
//! sigmoid, the keys and values kept; then attention and its gate, by query heads.

use bundle::{Scope, Store};
use kernels::ops::{rms_norm, rope_half, sigmoid};
use kernels::{Input, Linear, QUAD, Vector};

use crate::layer::{HeadNorms, Heads, Pass, Scratch};
use crate::rotary::{Position, Rotary, Rotation};
use crate::transformer::{KvCache, Span};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Elements of a head the rotary embedding turns, from its first.
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub rotation: Rotation,
}

impl Shape {
    pub(super) fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// Columns of the fused projection.
    pub(super) fn qkv(&self) -> usize {
        (2 * self.heads + 2 * self.kv_heads) * self.head_dim
    }
}

pub(super) struct Weights {
    pub(super) qkv: Linear,
    norms: HeadNorms,
    pub(super) out: Linear,
    shape: Shape,
}

/// One call's attention: layer `slot` of the cache, the rows of `span`, row `r` rotated by
/// `positions[r]`.
pub(super) struct Call<'call> {
    pub(super) cache: &'call KvCache,
    pub(super) slot: usize,
    pub(super) span: Span,
    pub(super) positions: &'call [Position],
}

/// What a head of the fused projection is.
enum Role {
    Query,
    Gate,
    /// A key or value head, numbered as the cache numbers them (query heads first).
    Key(usize),
    Value(usize),
}

impl Weights {
    pub(super) fn load(
        store: &mut Store,
        prefix: &str,
        shape: &Shape,
        hidden: usize,
        eps: f32,
    ) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        let width = shape.heads * shape.head_dim;
        Ok(Self {
            qkv: scope.load("qkv", [shape.qkv(), hidden])?,
            norms: HeadNorms {
                query: scope.load("q_norm", shape.head_dim)?,
                key: scope.load("k_norm", shape.head_dim)?,
                eps,
            },
            out: scope.load("o", [hidden, width])?,
            shape: *shape,
        })
    }

    fn role(&self, unit: usize) -> Role {
        let Shape {
            heads, kv_heads, ..
        } = self.shape;
        match unit {
            unit if unit < heads => Role::Query,
            unit if unit < 2 * heads => Role::Gate,
            unit if unit < 2 * heads + kv_heads => Role::Key(unit - heads),
            unit => Role::Value(unit - heads),
        }
    }

    /// `buffers.attention`: the gated attention of the normed rows `input` (prepared for `qkv`).
    pub(super) fn mix(&self, pass: &Pass, input: &Input, rotary: &Rotary, call: &Call) {
        self.project(pass, input, rotary, call);
        pass.ctx.barrier();
        let stride = self.shape.qkv();
        call.cache.attend(pass, call.slot, call.span, stride);
        self.gate(pass);
        pass.ctx.barrier();
    }

    /// This thread's heads of the projection: queries and keys normed and rotated, gates through
    /// their sigmoid, keys and values kept.
    fn project(&self, pass: &Pass, input: &Input, rotary: &Rotary, call: &Call) {
        let Shape {
            head_dim,
            rotary_dim,
            ..
        } = self.shape;
        let (buffers, width) = (pass.buffers, self.shape.qkv());
        let scratch = buffers.scratch.get(pass.ctx);
        let units = pass.ctx.split(width, head_dim);
        self.qkv
            .project_quads(input, units.clone(), |row, first, values| {
                let at = row * width + first;
                buffers.qkv.part(at..at + QUAD).copy_from_slice(&values);
            });
        scratch.cos.resize(rotary_dim / 2, 0.0);
        scratch.sin.resize(rotary_dim / 2, 0.0);
        scratch.head.resize(head_dim, 0.0);
        for row in 0..pass.rows {
            let position = call.span.first + row;
            rotary.angles(call.positions[row], &mut scratch.cos, &mut scratch.sin);
            for start in units.clone().step_by(head_dim) {
                let at = row * width + start;
                let head = buffers.qkv.part(at..at + head_dim);
                match self.role(start / head_dim) {
                    Role::Query => self.position(&self.norms.query, head, scratch),
                    Role::Gate => head.iter_mut().for_each(|gate| *gate = sigmoid(*gate)),
                    Role::Key(unit) => {
                        self.position(&self.norms.key, head, scratch);
                        call.cache.keep(call.slot, unit, position, head);
                    }
                    Role::Value(unit) => call.cache.keep(call.slot, unit, position, head),
                }
            }
        }
    }

    /// A query or key head normed by `weight`, then rotated in its first `rotary_dim` elements.
    fn position(&self, weight: &Vector, head: &mut [f32], scratch: &mut Scratch) {
        scratch.head.copy_from_slice(head);
        rms_norm(&scratch.head, weight.values(), self.norms.eps, head);
        rope_half(
            &mut head[..self.shape.rotary_dim],
            &scratch.cos,
            &scratch.sin,
        );
    }

    /// This thread's query heads of `buffers.attention` times their gates.
    fn gate(&self, pass: &Pass) {
        let Shape {
            heads, head_dim, ..
        } = self.shape;
        let (buffers, stride, width) = (pass.buffers, self.shape.qkv(), heads * head_dim);
        let qkv = buffers.qkv.read();
        for head in pass.ctx.split(heads, 1) {
            for row in 0..pass.rows {
                let at = row * width + head * head_dim;
                let gate = &qkv[row * stride + (heads + head) * head_dim..][..head_dim];
                let attended = buffers.attention.part(at..at + head_dim);
                for (value, gate) in attended.iter_mut().zip(gate) {
                    *value *= gate;
                }
            }
        }
    }
}
