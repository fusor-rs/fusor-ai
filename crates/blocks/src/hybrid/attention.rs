//! Gated attention: grouped-query attention over the KV cache whose output is gated by a sigmoid,
//! of a second head per query head (each element its own gate) or of one value per query head.
//! Each query and key head is RMS-normalized where the shape says so, then rotated in its first
//! `rotary_dim` elements (a partial rotary embedding, rotated by halves).
//!
//! Tensors under a layer's prefix `p`: `p.qkv` (the query heads, their gate heads when each
//! element has its own, the key heads, the value heads, then one gate per query head when each
//! head has one), `p.q_norm` and `p.k_norm` (`head_dim` wide) where heads are normed, `p.o`.
//!
//! `mix` is two phases: the projection by heads, each head normed and rotated or through its
//! sigmoid, the keys and values kept; then attention and its gate, by query heads.

use bundle::{Scope, Store};
use kernels::ops::{rms_norm, rope_half, sigmoid};
use kernels::{Input, Linear, QUAD, Vector};

use crate::layer::{HeadNorm, HeadNorms, Heads, Pass, Scratch};
use crate::rotary::{Position, Rotary, Rotation};
use crate::transformer::{KvCache, Span};

/// What the sigmoid gate of the attention's output is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// A gate head per query head, after the query heads: each element its own gate (Qwen3.5).
    Elementwise,
    /// One gate per query head, after the value heads (Spark).
    Headwise,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Elements of a head the rotary embedding turns, from its first.
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub rotation: Rotation,
    pub head_norm: HeadNorm,
    pub gate: Gate,
}

impl Shape {
    pub(crate) fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// Gate heads of the fused projection.
    fn gate_heads(&self) -> usize {
        match self.gate {
            Gate::Elementwise => self.heads,
            Gate::Headwise => 0,
        }
    }

    /// Columns of the fused projection's heads, before the gates of whole heads.
    fn head_columns(&self) -> usize {
        (self.heads + self.gate_heads() + 2 * self.kv_heads) * self.head_dim
    }

    /// Columns of the fused projection.
    pub(crate) fn qkv(&self) -> usize {
        match self.gate {
            Gate::Elementwise => self.head_columns(),
            Gate::Headwise => self.head_columns() + self.heads,
        }
    }
}

pub(crate) struct Weights {
    pub(crate) qkv: Linear,
    norms: Option<HeadNorms>,
    pub(crate) out: Linear,
    rotary: Rotary,
    shape: Shape,
}

/// One call's attention: layer `slot` of the cache, the rows of `span`, row `r` rotated by
/// `positions[r]`.
pub(crate) struct Call<'call> {
    pub(crate) cache: &'call KvCache,
    pub(crate) slot: usize,
    pub(crate) span: Span,
    pub(crate) positions: &'call [Position],
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
    /// # Errors
    /// When the tensors are missing or misshapen, or a shape with a gate per head has query
    /// heads in other than whole quads.
    pub(crate) fn load(
        store: &mut Store,
        prefix: &str,
        shape: &Shape,
        hidden: usize,
        eps: f32,
    ) -> Result<Self, bundle::Error> {
        if shape.gate == Gate::Headwise && !shape.heads.is_multiple_of(QUAD) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {} query heads, a gate each, are not whole quads",
                shape.heads
            )));
        }
        let mut scope = Scope::new(store, prefix);
        let width = shape.heads * shape.head_dim;
        let norms = match shape.head_norm {
            HeadNorm::None => None,
            HeadNorm::Rms => Some(HeadNorms {
                query: scope.load("q_norm", shape.head_dim)?,
                key: scope.load("k_norm", shape.head_dim)?,
                eps,
            }),
        };
        Ok(Self {
            qkv: scope.load("qkv", [shape.qkv(), hidden])?,
            norms,
            out: scope.load("o", [hidden, width])?,
            rotary: Rotary::new(shape.rotation, shape.rope_base, shape.rotary_dim),
            shape: *shape,
        })
    }

    fn role(&self, unit: usize) -> Role {
        let Shape {
            heads, kv_heads, ..
        } = self.shape;
        let gate_heads = self.shape.gate_heads();
        match unit {
            unit if unit < heads => Role::Query,
            unit if unit < heads + gate_heads => Role::Gate,
            unit if unit < heads + gate_heads + kv_heads => Role::Key(unit - gate_heads),
            unit => Role::Value(unit - gate_heads),
        }
    }

    /// `buffers.attention`: the gated attention of the normed rows `input` (prepared for `qkv`).
    pub(crate) fn mix(&self, pass: &Pass, input: &Input, call: &Call) {
        self.project(pass, input, call);
        pass.ctx.barrier();
        let stride = self.shape.qkv();
        call.cache.attend(pass, call.slot, call.span, stride);
        self.gate(pass);
        pass.ctx.barrier();
    }

    /// This thread's heads of the projection: queries and keys normed and rotated, gates through
    /// their sigmoid, keys and values kept.
    fn project(&self, pass: &Pass, input: &Input, call: &Call) {
        let Shape {
            head_dim,
            rotary_dim,
            ..
        } = self.shape;
        let (buffers, width) = (pass.buffers, self.shape.qkv());
        let scratch = buffers.scratch.get(pass.ctx);
        let units = pass.ctx.split(self.shape.head_columns(), head_dim);
        self.qkv
            .project_quads(input, units.clone(), |row, first, values| {
                let at = row * width + first;
                buffers.qkv.part(at..at + QUAD).copy_from_slice(&values);
            });
        self.project_head_gates(pass, input);
        scratch.cos.resize(rotary_dim / 2, 0.0);
        scratch.sin.resize(rotary_dim / 2, 0.0);
        scratch.head.resize(head_dim, 0.0);
        for row in 0..pass.rows {
            let position = call.span.first + row;
            let (cos, sin) = (&mut scratch.cos, &mut scratch.sin);
            self.rotary.angles(call.positions[row], cos, sin);
            for start in units.clone().step_by(head_dim) {
                let at = row * width + start;
                let head = buffers.qkv.part(at..at + head_dim);
                match self.role(start / head_dim) {
                    Role::Query => self.position(|norms| &norms.query, head, scratch),
                    Role::Gate => head.iter_mut().for_each(|gate| *gate = sigmoid(*gate)),
                    Role::Key(unit) => {
                        self.position(|norms| &norms.key, head, scratch);
                        call.cache.keep(call.slot, unit, position, head);
                    }
                    Role::Value(unit) => call.cache.keep(call.slot, unit, position, head),
                }
            }
        }
    }

    /// This thread's quads of the gates of whole heads, through their sigmoid, where the shape
    /// has them.
    fn project_head_gates(&self, pass: &Pass, input: &Input) {
        if self.shape.gate != Gate::Headwise {
            return;
        }
        let (buffers, width) = (pass.buffers, self.shape.qkv());
        let first_gate = self.shape.head_columns();
        let quads = pass.ctx.split(self.shape.heads, QUAD);
        let columns = first_gate + quads.start..first_gate + quads.end;
        self.qkv
            .project_quads(input, columns, |row, first, values| {
                let at = row * width + first;
                buffers
                    .qkv
                    .part(at..at + QUAD)
                    .copy_from_slice(&values.map(sigmoid));
            });
    }

    /// A query or key head normed by its `weight`, where the shape norms heads, then rotated in
    /// its first `rotary_dim` elements.
    fn position(
        &self,
        weight: impl Fn(&HeadNorms) -> &Vector,
        head: &mut [f32],
        scratch: &mut Scratch,
    ) {
        if let Some(norms) = &self.norms {
            scratch.head.copy_from_slice(head);
            rms_norm(&scratch.head, weight(norms).values(), norms.eps, head);
        }
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
                let gates = self.gates(&qkv[row * stride..][..stride], head);
                let attended = buffers.attention.part(at..at + head_dim);
                for (value, gate) in attended.iter_mut().zip(gates.iter().cycle()) {
                    *value *= gate;
                }
            }
        }
    }

    /// Query head `head`'s gates in a row of the projection: one per element, or one for all.
    fn gates<'row>(&self, row: &'row [f32], head: usize) -> &'row [f32] {
        let Shape {
            heads, head_dim, ..
        } = self.shape;
        match self.shape.gate {
            Gate::Elementwise => &row[(heads + head) * head_dim..][..head_dim],
            Gate::Headwise => std::slice::from_ref(&row[self.shape.head_columns() + head]),
        }
    }
}
