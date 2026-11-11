//! The single-stream block: every row the same weights, and for Motif-Video's a cross-attention
//! from the stream to the context's text.

use bundle::{Load, Scope, Store};
use kernels::ops::{Exponentials, gelu_tanh};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use super::pass::{HeadNorms, Modulation, Pass, modulate, normalize_heads};
use super::{Buffers, Cross, Rows, Shape};
use crate::attention::{Call, Operand, attend};
use crate::rotary::Rotary;
use crate::row_work::RowScratch;

/// The stream's attention output as queries over the context's text rows as keys and values.
struct CrossAttention {
    query: Linear,
    /// The keys, then the values.
    context: Linear,
    out: Linear,
    query_norm: Vector,
    key_norm: Vector,
}

impl CrossAttention {
    /// `attention`'s stream rows += the cross-attention's output; every phase ends with a barrier.
    fn forward(&self, pass: &Pass) {
        self.project_inputs(pass);
        self.prepare(pass);
        self.attend(pass);
        let buffers = pass.buffers;
        let hidden = pass.shape.hidden;
        let crossed = (buffers.crossed.read(), pass.stream());
        buffers
            .rows
            .project(pass.ctx, &self.out, crossed, |row, col, value| {
                let at = row * hidden + col;
                buffers.attention.set(at, buffers.attention.get(at) + value);
            });
        pass.ctx.barrier();
    }

    /// The queries of the stream's attention output, the keys and values of the normed text rows.
    fn project_inputs(&self, pass: &Pass) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let hidden = shape.hidden;
        let queries = (buffers.attention.read(), pass.stream());
        buffers
            .rows
            .project(ctx, &self.query, queries, |row, col, value| {
                buffers.cross_query.set(row * hidden + col, value);
            });
        let first_text = pass.total() - pass.rows.text;
        let text = (
            &buffers.normed.read()[first_text * hidden..],
            pass.rows.text,
        );
        buffers
            .rows
            .project(ctx, &self.context, text, |row, col, value| {
                buffers.cross_context.set(row * 2 * hidden + col, value);
            });
        ctx.barrier();
    }

    /// The query heads normalized and turned, the key heads normalized.
    fn prepare(&self, pass: &Pass) {
        let stream = pass.stream();
        let scratch = pass.buffers.rows.scratch(pass.ctx);
        for row in pass.ctx.split(stream + pass.rows.text, 1) {
            match row.checked_sub(stream) {
                None => self.prepare_query(pass, row, scratch),
                Some(text_row) => self.prepare_key(pass, text_row, scratch),
            }
        }
        pass.ctx.barrier();
    }

    fn prepare_query(&self, pass: &Pass, row: usize, scratch: &mut RowScratch) {
        let hidden = pass.shape.hidden;
        let heads = pass
            .buffers
            .cross_query
            .part(row * hidden..(row + 1) * hidden);
        normalize_heads(heads, (&self.query_norm, pass.shape.eps), &mut scratch.row);
        pass.rotate(heads, pass.rows.positions[row], scratch);
    }

    fn prepare_key(&self, pass: &Pass, text_row: usize, scratch: &mut RowScratch) {
        let hidden = pass.shape.hidden;
        let start = text_row * 2 * hidden;
        let heads = pass.buffers.cross_context.part(start..start + hidden);
        normalize_heads(heads, (&self.key_norm, pass.shape.eps), &mut scratch.row);
    }

    /// `crossed` = the stream rows' attention over the text rows.
    fn attend(&self, pass: &Pass) {
        let buffers = pass.buffers;
        let hidden = pass.shape.hidden;
        let context = |offset| Operand {
            values: buffers.cross_context.read(),
            stride: 2 * hidden,
            offset,
        };
        let call = Call {
            heads: pass.shape.heads,
            kv_heads: pass.shape.heads,
            head_dim: pass.shape.head_dim(),
            images: 1,
            queries: pass.stream(),
            keys: pass.rows.text,
            query: Operand {
                values: buffers.cross_query.read(),
                stride: hidden,
                offset: 0,
            },
            key: context(0),
            value: context(hidden),
            exponentials: Exponentials::Exact,
        };
        attend(
            pass.ctx,
            &call,
            &buffers.crossed,
            buffers.scratch.get(pass.ctx),
        );
        pass.ctx.barrier();
    }
}

impl Load for CrossAttention {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let hidden = shape.hidden;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            query: scope.load("query", [hidden, hidden])?,
            context: scope.load("context", [2 * hidden, hidden])?,
            out: scope.load("out", [hidden, hidden])?,
            query_norm: scope.load("query_norm", shape.head_dim())?,
            key_norm: scope.load("key_norm", shape.head_dim())?,
        })
    }
}

pub struct Single {
    shape: Shape,
    rotary: Rotary,
    modulation: Linear,
    qkv: Linear,
    mlp: Linear,
    out: Linear,
    q_norm: Vector,
    k_norm: Vector,
    cross: Option<CrossAttention>,
}

impl Single {
    /// This block's modulations of `time` (the SiLU of the time embedding) into `out`, every scale
    /// as `1 +` it; this thread's columns, no barrier.
    pub fn modulate(&self, ctx: &Ctx, buffers: &Buffers, (time, out): (&[f32], &SharedBuf)) {
        modulate(ctx, buffers, &self.modulation, (time, out));
    }

    /// The block over `rows` of `buffers.hidden` (in and out) under `modulation` (its
    /// `modulate`'s).
    ///
    /// # Panics
    /// When the buffers hold fewer rows than `rows` has.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, modulation: &[f32], rows: Rows) {
        let pass = Pass::new(ctx, buffers, (&self.shape, &self.rotary), rows);
        let norm = Modulation::of(modulation, self.shape.hidden, 0);
        pass.modulated_norm(&norm, &norm);
        self.project_inputs(&pass);
        let norms = HeadNorms {
            query: &self.q_norm,
            key: &self.k_norm,
        };
        pass.head_norms(norms, norms);
        pass.joint_attention();
        if let Some(cross) = &self.cross {
            cross.forward(&pass);
        }
        self.join_outputs(&pass);
        pass.gated(&self.out, &buffers.inner, 0..pass.total(), norm.gate);
        ctx.barrier();
    }

    /// `qkv` and the MLP's activations (into `inner`, after the attention output's room) of
    /// `normed`; then a barrier.
    fn project_inputs(&self, pass: &Pass) {
        let Pass { ctx, buffers, .. } = pass;
        let Shape {
            hidden,
            feed_forward,
            ..
        } = *pass.shape;
        let normed = (buffers.normed.read(), pass.total());
        buffers
            .rows
            .project(ctx, &self.qkv, normed, |row, col, value| {
                buffers.qkv.set(row * 3 * hidden + col, value);
            });
        buffers
            .rows
            .project(ctx, &self.mlp, normed, |row, col, value| {
                let at = row * (hidden + feed_forward) + hidden + col;
                buffers.inner.set(at, gelu_tanh(value));
            });
        ctx.barrier();
    }

    /// The attention output into `inner`, in front of the MLP's activations; then a barrier.
    fn join_outputs(&self, pass: &Pass) {
        let hidden = pass.shape.hidden;
        let width = hidden + pass.shape.feed_forward;
        for row in pass.ctx.split(pass.total(), 1) {
            let attention = &pass.buffers.attention.read()[row * hidden..(row + 1) * hidden];
            pass.buffers
                .inner
                .part(row * width..row * width + hidden)
                .copy_from_slice(attention);
        }
        pass.ctx.barrier();
    }
}

impl Load for Single {
    type Shape = (Shape, Cross);

    fn load(
        store: &mut Store,
        prefix: &str,
        (shape, cross): (Shape, Cross),
    ) -> Result<Self, bundle::Error> {
        let rotary = shape.rotary(prefix)?;
        let Shape {
            hidden,
            feed_forward,
            ..
        } = shape;
        let head_dim = shape.head_dim();
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            rotary,
            modulation: scope.load("modulation", [shape.single_modulations(), hidden])?,
            qkv: scope.load("qkv", [3 * hidden, hidden])?,
            mlp: scope.load("mlp", [feed_forward, hidden])?,
            out: scope.load("out", [hidden, hidden + feed_forward])?,
            q_norm: scope.load("q_norm", head_dim)?,
            k_norm: scope.load("k_norm", head_dim)?,
            cross: match cross {
                Cross::None => None,
                Cross::Text => Some(scope.load("cross", shape)?),
            },
        })
    }
}
