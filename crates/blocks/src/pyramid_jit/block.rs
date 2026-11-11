//! One P-JiT block: attention, then a SiLU-gated feed-forward, each between an RMSNorm scaled by
//! the timestep and a post-norm, added to the residual stream through a gate.
//!
//! A block runs over a range of `hidden`'s rows (one sequence; the rows below `Pass::tokens` are
//! the image's patches); its other buffers hold those rows from their row 0. Between two barriers
//! each phase reads buffers written before the first and writes disjoint ranges: a thread's rows,
//! or the output columns of a product.

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::ops::{Exponentials, rms_norm, rope_interleaved, sigmoid, silu};
use kernels::{Linear, Vector};
use team::SharedBuf;

use super::{BLOCK_CHUNKS, Chunk, Pass, Shape};
use crate::attention::{Call, Operand, attend};
use crate::section;

pub(super) struct Block {
    norm1: Vector,
    qkv_gate: Linear,
    q_norm: Vector,
    k_norm: Vector,
    out: Linear,
    post_norm1: Vector,
    norm2: Vector,
    ff1: Linear,
    ff2: Linear,
    post_norm2: Vector,
    /// From the timestep's root to the block's modulations.
    pub(super) modulation: Linear,
}

/// Chunk `chunk` of block `index`'s modulations at the timestep last given to `Dit::time`.
fn modulation<'call>(pass: &'call Pass, index: usize, chunk: Chunk) -> &'call [f32] {
    let hidden = pass.shape.hidden;
    let block = &pass.buffers.modulations.read()[index * BLOCK_CHUNKS * hidden..];
    section(block, hidden, chunk as usize)
}

impl Block {
    /// The block over `rows` of `hidden`; `index` selects its modulations.
    pub(super) fn forward(&self, pass: &Pass, rows: Range<usize>, index: usize) {
        self.attention(pass, &rows, index);
        self.feed_forward(pass, &rows, index);
    }

    fn attention(&self, pass: &Pass, rows: &Range<usize>, index: usize) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (hidden, len, stride) = (shape.hidden, rows.len(), shape.qkv_gate());
        let scale = modulation(pass, index, Chunk::AttentionScale);
        scaled_norm(pass, rows, &self.norm1, scale);
        let normed = (&buffers.normed, len);
        buffers
            .rows
            .project_synced(ctx, &self.qkv_gate, normed, |row, col, value| {
                buffers.qkv.set(row * stride + col, value);
            });
        self.normalize_and_rotate(pass, rows);
        ctx.barrier();
        let part = |offset| Operand {
            values: buffers.qkv.read(),
            stride,
            offset,
        };
        let call = Call {
            heads: shape.heads,
            kv_heads: shape.heads,
            head_dim: shape.head_dim(),
            images: 1,
            queries: len,
            keys: len,
            query: part(0),
            key: part(hidden),
            value: part(2 * hidden),
            exponentials: Exponentials::Exact,
        };
        attend(ctx, &call, &buffers.attention, buffers.scratch.get(ctx));
        ctx.barrier();
        gate_heads(pass, len);
        let gate = modulation(pass, index, Chunk::AttentionGate);
        add_gated(
            pass,
            rows,
            (&self.out, &buffers.attention),
            (&self.post_norm1, gate),
        );
    }

    /// This thread's rows' queries and keys RMS-normalized over their whole width, then the
    /// image's tokens' turned by the rotary angles of their patch row and column; a text row's
    /// stay as they are (the identity).
    fn normalize_and_rotate(&self, pass: &Pass, rows: &Range<usize>) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (hidden, len, head_dim) = (shape.hidden, rows.len(), shape.head_dim());
        let stride = shape.qkv_gate();
        let q_norm = (&self.q_norm, shape.eps);
        let k_norm = (&self.k_norm, shape.eps);
        buffers
            .rows
            .rms_rows(ctx, &buffers.qkv, (len, stride, 0), q_norm);
        buffers
            .rows
            .rms_rows(ctx, &buffers.qkv, (len, stride, hidden), k_norm);
        let scratch = buffers.rows.scratch(ctx);
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        for row in ctx.split(len, 1) {
            let patch = rows.start + row;
            if patch >= pass.tokens() {
                continue;
            }
            let position = pass.patching.position(patch);
            pass.rotary
                .angles(position, &mut scratch.cos, &mut scratch.sin);
            let queries_and_keys = buffers.qkv.part(row * stride..row * stride + 2 * hidden);
            for head in queries_and_keys.chunks_exact_mut(head_dim) {
                rope_interleaved(head, &scratch.cos, &scratch.sin);
            }
        }
    }

    fn feed_forward(&self, pass: &Pass, rows: &Range<usize>, index: usize) {
        let Pass { ctx, buffers, .. } = pass;
        let len = rows.len();
        let scale = modulation(pass, index, Chunk::FeedForwardScale);
        scaled_norm(pass, rows, &self.norm2, scale);
        let width = self.ff2.matrix().cols;
        let normed = (&buffers.normed, len);
        buffers
            .rows
            .project_synced(ctx, &self.ff1, normed, |row, col, value| {
                buffers.inner.set(row * 2 * width + col, value);
            });
        let inner = buffers.inner.read();
        for row in ctx.split(len, 1) {
            let gates = &inner[row * 2 * width..row * 2 * width + width];
            let values = &inner[row * 2 * width + width..(row + 1) * 2 * width];
            let gated = buffers.gated.part(row * width..(row + 1) * width);
            for ((target, &gate), &value) in gated.iter_mut().zip(gates).zip(values) {
                *target = silu(gate) * value;
            }
        }
        ctx.barrier();
        let gate = modulation(pass, index, Chunk::FeedForwardGate);
        add_gated(
            pass,
            rows,
            (&self.ff2, &buffers.gated),
            (&self.post_norm2, gate),
        );
    }
}

/// `normed` = `hidden`'s `rows` RMS-normalized by `norm`, then multiplied by `scale`; every
/// thread's rows, then a barrier.
fn scaled_norm(pass: &Pass, rows: &Range<usize>, norm: &Vector, scale: &[f32]) {
    let Pass {
        ctx,
        buffers,
        shape,
        ..
    } = pass;
    let hidden = shape.hidden;
    for row in ctx.split(rows.len(), 1) {
        let at = (rows.start + row) * hidden;
        let input = &buffers.hidden.read()[at..at + hidden];
        let output = buffers.normed.part(row * hidden..(row + 1) * hidden);
        rms_norm(input, norm.values(), shape.eps, output);
        for (value, &factor) in output.iter_mut().zip(scale) {
            *value *= factor;
        }
    }
    ctx.barrier();
}

/// Each head's attention output times the sigmoid of the head's gate logit; every thread's rows,
/// then a barrier.
fn gate_heads(pass: &Pass, len: usize) {
    let Pass {
        ctx,
        buffers,
        shape,
        ..
    } = pass;
    let (hidden, stride) = (shape.hidden, shape.qkv_gate());
    let qkv = buffers.qkv.read();
    for row in ctx.split(len, 1) {
        let logits = &qkv[row * stride + 3 * hidden..(row + 1) * stride];
        let output = buffers.attention.part(row * hidden..(row + 1) * hidden);
        for (head, &logit) in output.chunks_exact_mut(shape.head_dim()).zip(logits) {
            let gate = sigmoid(logit);
            for value in head {
                *value *= gate;
            }
        }
    }
    ctx.barrier();
}

/// `hidden`'s `rows` += `gate ⊙ post_norm(linear(input))`, `input` holding those rows from its
/// row 0.
fn add_gated(
    pass: &Pass,
    rows: &Range<usize>,
    (linear, input): (&Linear, &SharedBuf),
    (post_norm, gate): (&Vector, &[f32]),
) {
    let Pass {
        ctx,
        buffers,
        shape,
        ..
    } = pass;
    let hidden = shape.hidden;
    buffers
        .rows
        .project_synced(ctx, linear, (input, rows.len()), |row, col, value| {
            buffers.normed.set(row * hidden + col, value);
        });
    let scratch = &mut buffers.rows.scratch(ctx).row;
    scratch.resize(hidden, 0.0);
    for row in ctx.split(rows.len(), 1) {
        let projected = &buffers.normed.read()[row * hidden..(row + 1) * hidden];
        rms_norm(projected, post_norm.values(), shape.eps, scratch);
        let at = (rows.start + row) * hidden;
        let residual = buffers.hidden.part(at..at + hidden);
        for ((value, &factor), &normed) in residual.iter_mut().zip(gate).zip(scratch.iter()) {
            *value += factor * normed;
        }
    }
    ctx.barrier();
}

/// `p.{norm1, q_norm, k_norm, post_norm1, norm2, post_norm2}` vectors and `p.{qkv_gate, out, ff1,
/// ff2, modulation}` linears, the shape and the width of the feed-forward's gates.
impl Load for Block {
    type Shape = (Shape, usize);

    fn load(
        store: &mut Store,
        prefix: &str,
        (shape, feed_forward): (Shape, usize),
    ) -> Result<Self, bundle::Error> {
        let hidden = shape.hidden;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", hidden)?,
            qkv_gate: scope.load("qkv_gate", [shape.qkv_gate(), hidden])?,
            q_norm: scope.load("q_norm", hidden)?,
            k_norm: scope.load("k_norm", hidden)?,
            out: scope.load("out", [hidden, hidden])?,
            post_norm1: scope.load("post_norm1", hidden)?,
            norm2: scope.load("norm2", hidden)?,
            ff1: scope.load("ff1", [2 * feed_forward, hidden])?,
            ff2: scope.load("ff2", [hidden, feed_forward])?,
            post_norm2: scope.load("post_norm2", hidden)?,
            modulation: scope.load("modulation", [BLOCK_CHUNKS * hidden, shape.rank])?,
        })
    }
}
