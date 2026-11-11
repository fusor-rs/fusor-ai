//! One block of the transformer: self-attention, cross-attention to the text, an MLP, each added
//! to the residual stream.
//!
//! Between two barriers each phase reads buffers written before the first and writes disjoint
//! ranges: a thread's tokens, or the output columns of a product.

use bundle::{Load, Scope, Store};
use kernels::ops::{gelu_tanh, rope_interleaved};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use super::{BLOCK_CHUNKS, Buffers, Chunk, Pass, Shape};
use crate::attention::{Call, Operand, attend};
use crate::norm::LayerNorm;
use crate::section;
use kernels::ops::Exponentials;

pub(super) struct Block {
    /// The block's own modulation parameters: `BLOCK_CHUNKS` chunks of `hidden`.
    pub(super) modulation: Vector,
    qkv: Linear,
    q_norm: Vector,
    k_norm: Vector,
    out: Linear,
    norm3: LayerNorm,
    cross_query: Linear,
    cross_q_norm: Vector,
    cross_k_norm: Vector,
    cross_context: Linear,
    cross_out: Linear,
    ff1: Linear,
    ff2: Linear,
}

/// Chunk `chunk` of block `index`'s modulations at the timestep last given to `Dit::time`.
fn modulation<'call>(pass: &'call Pass, index: usize, chunk: Chunk) -> &'call [f32] {
    let hidden = pass.shape.hidden;
    let block = &pass.buffers.modulations.read()[index * BLOCK_CHUNKS * hidden..];
    section(block, hidden, chunk as usize)
}

impl Block {
    /// The keys and values of `buffers.text_embedded` into `keys`.
    pub(super) fn condition(&self, ctx: &Ctx, buffers: &Buffers, shape: &Shape, keys: &SharedBuf) {
        let (width, rows) = (2 * shape.hidden, shape.text_tokens);
        let text = (&buffers.text_embedded, rows);
        buffers
            .rows
            .project_synced(ctx, &self.cross_context, text, |row, col, value| {
                keys.set(row * width + col, value);
            });
        let norm = (&self.cross_k_norm, shape.eps);
        buffers.rows.rms_rows(ctx, keys, (rows, width, 0), norm);
        ctx.barrier();
    }

    pub(super) fn forward(&self, pass: &Pass, index: usize, keys: &SharedBuf) {
        self.self_attention(pass, index);
        self.cross_attention(pass, keys);
        self.feed_forward(pass, index);
    }

    fn self_attention(&self, pass: &Pass, index: usize) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (hidden, tokens) = (shape.hidden, pass.tokens());
        let (shifts, scales) = (
            modulation(pass, index, Chunk::AttentionShift),
            modulation(pass, index, Chunk::AttentionScale),
        );
        pass.modulated_norm(shifts, scales);
        let normed = (&buffers.normed, tokens);
        buffers
            .rows
            .project_synced(ctx, &self.qkv, normed, |token, col, value| {
                buffers.qkv.set(token * 3 * hidden + col, value);
            });
        self.normalize_and_rotate(pass);
        ctx.barrier();
        let part = |offset| Operand {
            values: buffers.qkv.read(),
            stride: 3 * hidden,
            offset,
        };
        let call = Call {
            heads: shape.heads,
            kv_heads: shape.heads,
            head_dim: shape.head_dim(),
            images: 1,
            queries: tokens,
            keys: tokens,
            query: part(0),
            key: part(hidden),
            value: part(2 * hidden),
            exponentials: Exponentials::Exact,
        };
        attend(ctx, &call, &buffers.attention, buffers.scratch.get(ctx));
        ctx.barrier();
        let gate = modulation(pass, index, Chunk::AttentionGate);
        pass.residual(&self.out, &buffers.attention, Some(gate));
    }

    /// This thread's tokens' queries and keys RMS-normalized over their whole width, then turned
    /// by the rotary angles of the token's frame, row and column.
    fn normalize_and_rotate(&self, pass: &Pass) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (hidden, tokens, head_dim) = (shape.hidden, pass.tokens(), shape.head_dim());
        let stride = 3 * hidden;
        let q_norm = (&self.q_norm, shape.eps);
        let k_norm = (&self.k_norm, shape.eps);
        buffers
            .rows
            .rms_rows(ctx, &buffers.qkv, (tokens, stride, 0), q_norm);
        buffers
            .rows
            .rms_rows(ctx, &buffers.qkv, (tokens, stride, hidden), k_norm);
        let scratch = buffers.rows.scratch(ctx);
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        for token in ctx.split(tokens, 1) {
            let position = pass.patching.position(token);
            pass.rotary
                .angles(position, &mut scratch.cos, &mut scratch.sin);
            let queries_and_keys = buffers
                .qkv
                .part(token * stride..token * stride + 2 * hidden);
            for head in queries_and_keys.chunks_exact_mut(head_dim) {
                rope_interleaved(head, &scratch.cos, &scratch.sin);
            }
        }
    }

    fn cross_attention(&self, pass: &Pass, keys: &SharedBuf) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (hidden, tokens) = (shape.hidden, pass.tokens());
        for token in ctx.split(tokens, 1) {
            let row = token * hidden..(token + 1) * hidden;
            let input = &buffers.hidden.read()[row.clone()];
            self.norm3.apply(input, shape.eps, buffers.normed.part(row));
        }
        ctx.barrier();
        let normed = (&buffers.normed, tokens);
        buffers
            .rows
            .project_synced(ctx, &self.cross_query, normed, |token, col, value| {
                buffers.qkv.set(token * hidden + col, value);
            });
        let norm = (&self.cross_q_norm, shape.eps);
        buffers
            .rows
            .rms_rows(ctx, &buffers.qkv, (tokens, hidden, 0), norm);
        ctx.barrier();
        let context = |offset| Operand {
            values: keys.read(),
            stride: 2 * hidden,
            offset,
        };
        let call = Call {
            heads: shape.heads,
            kv_heads: shape.heads,
            head_dim: shape.head_dim(),
            images: 1,
            queries: tokens,
            keys: shape.text_tokens,
            query: Operand {
                values: buffers.qkv.read(),
                stride: hidden,
                offset: 0,
            },
            key: context(0),
            value: context(hidden),
            exponentials: Exponentials::Exact,
        };
        attend(ctx, &call, &buffers.attention, buffers.scratch.get(ctx));
        ctx.barrier();
        pass.residual(&self.cross_out, &buffers.attention, None);
    }

    fn feed_forward(&self, pass: &Pass, index: usize) {
        let Pass {
            ctx,
            buffers,
            shape,
            ..
        } = pass;
        let (shifts, scales) = (
            modulation(pass, index, Chunk::MlpShift),
            modulation(pass, index, Chunk::MlpScale),
        );
        pass.modulated_norm(shifts, scales);
        let width = shape.feed_forward;
        let normed = (&buffers.normed, pass.tokens());
        buffers
            .rows
            .project_synced(ctx, &self.ff1, normed, |token, col, value| {
                buffers.inner.set(token * width + col, gelu_tanh(value));
            });
        let gate = modulation(pass, index, Chunk::MlpGate);
        pass.residual(&self.ff2, &buffers.inner, Some(gate));
    }
}

impl Load for Block {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            feed_forward,
            ..
        } = shape;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            modulation: scope.load("modulation", BLOCK_CHUNKS * hidden)?,
            qkv: scope.load("qkv", [3 * hidden, hidden])?,
            q_norm: scope.load("q_norm", hidden)?,
            k_norm: scope.load("k_norm", hidden)?,
            out: scope.load("out", [hidden, hidden])?,
            norm3: scope.load("norm3", hidden)?,
            cross_query: scope.load("cross_query", [hidden, hidden])?,
            cross_q_norm: scope.load("cross_q_norm", hidden)?,
            cross_k_norm: scope.load("cross_k_norm", hidden)?,
            cross_context: scope.load("cross_context", [2 * hidden, hidden])?,
            cross_out: scope.load("cross_out", [hidden, hidden])?,
            ff1: scope.load("ff1", [feed_forward, hidden])?,
            ff2: scope.load("ff2", [hidden, feed_forward])?,
        })
    }
}
