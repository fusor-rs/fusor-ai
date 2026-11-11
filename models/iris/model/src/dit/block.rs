//! The trunk's blocks: a dual-stream block gives the text rows and the image rows their own
//! weights, a single-stream block gives every row the same; each is a modulated RMSNorm, joint
//! attention over every row (grouped-query, per-head RMSNorms of queries and keys, a sigmoid gate
//! on the output) and a SwiGLU MLP, each branch RMS-normalized before its gated residual.
//!
//! Every phase reads buffers written before the barrier that opens it and writes disjoint
//! ranges: a thread's rows, or the output columns of a product.

use std::ops::Range;

use blocks::attention::{Call, Operand, attend};
use bundle::{Scope, Store};
use kernels::ops::{Exponentials, rms_norm, sigmoid, silu};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use super::{
    DIT, HEAD_DIM, HEADS, Half, KV_HEADS, MODULATION, PAIRS, Pass, QKV, QUERIES, turn_heads,
};
use crate::Error;
use crate::model::{FEED_FORWARD, HIDDEN, NORM_EPS, TEXT_ROWS};

/// A half's modulation chunks: its shift, scale (stored as `1 +` it) and gate.
const SHIFT: usize = 0;
const SCALE: usize = 1;
const GATE: usize = 2;
const CHUNKS_PER_HALF: usize = 3;

impl Half {
    /// The index of this half's `part` (`SHIFT`, `SCALE` or `GATE`) among a stream's chunks.
    fn chunk(self, part: usize) -> usize {
        self as usize * CHUNKS_PER_HALF + part
    }
}

/// One stream's weights.
struct Stream {
    norm1: Vector,
    qkv: Linear,
    q_norm: Vector,
    k_norm: Vector,
    gate: Linear,
    out: Linear,
    post_attention: Vector,
    norm2: Vector,
    ff_gate: Linear,
    ff_up: Linear,
    ff_down: Linear,
    post_ff: Vector,
    /// The block's own bias on the shared core's modulation.
    bias: Vector,
}

/// A block's streams.
#[derive(Clone, Copy)]
pub(super) enum Streams {
    /// The text rows, then the image rows.
    Dual,
    /// Every row.
    Single,
}

impl Streams {
    fn names(self) -> &'static [&'static str] {
        match self {
            Streams::Dual => &["text", "image"],
            Streams::Single => &["shared"],
        }
    }
}

pub(super) struct Block {
    index: usize,
    streams: Vec<Stream>,
}

impl Stream {
    fn load(store: &mut Store, prefix: &str) -> Result<Self, Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm1: scope.load("norm1", HIDDEN)?,
            qkv: scope.load("qkv", [QKV, HIDDEN])?,
            q_norm: scope.load("q_norm", HEAD_DIM)?,
            k_norm: scope.load("k_norm", HEAD_DIM)?,
            gate: scope.load("gate", [HIDDEN; 2])?,
            out: scope.load("out", [HIDDEN; 2])?,
            post_attention: scope.load("post_attention", HIDDEN)?,
            norm2: scope.load("norm2", HIDDEN)?,
            ff_gate: scope.load("ff_gate", [FEED_FORWARD, HIDDEN])?,
            ff_up: scope.load("ff_up", [FEED_FORWARD, HIDDEN])?,
            ff_down: scope.load("ff_down", [HIDDEN, FEED_FORWARD])?,
            post_ff: scope.load("post_ff", HIDDEN)?,
            bias: scope.load("bias", MODULATION)?,
        })
    }

    fn norm(&self, half: Half) -> &Vector {
        match half {
            Half::Attention => &self.norm1,
            Half::FeedForward => &self.norm2,
        }
    }

    fn post_norm(&self, half: Half) -> &Vector {
        match half {
            Half::Attention => &self.post_attention,
            Half::FeedForward => &self.post_ff,
        }
    }
}

impl Block {
    pub(super) fn load(
        store: &mut Store,
        (index, streams): (usize, Streams),
    ) -> Result<Self, Error> {
        let streams = streams
            .names()
            .iter()
            .map(|name| Stream::load(store, &format!("{DIT}.blocks.{index}.{name}")))
            .collect::<Result<_, _>>()?;
        Ok(Self { index, streams })
    }

    /// This block's modulations of the timestep into `out`: each stream's core (`cores`: the
    /// image core's, then the text core's) plus the stream's bias, every scale as `1 +` it; this
    /// thread's columns, no barrier.
    pub(super) fn modulate(&self, ctx: &Ctx, cores: &[f32], out: &SharedBuf) {
        for (slot, stream) in self.streams.iter().enumerate() {
            let core = &cores[usize::from(self.is_text(slot)) * MODULATION..][..MODULATION];
            let first = (self.index * 2 + slot) * MODULATION;
            for column in ctx.split(MODULATION, 1) {
                let value = core[column] + stream.bias.values()[column];
                let is_scale = column / HIDDEN % CHUNKS_PER_HALF == SCALE;
                out.set(first + column, if is_scale { 1.0 + value } else { value });
            }
        }
    }

    /// Whether stream `slot` is the text stream of a dual block.
    fn is_text(&self, slot: usize) -> bool {
        self.streams.len() == 2 && slot == 0
    }

    /// Each stream's slot, rows and weights over a call of `rows` rows.
    fn spans(&self, rows: usize) -> impl Iterator<Item = (usize, Range<usize>, &Stream)> {
        let split = if self.streams.len() == 2 {
            TEXT_ROWS
        } else {
            rows
        };
        self.streams.iter().enumerate().map(move |(slot, stream)| {
            let range = if slot == 0 { 0..split } else { split..rows };
            (slot, range, stream)
        })
    }

    /// `visit(row, slot, stream)` for this thread's share of each stream's rows.
    fn each_row(&self, pass: &Pass, mut visit: impl FnMut(usize, usize, &Stream)) {
        for (slot, range, stream) in self.spans(pass.rows()) {
            for row in pass.ctx.split(range.len(), 1) {
                visit(range.start + row, slot, stream);
            }
        }
    }

    /// `part` of `half`'s modulation of stream `slot`.
    fn modulation<'pass>(
        &self,
        pass: &'pass Pass,
        (slot, half, part): (usize, Half, usize),
    ) -> &'pass [f32] {
        let at = (self.index * 2 + slot) * MODULATION + half.chunk(part) * HIDDEN;
        &pass.buffers.modulations.read()[at..at + HIDDEN]
    }

    /// `store(row, col, value)` of each stream's `linear` over its rows of `input`; then a
    /// barrier.
    fn project(
        &self,
        pass: &Pass,
        linear: impl Fn(&Stream) -> &Linear,
        input: &SharedBuf,
        mut store: impl FnMut(usize, usize, f32),
    ) {
        for (_, range, stream) in self.spans(pass.rows()) {
            pass.product(linear(stream), (input, range), &mut store);
        }
        pass.ctx.barrier();
    }

    pub(super) fn forward(&self, pass: &Pass) {
        let buffers = pass.buffers;
        self.normalize(pass, Half::Attention);
        let normed = &buffers.normed;
        self.project(
            pass,
            |stream| &stream.qkv,
            normed,
            |row, col, value| {
                buffers.qkv.set(row * QKV + col, value);
            },
        );
        self.project(
            pass,
            |stream| &stream.gate,
            normed,
            |row, col, value| {
                buffers.gate.set(row * HIDDEN + col, sigmoid(value));
            },
        );
        self.prepare_heads(pass);
        joint_attention(pass);
        gate_attention(pass);
        let attention = &buffers.attention;
        self.project(
            pass,
            |stream| &stream.out,
            attention,
            |row, col, value| {
                buffers.branch.set(row * HIDDEN + col, value);
            },
        );
        self.add_branch(pass, Half::Attention);
        self.normalize(pass, Half::FeedForward);
        self.project(
            pass,
            |stream| &stream.ff_gate,
            normed,
            |row, col, value| {
                buffers.inner.set(row * FEED_FORWARD + col, silu(value));
            },
        );
        self.project(
            pass,
            |stream| &stream.ff_up,
            normed,
            |row, col, value| {
                let at = row * FEED_FORWARD + col;
                buffers.inner.set(at, buffers.inner.get(at) * value);
            },
        );
        self.project(
            pass,
            |stream| &stream.ff_down,
            &buffers.inner,
            |row, col, value| {
                buffers.branch.set(row * HIDDEN + col, value);
            },
        );
        self.add_branch(pass, Half::FeedForward);
    }

    /// `normed` = `hidden` normalized by `half`'s norm, scaled and shifted by its modulation;
    /// then a barrier.
    fn normalize(&self, pass: &Pass, half: Half) {
        let buffers = pass.buffers;
        let hidden = buffers.hidden.read();
        self.each_row(pass, |row, slot, stream| {
            let range = row * HIDDEN..(row + 1) * HIDDEN;
            let normed = buffers.normed.part(range.clone());
            rms_norm(&hidden[range], stream.norm(half).values(), NORM_EPS, normed);
            let scale = self.modulation(pass, (slot, half, SCALE));
            let shift = self.modulation(pass, (slot, half, SHIFT));
            for ((value, scale), shift) in normed.iter_mut().zip(scale).zip(shift) {
                *value = *value * scale + shift;
            }
        });
        pass.ctx.barrier();
    }

    /// The query and key heads of every row's `qkv` RMS-normalized by its stream's weights and
    /// turned by its rotary angles; then a barrier.
    fn prepare_heads(&self, pass: &Pass) {
        let buffers = pass.buffers;
        let scratch = buffers.heads.get(pass.ctx);
        self.each_row(pass, |row, _, stream| {
            let start = row * QKV;
            let heads = buffers
                .qkv
                .part(start..start + QUERIES + KV_HEADS * HEAD_DIM);
            let (cos, sin, angle_row) = if row < TEXT_ROWS {
                (&pass.text_angles.cos[..], &pass.text_angles.sin[..], row)
            } else {
                let (cos, sin) = (buffers.image_cos.read(), buffers.image_sin.read());
                (cos, sin, row - TEXT_ROWS)
            };
            let angles = angle_row * PAIRS..(angle_row + 1) * PAIRS;
            let norms = (&stream.q_norm, &stream.k_norm);
            let angles = (&cos[angles.clone()], &sin[angles]);
            turn_heads(heads, QUERIES, norms, angles, scratch);
        });
        pass.ctx.barrier();
    }

    /// `hidden` += `half`'s gate times the RMSNorm of `branch`, through `normed`; then a
    /// barrier.
    fn add_branch(&self, pass: &Pass, half: Half) {
        let buffers = pass.buffers;
        let branch = buffers.branch.read();
        self.each_row(pass, |row, slot, stream| {
            let range = row * HIDDEN..(row + 1) * HIDDEN;
            let normed = buffers.normed.part(range.clone());
            rms_norm(
                &branch[range.clone()],
                stream.post_norm(half).values(),
                NORM_EPS,
                normed,
            );
            let gate = self.modulation(pass, (slot, half, GATE));
            let hidden = buffers.hidden.part(range);
            for ((value, gate), added) in hidden.iter_mut().zip(gate).zip(normed.iter()) {
                *value += gate * added;
            }
        });
        pass.ctx.barrier();
    }
}

/// `attention` = one attention of every row over all rows; then a barrier.
fn joint_attention(pass: &Pass) {
    let buffers = pass.buffers;
    let operand = |offset| Operand {
        values: buffers.qkv.read(),
        stride: QKV,
        offset,
    };
    let rows = pass.rows();
    let call = Call {
        heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        images: 1,
        queries: rows,
        keys: rows,
        query: operand(0),
        key: operand(QUERIES),
        value: operand(QUERIES + KV_HEADS * HEAD_DIM),
        exponentials: Exponentials::Exact,
    };
    let scratch = buffers.scratch.get(pass.ctx);
    attend(pass.ctx, &call, &buffers.attention, scratch);
    pass.ctx.barrier();
}

/// `attention` times the sigmoid gate; then a barrier.
fn gate_attention(pass: &Pass) {
    let buffers = pass.buffers;
    let gate = buffers.gate.read();
    for row in pass.ctx.split(pass.rows(), 1) {
        let range = row * HIDDEN..(row + 1) * HIDDEN;
        let attention = buffers.attention.part(range.clone());
        for (value, gate) in attention.iter_mut().zip(&gate[range]) {
            *value *= gate;
        }
    }
    pass.ctx.barrier();
}
