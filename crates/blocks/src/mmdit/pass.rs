//! The phases a dual-stream and a single-stream block share.
//!
//! Between two barriers each phase reads buffers written before the first and writes disjoint
//! ranges: a thread's rows, or the output columns of a product.

use std::ops::Range;

use kernels::ops::{Exponentials, layer_norm, rms_norm, rope_interleaved};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use super::{Buffers, Chunk, NORM_CHUNKS, Rows, Shape};
use crate::attention::{Call, Operand, attend};
use crate::rotary::{Position, Rotary};
use crate::row_work::RowScratch;

/// One modulated norm's shift, scale (as `1 +` it) and gate.
pub(super) struct Modulation<'values> {
    pub(super) shift: &'values [f32],
    pub(super) scale: &'values [f32],
    pub(super) gate: &'values [f32],
}

impl<'values> Modulation<'values> {
    /// Norm `norm`'s chunks of a block's modulations (`hidden` values each).
    pub(super) fn of(values: &'values [f32], hidden: usize, norm: usize) -> Self {
        let chunk = |chunk: Chunk| {
            let start = (norm * NORM_CHUNKS + chunk as usize) * hidden;
            &values[start..start + hidden]
        };
        Self {
            shift: chunk(Chunk::Shift),
            scale: chunk(Chunk::Scale),
            gate: chunk(Chunk::Gate),
        }
    }
}

/// The RMSNorm weights of a row's query and key heads.
#[derive(Clone, Copy)]
pub(super) struct HeadNorms<'norms> {
    pub(super) query: &'norms Vector,
    pub(super) key: &'norms Vector,
}

/// `out` = what `modulation` makes of `time` (a row of the width `hidden`), every scale as `1 +`
/// it; this thread's columns, no barrier.
pub(super) fn modulate(
    ctx: &Ctx,
    buffers: &Buffers,
    modulation: &Linear,
    (time, out): (&[f32], &SharedBuf),
) {
    let hidden = modulation.matrix().cols;
    buffers
        .rows
        .project(ctx, modulation, (time, 1), |_, col, value| {
            let is_scale = col / hidden % NORM_CHUNKS == Chunk::Scale as usize;
            out.set(col, if is_scale { 1.0 + value } else { value });
        });
}

/// `run`'s heads (rows of the weight's length) RMS-normalized in place, through `scratch`.
pub fn normalize_heads(run: &mut [f32], (weight, eps): (&Vector, f32), scratch: &mut Vec<f32>) {
    scratch.resize(run.len(), 0.0);
    rms_norm(run, weight.values(), eps, scratch);
    run.copy_from_slice(scratch);
}

/// What one call shares among its phases.
pub(super) struct Pass<'call> {
    pub(super) ctx: &'call Ctx<'call>,
    pub(super) buffers: &'call Buffers,
    pub(super) shape: &'call Shape,
    pub(super) rotary: &'call Rotary,
    pub(super) rows: Rows<'call>,
}

impl<'call> Pass<'call> {
    /// # Panics
    /// When the buffers hold fewer rows than `rows` has.
    pub(super) fn new(
        ctx: &'call Ctx<'call>,
        buffers: &'call Buffers,
        (shape, rotary): (&'call Shape, &'call Rotary),
        rows: Rows<'call>,
    ) -> Self {
        let pass = Self {
            ctx,
            buffers,
            shape,
            rotary,
            rows,
        };
        let capacity = buffers.capacity;
        assert!(
            pass.total() <= capacity.rows,
            "the buffers hold the call's rows"
        );
        assert!(
            pass.stream() <= capacity.stream,
            "the buffers hold the stream's rows"
        );
        assert!(
            rows.text <= rows.context.min(capacity.text),
            "the buffers hold the context's text rows"
        );
        pass
    }

    /// The stream's rows, the call's first.
    pub(super) fn stream(&self) -> usize {
        self.rows.positions.len()
    }

    pub(super) fn total(&self) -> usize {
        self.stream() + self.rows.context
    }

    /// `normed` = `hidden` normalized, then scaled and shifted by `stream`'s for the stream's
    /// rows and by `context`'s for the others; then a barrier.
    pub(super) fn modulated_norm(&self, stream: &Modulation, context: &Modulation) {
        let hidden = self.shape.hidden;
        for row in self.ctx.split(self.total(), 1) {
            let modulation = if row < self.stream() { stream } else { context };
            let range = row * hidden..(row + 1) * hidden;
            let input = &self.buffers.hidden.read()[range.clone()];
            let output = self.buffers.normed.part(range);
            layer_norm(
                input,
                modulation.scale,
                modulation.shift,
                self.shape.eps,
                output,
            );
        }
        self.ctx.barrier();
    }

    /// `store(row, col, value)` of `stream`'s weights over the stream's rows of `input` and of
    /// `context`'s over the other rows, counting rows from the call's first; no barrier.
    pub(super) fn product(
        &self,
        (stream, context): (&Linear, &Linear),
        input: &SharedBuf,
        mut store: impl FnMut(usize, usize, f32),
    ) {
        let (ctx, buffers) = (self.ctx, self.buffers);
        let values = input.read();
        buffers
            .rows
            .project(ctx, stream, (values, self.stream()), &mut store);
        let stream_values = self.stream() * stream.matrix().cols;
        let others = (&values[stream_values..], self.rows.context);
        buffers
            .rows
            .project(ctx, context, others, |row, col, value| {
                store(self.stream() + row, col, value);
            });
    }

    /// The query and key heads of every row's `qkv` RMS-normalized by its stream's weights, those
    /// of the stream's rows then turned by their positions; then a barrier.
    pub(super) fn head_norms(&self, stream: HeadNorms, context: HeadNorms) {
        let hidden = self.shape.hidden;
        let eps = self.shape.eps;
        let scratch = self.buffers.rows.scratch(self.ctx);
        for row in self.ctx.split(self.total(), 1) {
            let norms = if row < self.stream() { stream } else { context };
            let start = row * 3 * hidden;
            let queries = self.buffers.qkv.part(start..start + hidden);
            normalize_heads(queries, (norms.query, eps), &mut scratch.row);
            let keys = self.buffers.qkv.part(start + hidden..start + 2 * hidden);
            normalize_heads(keys, (norms.key, eps), &mut scratch.row);
            if let Some(&position) = self.rows.positions.get(row) {
                let both = self.buffers.qkv.part(start..start + 2 * hidden);
                self.rotate(both, position, scratch);
            }
        }
        self.ctx.barrier();
    }

    /// `heads` turned by the rotary angles of `position`.
    pub(super) fn rotate(&self, heads: &mut [f32], position: Position, scratch: &mut RowScratch) {
        let head_dim = self.shape.head_dim();
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        self.rotary
            .angles(position, &mut scratch.cos, &mut scratch.sin);
        for head in heads.chunks_exact_mut(head_dim) {
            rope_interleaved(head, &scratch.cos, &scratch.sin);
        }
    }

    /// `attention` = one attention of every row over all rows; then a barrier.
    pub(super) fn joint_attention(&self) {
        let hidden = self.shape.hidden;
        let operand = |offset| Operand {
            values: self.buffers.qkv.read(),
            stride: 3 * hidden,
            offset,
        };
        let call = Call {
            heads: self.shape.heads,
            kv_heads: self.shape.heads,
            head_dim: self.shape.head_dim(),
            images: 1,
            queries: self.total(),
            keys: self.total(),
            query: operand(0),
            key: operand(hidden),
            value: operand(2 * hidden),
            exponentials: Exponentials::Exact,
        };
        let scratch = self.buffers.scratch.get(self.ctx);
        attend(self.ctx, &call, &self.buffers.attention, scratch);
        self.ctx.barrier();
    }

    /// `hidden` rows `rows` += `gate ⊙ linear(input)` over the same rows of `input`; no barrier.
    pub(super) fn gated(
        &self,
        linear: &Linear,
        input: &SharedBuf,
        rows: Range<usize>,
        gate: &[f32],
    ) {
        let hidden = self.shape.hidden;
        let target = &self.buffers.hidden;
        let first = rows.start;
        let values = &input.read()[first * linear.matrix().cols..];
        self.buffers
            .rows
            .project(self.ctx, linear, (values, rows.len()), |row, col, value| {
                let at = (first + row) * hidden + col;
                target.set(at, target.get(at) + gate[col] * value);
            });
    }

    /// `gated` over the stream's rows by `stream`'s weights and gate and over the context's by
    /// `context`'s; then a barrier.
    pub(super) fn gated_streams(
        &self,
        (stream, context): (&Linear, &Linear),
        input: &SharedBuf,
        (stream_gate, context_gate): (&[f32], &[f32]),
    ) {
        self.gated(stream, input, 0..self.stream(), stream_gate);
        self.gated(context, input, self.stream()..self.total(), context_gate);
        self.ctx.barrier();
    }
}
