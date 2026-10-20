//! DiT blocks: pre-RMSNorm transformer blocks with non-causal attention inside short sequences,
//! interleaved rotary embeddings (x-transformers) at positions `0..len`, and a GELU-tanh MLP.
//!
//! Tensors under a prefix `p`: `p.blocks.{i}.{norm1, qkv, out, norm2, ff1, ff2}` (q, k and v fused
//! into `qkv`, every linear with a bias), and for `FinalLayer`, `p.final_norm` and `p.final`.
//!
//! `forward` runs inside a team run over `Sequences`: consecutive sequences of equal length.

use bundle::{Load, Store};
use kernels::ops::{attend, gelu_tanh, rms_norm, rope_angles, rope_interleaved, rope_inv_freq};
use kernels::{
    COLUMN_ALIGN, Error, Linear, NARROW_COLUMN_ALIGN, QuantizedRows, SharedRows, Tensor, matmul,
    project_residual,
};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one DiT stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub depth: usize,
    pub rope_base: f32,
    /// The longest sequence the stack attends over.
    pub max_sequence: usize,
}

impl Shape {
    fn qkv(&self) -> usize {
        3 * self.hidden
    }
}

pub struct Block {
    pub norm1: Tensor,
    pub qkv: Linear,
    pub out: Linear,
    pub norm2: Tensor,
    pub ff1: Linear,
    pub ff2: Linear,
}

/// `rows` rows of `Buffers::hidden`, forming consecutive sequences of `length` tokens that
/// attend within themselves.
#[derive(Debug, Clone, Copy)]
pub struct Sequences {
    pub rows: usize,
    pub length: usize,
}

/// The output projection after the last block: RMSNorm, then a linear layer.
pub struct FinalLayer {
    pub norm: Tensor,
    pub linear: Linear,
}

pub struct Dit {
    shape: Shape,
    blocks: Vec<Block>,
    /// cos/sin of position `p`, pair `i`: `[p · head_dim / 2 + i]`.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

/// Activations of DiT passes of up to `rows` rows.
pub struct Buffers {
    hidden_width: usize,
    /// The residual stream, `rows × hidden`: callers fill the input rows, `forward` updates them.
    pub hidden: SharedBuf,
    qkv: SharedBuf,
    attention: SharedBuf,
    feed_forward: SharedBuf,
    /// `FinalLayer` outputs, one row per projected row.
    pub out: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    normed: Vec<f32>,
    quant: QuantizedRows,
    scores: Vec<f32>,
}

impl Buffers {
    /// Room for `rows` rows and `rows` final-layer outputs of up to `outputs` values.
    pub fn new(shape: Shape, rows: usize, outputs: usize, threads: usize) -> Self {
        Self {
            hidden_width: shape.hidden,
            hidden: SharedBuf::new(rows * shape.hidden),
            qkv: SharedBuf::new(rows * shape.qkv()),
            attention: SharedBuf::new(rows * shape.hidden),
            feed_forward: SharedBuf::new(rows * shape.feed_forward),
            out: SharedBuf::new(rows * outputs),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    pub fn output(&self, len: usize) -> &[f32] {
        &self.out.read()[..len]
    }

    /// `hidden[row, cols] = x · Wᵀ + b` for this thread's columns (an input embedding).
    pub fn embed_row(&self, ctx: &Ctx, linear: &Linear, x: &[f32], row: usize) {
        let matrix = linear.matrix();
        let bias = linear.bias().expect("an embedding has a bias");
        let quant = &mut self.scratch.get(ctx).quant;
        let input = quant.input(&matrix, x, 1);
        let width = self.hidden_width;
        matmul(
            &matrix,
            &input,
            1,
            ctx.split(width, COLUMN_ALIGN),
            |_, col, value| {
                self.hidden.set(row * width + col, value + bias[col]);
            },
        );
    }
}

impl Dit {
    pub fn new(shape: Shape, blocks: Vec<Block>) -> Result<Self, Error> {
        if blocks.len() != shape.depth {
            return Err(Error::Format(format!(
                "{} blocks for a {}-block shape",
                blocks.len(),
                shape.depth
            )));
        }
        let inv_freq = rope_inv_freq(shape.rope_base, shape.head_dim);
        let half = shape.head_dim / 2;
        let mut rope_cos = vec![0.0; shape.max_sequence * half];
        let mut rope_sin = vec![0.0; shape.max_sequence * half];
        let tables = rope_cos
            .chunks_exact_mut(half)
            .zip(rope_sin.chunks_exact_mut(half));
        for (position, (cos, sin)) in tables.enumerate() {
            rope_angles(position, &inv_freq, cos, sin);
        }
        Ok(Self {
            shape,
            blocks,
            rope_cos,
            rope_sin,
        })
    }

    /// All blocks over the rows of `buffers.hidden` that `sequences` describes.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, sequences: Sequences) {
        assert!(
            sequences.length <= self.shape.max_sequence,
            "the rotary table covers the sequence"
        );
        let rows = sequences.rows;
        let quant = || &mut buffers.scratch.get(ctx).quant;
        let shared = |values| SharedRows { values, rows };
        for block in &self.blocks {
            self.attention_input(ctx, block, buffers, sequences);
            ctx.barrier();
            self.attention(ctx, buffers, sequences);
            ctx.barrier();
            project_residual(
                ctx,
                &block.out,
                shared(&buffers.attention),
                &buffers.hidden,
                quant(),
            );
            ctx.barrier();
            self.feed_forward(ctx, block, buffers, rows);
            ctx.barrier();
            project_residual(
                ctx,
                &block.ff2,
                shared(&buffers.feed_forward),
                &buffers.hidden,
                quant(),
            );
            ctx.barrier();
        }
    }

    /// `out[i, cols] = final(norm(hidden[rows[i]]))` for this thread's output columns.
    pub fn project(&self, ctx: &Ctx, buffers: &Buffers, layer: &FinalLayer, rows: &[usize]) {
        let hidden = self.shape.hidden;
        let scratch = buffers.scratch.get(ctx);
        let values = buffers.hidden.read();
        let norm = layer.norm.values().expect("norm weights are f32");
        scratch.normed.resize(rows.len() * hidden, 0.0);
        for (out, &row) in scratch.normed.chunks_exact_mut(hidden).zip(rows) {
            rms_norm(&values[row * hidden..(row + 1) * hidden], norm, out);
        }
        let matrix = layer.linear.matrix();
        let outputs = matrix.rows;
        let input = scratch.quant.input(&matrix, &scratch.normed, rows.len());
        let bias = layer.linear.bias().expect("the final layer has a bias");
        let cols = ctx.split(outputs, NARROW_COLUMN_ALIGN);
        matmul(&matrix, &input, rows.len(), cols, |row, col, value| {
            buffers.out.set(row * outputs + col, value + bias[col]);
        });
    }

    fn normalize(&self, buffers: &Buffers, rows: usize, weight: &Tensor, out: &mut Vec<f32>) {
        let len = rows * self.shape.hidden;
        out.resize(len, 0.0);
        let weight = weight.values().expect("norm weights are f32");
        rms_norm(&buffers.hidden.read()[..len], weight, out);
    }

    /// QKV for this thread's heads, then RoPE on its query and key heads.
    fn attention_input(&self, ctx: &Ctx, block: &Block, buffers: &Buffers, sequences: Sequences) {
        let Sequences { rows, length: seq } = sequences;
        let Shape {
            head_dim, hidden, ..
        } = self.shape;
        let qkv_width = self.shape.qkv();
        let scratch = buffers.scratch.get(ctx);
        self.normalize(buffers, rows, &block.norm1, &mut scratch.normed);
        let matrix = block.qkv.matrix();
        let input = scratch.quant.input(&matrix, &scratch.normed, rows);
        let heads = ctx.split(qkv_width, head_dim);
        let bias = block.qkv.bias().expect("qkv has a bias");
        matmul(&matrix, &input, rows, heads.clone(), |row, col, value| {
            buffers.qkv.set(row * qkv_width + col, value + bias[col]);
        });
        let half = head_dim / 2;
        for row in 0..rows {
            let position = row % seq;
            let cos = &self.rope_cos[position * half..(position + 1) * half];
            let sin = &self.rope_sin[position * half..(position + 1) * half];
            for start in heads
                .clone()
                .step_by(head_dim)
                .filter(|&start| start < 2 * hidden)
            {
                let at = row * qkv_width + start;
                rope_interleaved(buffers.qkv.part(at..at + head_dim), cos, sin);
            }
        }
    }

    /// Non-causal attention inside each sequence, split over (sequence, head) pairs.
    fn attention(&self, ctx: &Ctx, buffers: &Buffers, sequences: Sequences) {
        let Sequences { rows, length: seq } = sequences;
        let Shape {
            heads,
            head_dim,
            hidden,
            ..
        } = self.shape;
        let qkv_width = self.shape.qkv();
        let scratch = buffers.scratch.get(ctx);
        let qkv = buffers.qkv.read();
        for unit in ctx.split(rows / seq * heads, 1) {
            let (sequence, head) = (unit / heads, unit % heads);
            let first = sequence * seq;
            let at = |row: usize, part: usize| {
                (first + row) * qkv_width + part * hidden + head * head_dim
            };
            for row in 0..seq {
                let out = (first + row) * hidden + head * head_dim;
                attend(
                    &qkv[at(row, 0)..at(row, 0) + head_dim],
                    seq,
                    |j| {
                        (
                            &qkv[at(j, 1)..at(j, 1) + head_dim],
                            &qkv[at(j, 2)..at(j, 2) + head_dim],
                        )
                    },
                    &mut scratch.scores,
                    buffers.attention.part(out..out + head_dim),
                );
            }
        }
    }

    /// `feed_forward[:, cols] = gelu(norm2(hidden) · W1ᵀ + b1)` for this thread's columns.
    fn feed_forward(&self, ctx: &Ctx, block: &Block, buffers: &Buffers, rows: usize) {
        let width = self.shape.feed_forward;
        let scratch = buffers.scratch.get(ctx);
        self.normalize(buffers, rows, &block.norm2, &mut scratch.normed);
        let matrix = block.ff1.matrix();
        let input = scratch.quant.input(&matrix, &scratch.normed, rows);
        let bias = block.ff1.bias().expect("ff1 has a bias");
        let cols = ctx.split(width, COLUMN_ALIGN);
        matmul(&matrix, &input, rows, cols.clone(), |row, col, value| {
            buffers
                .feed_forward
                .set(row * width + col, value + bias[col]);
        });
        for row in 0..rows {
            let values = buffers
                .feed_forward
                .part(row * width + cols.start..row * width + cols.end);
            values
                .iter_mut()
                .for_each(|value| *value = gelu_tanh(*value));
        }
    }
}

impl Load for Block {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm1: Tensor::load(store, &name("norm1"), ())?,
            qkv: Linear::load(store, &name("qkv"), ())?,
            out: Linear::load(store, &name("out"), ())?,
            norm2: Tensor::load(store, &name("norm2"), ())?,
            ff1: Linear::load(store, &name("ff1"), ())?,
            ff2: Linear::load(store, &name("ff2"), ())?,
        })
    }
}

impl Load for Dit {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let blocks = (0..shape.depth)
            .map(|i| Block::load(store, &format!("{prefix}.blocks.{i}"), ()))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(shape, blocks)?)
    }
}

impl Load for FinalLayer {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        Ok(Self {
            norm: Tensor::load(store, &format!("{prefix}.final_norm"), ())?,
            linear: Linear::load(store, &format!("{prefix}.final"), ())?,
        })
    }
}
