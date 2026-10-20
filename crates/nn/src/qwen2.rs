//! The Qwen2 decoder stack: RMSNorm, fused QKV with biases, rotary embeddings by halves,
//! grouped-query attention over a KV cache (causal or sliding-window), SiLU-gated MLP.
//!
//! `forward` runs inside a team run: per layer five phases separated by barriers — QKV, RoPE
//! and cache writes by heads; attention by query heads; output projection plus residual by
//! columns; gate/up plus SiLU by intermediate columns; down projection plus residual by columns.

use kernels::ops::{attend, rms_norm, rope_angles, rope_half, rope_inv_freq, silu};
use kernels::{
    COLUMN_ALIGN, Error, Linear, QuantizedRows, SharedRows, Tensor, matmul, project_residual,
};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one Qwen2 stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_base: f32,
}

impl Shape {
    /// Columns of the fused QKV projection: query heads, then key heads, then value heads.
    fn qkv(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }
}

pub struct Layer {
    pub input_norm: Tensor,
    pub qkv: Linear,
    pub output: Linear,
    pub post_norm: Tensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct Qwen2 {
    shape: Shape,
    layers: Vec<Layer>,
    norm: Tensor,
    inv_freq: Vec<f32>,
}

/// Which keys a query at position `p` sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// Every position up to `p`.
    Causal,
    /// The last `n` positions up to and including `p`.
    Window(usize),
}

/// The KV cache: `capacity` slots per layer and KV head; position `p` lives in slot
/// `p % capacity`. A causal cache must hold the whole sequence; a sliding-window ring must be
/// larger than the window plus one call's rows, so no slot is overwritten while still visible.
pub struct KvCache {
    shape: Shape,
    capacity: usize,
    keys: SharedBuf,
    values: SharedBuf,
}

impl KvCache {
    pub fn new(shape: Shape, capacity: usize) -> Self {
        let len = shape.layers * shape.kv_heads * capacity * shape.head_dim;
        Self {
            shape,
            capacity,
            keys: SharedBuf::new(len),
            values: SharedBuf::new(len),
        }
    }

    fn offset(&self, layer: usize, kv_head: usize, position: usize) -> usize {
        let slot = position % self.capacity;
        ((layer * self.shape.kv_heads + kv_head) * self.capacity + slot) * self.shape.head_dim
    }
}

/// Activations of one forward call of up to `rows` positions.
pub struct Buffers {
    rows: usize,
    /// The residual stream, `rows × hidden`: the input of `forward` and its output.
    pub hidden: SharedBuf,
    qkv: SharedBuf,
    attention: SharedBuf,
    mlp: SharedBuf,
    up: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    normed: Vec<f32>,
    quant: QuantizedRows,
    scores: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Buffers {
    pub fn new(shape: Shape, rows: usize, threads: usize) -> Self {
        Self {
            rows,
            hidden: SharedBuf::new(rows * shape.hidden),
            qkv: SharedBuf::new(rows * shape.qkv()),
            attention: SharedBuf::new(rows * shape.hidden),
            mlp: SharedBuf::new(rows * shape.intermediate),
            up: SharedBuf::new(rows * shape.intermediate),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }
}

/// One call: `rows` consecutive positions starting at `first`, attending under `mask`.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub rows: usize,
    pub first: usize,
    pub mask: Mask,
}

/// Where a QKV head goes: its layer, its unit (query heads, then key heads, then value heads)
/// and its sequence position.
#[derive(Clone, Copy)]
struct HeadAt {
    layer: usize,
    unit: usize,
    position: usize,
}

/// Everything one thread's forward pass reads.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    cache: &'call KvCache,
    buffers: &'call Buffers,
    span: Span,
}

impl Qwen2 {
    pub fn new(shape: Shape, layers: Vec<Layer>, norm: Tensor) -> Result<Self, Error> {
        if layers.len() != shape.layers {
            return Err(Error::Format(format!(
                "{} layers for a {}-layer shape",
                layers.len(),
                shape.layers
            )));
        }
        Ok(Self {
            shape,
            layers,
            norm,
            inv_freq: rope_inv_freq(shape.rope_base, shape.head_dim),
        })
    }

    /// Runs every layer over `buffers.hidden[..span.rows]` (the residual stream, in and out).
    pub fn forward(&self, ctx: &Ctx, cache: &KvCache, buffers: &Buffers, span: Span) {
        assert!(span.rows <= buffers.rows, "the buffers hold the span");
        let pass = Pass {
            ctx,
            cache,
            buffers,
            span,
        };
        let quant = || &mut buffers.scratch.get(ctx).quant;
        let rows = |values| SharedRows {
            values,
            rows: span.rows,
        };
        for (index, layer) in self.layers.iter().enumerate() {
            self.attention_input(&pass, index, layer);
            ctx.barrier();
            self.attention(&pass, index);
            ctx.barrier();
            project_residual(
                ctx,
                &layer.output,
                rows(&buffers.attention),
                &buffers.hidden,
                quant(),
            );
            ctx.barrier();
            self.gated_mlp(&pass, layer);
            ctx.barrier();
            project_residual(
                ctx,
                &layer.down,
                rows(&buffers.mlp),
                &buffers.hidden,
                quant(),
            );
            ctx.barrier();
        }
    }

    /// The final norm of hidden row `row` (after `forward`, outside or inside a team run).
    pub fn final_norm(&self, buffers: &Buffers, row: usize, out: &mut [f32]) {
        let hidden = self.shape.hidden;
        let x = &buffers.hidden.read()[row * hidden..(row + 1) * hidden];
        rms_norm(x, self.norm.values().expect("norm weights are f32"), out);
    }

    fn normalize(&self, x: &SharedBuf, rows: usize, weight: &Tensor, out: &mut Vec<f32>) {
        let len = rows * self.shape.hidden;
        out.resize(len, 0.0);
        rms_norm(
            &x.read()[..len],
            weight.values().expect("norm weights are f32"),
            out,
        );
    }

    /// QKV for this thread's heads, RoPE on its query and key heads, its key and value heads
    /// into the cache.
    fn attention_input(&self, pass: &Pass, index: usize, layer: &Layer) {
        let Shape { head_dim, .. } = self.shape;
        let (buffers, rows) = (pass.buffers, pass.span.rows);
        let qkv_width = self.shape.qkv();
        let scratch = buffers.scratch.get(pass.ctx);
        self.normalize(
            &buffers.hidden,
            rows,
            &layer.input_norm,
            &mut scratch.normed,
        );
        let matrix = layer.qkv.matrix();
        let input = scratch.quant.input(&matrix, &scratch.normed, rows);
        let heads = pass.ctx.split(qkv_width, head_dim);
        let bias = layer.qkv.bias().expect("qkv has a bias");
        matmul(&matrix, &input, rows, heads.clone(), |row, col, value| {
            buffers.qkv.set(row * qkv_width + col, value + bias[col]);
        });
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        for row in 0..rows {
            let position = pass.span.first + row;
            rope_angles(position, &self.inv_freq, &mut scratch.cos, &mut scratch.sin);
            for start in heads.clone().step_by(head_dim) {
                let at = row * qkv_width + start;
                let head_at = HeadAt {
                    layer: index,
                    unit: start / head_dim,
                    position,
                };
                self.place_head(pass, head_at, buffers.qkv.part(at..at + head_dim), scratch);
            }
        }
    }

    /// Rotates a query or key head and stores key and value heads in the cache.
    fn place_head(&self, pass: &Pass, head_at: HeadAt, head: &mut [f32], scratch: &Scratch) {
        let Shape {
            heads, kv_heads, ..
        } = self.shape;
        let HeadAt {
            layer,
            unit,
            position,
        } = head_at;
        if unit < heads + kv_heads {
            rope_half(head, &scratch.cos, &scratch.sin);
        }
        if unit < heads {
            return;
        }
        let cache = pass.cache;
        let target = if unit < heads + kv_heads {
            &cache.keys
        } else {
            &cache.values
        };
        let at = cache.offset(layer, (unit - heads) % kv_heads, position);
        target.part(at..at + head.len()).copy_from_slice(head);
    }

    /// Attention for this thread's query heads, every row of the span.
    fn attention(&self, pass: &Pass, layer: usize) {
        let Shape {
            heads,
            kv_heads,
            head_dim,
            hidden,
            ..
        } = self.shape;
        let (cache, buffers, span) = (pass.cache, pass.buffers, pass.span);
        let qkv_width = self.shape.qkv();
        let scratch = buffers.scratch.get(pass.ctx);
        let qkv = buffers.qkv.read();
        let (keys, values) = (cache.keys.read(), cache.values.read());
        for head in pass.ctx.split(heads, 1) {
            let kv_head = head / (heads / kv_heads);
            for row in 0..span.rows {
                let position = span.first + row;
                let first_key = match span.mask {
                    Mask::Causal => 0,
                    Mask::Window(window) => (position + 1).saturating_sub(window),
                };
                let query = &qkv[row * qkv_width + head * head_dim..][..head_dim];
                let slot = |j: usize| cache.offset(layer, kv_head, first_key + j);
                let at = row * hidden + head * head_dim;
                attend(
                    query,
                    position + 1 - first_key,
                    |j| {
                        let at = slot(j);
                        (&keys[at..at + head_dim], &values[at..at + head_dim])
                    },
                    &mut scratch.scores,
                    buffers.attention.part(at..at + head_dim),
                );
            }
        }
    }

    /// `mlp[:, cols] = silu(gate) · up` for this thread's intermediate columns.
    fn gated_mlp(&self, pass: &Pass, layer: &Layer) {
        let intermediate = self.shape.intermediate;
        let (buffers, rows) = (pass.buffers, pass.span.rows);
        let scratch = buffers.scratch.get(pass.ctx);
        self.normalize(&buffers.hidden, rows, &layer.post_norm, &mut scratch.normed);
        let (gate, up) = (layer.gate.matrix(), layer.up.matrix());
        let input = scratch.quant.input(&gate, &scratch.normed, rows);
        let cols = pass.ctx.split(intermediate, COLUMN_ALIGN);
        matmul(&gate, &input, rows, cols.clone(), |row, col, value| {
            buffers.mlp.set(row * intermediate + col, value);
        });
        matmul(&up, &input, rows, cols.clone(), |row, col, value| {
            buffers.up.set(row * intermediate + col, value);
        });
        for row in 0..rows {
            let range = row * intermediate + cols.start..row * intermediate + cols.end;
            let up_values = buffers.up.part(range.clone());
            for (gated, up_value) in buffers.mlp.part(range).iter_mut().zip(up_values.iter()) {
                *gated = silu(*gated) * up_value;
            }
        }
    }
}
