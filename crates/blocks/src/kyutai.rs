//! Kyutai's streaming transformer (Moshi, Mimi, Pocket TTS, Phonon): pre-LayerNorm layers with
//! a bias-free fused QKV, interleaved rotary embeddings, multi-head attention over a KV cache
//! (causal or sliding-window) and an exact-GELU MLP without biases. A checkpoint's layer-scale
//! factors are folded into `out` and `ff2` by its compiler.
//!
//! `forward` runs inside a team run: per layer five phases separated by barriers — QKV, RoPE
//! and cache writes by heads; attention by heads; output projection plus residual by columns;
//! the first MLP projection plus GELU by columns; the second plus residual by columns.
//!
//! Tensors under a prefix `p`: `p.layers.{i}.{norm1, qkv, out, norm2, ff1, ff2}`, the norms as
//! `weight` and `bias` vectors.

use bundle::{Load, Store};
use kernels::ops::{attend, gelu_erf, layer_norm, rope_angles, rope_interleaved, rope_inv_freq};
use kernels::{
    COLUMN_ALIGN, Error, Linear, QuantizedRows, SharedRows, Tensor, matmul, project_residual,
};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

/// A LayerNorm's affine parameters.
pub struct Norm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct Layer {
    pub norm1: Norm,
    pub qkv: Linear,
    pub out: Linear,
    pub norm2: Norm,
    pub ff1: Linear,
    pub ff2: Linear,
}

pub struct Transformer {
    shape: Shape,
    layers: Vec<Layer>,
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

/// The KV cache: `capacity` slots per layer and head; position `p` lives in slot
/// `p % capacity`. A causal cache must hold the whole sequence; a sliding-window ring must hold
/// the window plus one call's rows, so no slot is overwritten while still visible.
pub struct KvCache {
    shape: Shape,
    capacity: usize,
    keys: SharedBuf,
    values: SharedBuf,
}

impl KvCache {
    pub fn new(shape: Shape, capacity: usize) -> Self {
        let len = shape.layers * shape.heads * capacity * shape.head_dim;
        Self {
            shape,
            capacity,
            keys: SharedBuf::new(len),
            values: SharedBuf::new(len),
        }
    }

    fn offset(&self, layer: usize, head: usize, position: usize) -> usize {
        let slot = position % self.capacity;
        ((layer * self.shape.heads + head) * self.capacity + slot) * self.shape.head_dim
    }
}

/// Activations of one forward call of up to `rows` positions.
pub struct Buffers {
    rows: usize,
    /// The residual stream, `rows × hidden`: the input of `forward` and its output.
    pub hidden: SharedBuf,
    qkv: SharedBuf,
    attention: SharedBuf,
    feed_forward: SharedBuf,
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
            qkv: SharedBuf::new(rows * 3 * shape.hidden),
            attention: SharedBuf::new(rows * shape.hidden),
            feed_forward: SharedBuf::new(rows * shape.feed_forward),
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

impl Transformer {
    pub fn new(shape: Shape, layers: Vec<Layer>) -> Result<Self, Error> {
        if layers.len() != shape.layers || shape.heads * shape.head_dim != shape.hidden {
            return Err(Error::Format(format!(
                "{} layers for a shape of {} layers × {} heads of {} = {}",
                layers.len(),
                shape.layers,
                shape.heads,
                shape.head_dim,
                shape.hidden
            )));
        }
        Ok(Self {
            shape,
            layers,
            inv_freq: rope_inv_freq(shape.rope_base, shape.head_dim),
        })
    }

    /// Runs every layer over `buffers.hidden[..span.rows]` (the residual stream, in and out).
    ///
    /// # Panics
    /// When the cache would overwrite a key the span still sees: a causal span past the cache's
    /// capacity, or a window plus the span's rows over it.
    pub fn forward(&self, ctx: &Ctx, cache: &KvCache, buffers: &Buffers, span: Span) {
        assert!(span.rows <= buffers.rows, "the buffers hold the span");
        let reach = match span.mask {
            Mask::Causal => span.first + span.rows,
            Mask::Window(window) => window + span.rows,
        };
        assert!(
            reach <= cache.capacity,
            "the cache holds every key the span sees"
        );
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
            let attention = rows(&buffers.attention);
            project_residual(ctx, &layer.out, attention, &buffers.hidden, quant());
            ctx.barrier();
            self.feed_forward(&pass, layer);
            ctx.barrier();
            let feed_forward = rows(&buffers.feed_forward);
            project_residual(ctx, &layer.ff2, feed_forward, &buffers.hidden, quant());
            ctx.barrier();
        }
    }

    /// `norm(hidden)` of the span's rows into this thread's scratch.
    fn normalize(&self, pass: &Pass, norm: &Norm, out: &mut Vec<f32>) {
        let len = pass.span.rows * self.shape.hidden;
        out.resize(len, 0.0);
        norm.apply(&pass.buffers.hidden.read()[..len], self.shape.norm_eps, out);
    }

    /// QKV for this thread's heads, RoPE on its query and key heads, its key and value heads
    /// into the cache.
    fn attention_input(&self, pass: &Pass, layer: usize, weights: &Layer) {
        let Shape {
            head_dim, hidden, ..
        } = self.shape;
        let (buffers, rows) = (pass.buffers, pass.span.rows);
        let width = 3 * hidden;
        let scratch = buffers.scratch.get(pass.ctx);
        self.normalize(pass, &weights.norm1, &mut scratch.normed);
        let matrix = weights.qkv.matrix();
        let input = scratch.quant.input(&matrix, &scratch.normed, rows);
        let units = pass.ctx.split(width, head_dim);
        matmul(&matrix, &input, rows, units.clone(), |row, col, value| {
            buffers.qkv.set(row * width + col, value);
        });
        scratch.cos.resize(head_dim / 2, 0.0);
        scratch.sin.resize(head_dim / 2, 0.0);
        for row in 0..rows {
            let position = pass.span.first + row;
            rope_angles(position, &self.inv_freq, &mut scratch.cos, &mut scratch.sin);
            for start in units.clone().step_by(head_dim) {
                let at = row * width + start;
                let head_at = HeadAt {
                    layer,
                    unit: start / head_dim,
                    position,
                };
                self.place_head(pass, head_at, buffers.qkv.part(at..at + head_dim), scratch);
            }
        }
    }

    /// Rotates a query or key head and stores key and value heads in the cache.
    fn place_head(&self, pass: &Pass, head_at: HeadAt, head: &mut [f32], scratch: &Scratch) {
        let heads = self.shape.heads;
        let HeadAt {
            layer,
            unit,
            position,
        } = head_at;
        if unit < 2 * heads {
            rope_interleaved(head, &scratch.cos, &scratch.sin);
        }
        if unit < heads {
            return;
        }
        let cache = pass.cache;
        let target = if unit < 2 * heads {
            &cache.keys
        } else {
            &cache.values
        };
        let slot = cache.offset(layer, unit % heads, position);
        target.part(slot..slot + head.len()).copy_from_slice(head);
    }

    /// Attention for this thread's heads, every row of the span.
    fn attention(&self, pass: &Pass, layer: usize) {
        let Shape {
            heads,
            head_dim,
            hidden,
            ..
        } = self.shape;
        let (cache, buffers, span) = (pass.cache, pass.buffers, pass.span);
        let scratch = buffers.scratch.get(pass.ctx);
        let qkv = buffers.qkv.read();
        let (keys, values) = (cache.keys.read(), cache.values.read());
        for head in pass.ctx.split(heads, 1) {
            for row in 0..span.rows {
                let position = span.first + row;
                let first_key = match span.mask {
                    Mask::Causal => 0,
                    Mask::Window(window) => (position + 1).saturating_sub(window),
                };
                let query = &qkv[row * 3 * hidden + head * head_dim..][..head_dim];
                let at = row * hidden + head * head_dim;
                attend(
                    query,
                    position + 1 - first_key,
                    |j| {
                        let slot = cache.offset(layer, head, first_key + j);
                        (&keys[slot..slot + head_dim], &values[slot..slot + head_dim])
                    },
                    &mut scratch.scores,
                    buffers.attention.part(at..at + head_dim),
                );
            }
        }
    }

    /// `feed_forward[:, cols] = gelu(norm2(hidden) · ff1ᵀ)` for this thread's columns.
    fn feed_forward(&self, pass: &Pass, weights: &Layer) {
        let width = self.shape.feed_forward;
        let (buffers, rows) = (pass.buffers, pass.span.rows);
        let scratch = buffers.scratch.get(pass.ctx);
        self.normalize(pass, &weights.norm2, &mut scratch.normed);
        let matrix = weights.ff1.matrix();
        let input = scratch.quant.input(&matrix, &scratch.normed, rows);
        let cols = pass.ctx.split(width, COLUMN_ALIGN);
        matmul(&matrix, &input, rows, cols, |row, col, value| {
            buffers.feed_forward.set(row * width + col, gelu_erf(value));
        });
    }
}

impl Norm {
    /// The LayerNorm of every row of `x` (rows of the norm's width).
    pub fn apply(&self, x: &[f32], eps: f32, out: &mut [f32]) {
        let weight = self.weight.values().expect("checked at load");
        let bias = self.bias.values().expect("checked at load");
        layer_norm(x, weight, bias, eps, out);
    }
}

/// `p.weight` and `p.bias`, both f32 vectors.
impl Load for Norm {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let weight = Tensor::load(store, &format!("{prefix}.weight"), ())?;
        let bias = Tensor::load(store, &format!("{prefix}.bias"), ())?;
        if weight.values()?.len() != bias.values()?.len() {
            return Err(bundle::Error::Format(format!(
                "{prefix}: weight and bias differ in length"
            )));
        }
        Ok(Self { weight, bias })
    }
}

impl Load for Layer {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            norm1: Norm::load(store, &name("norm1"), ())?,
            qkv: Linear::load(store, &name("qkv"), ())?,
            out: Linear::load(store, &name("out"), ())?,
            norm2: Norm::load(store, &name("norm2"), ())?,
            ff1: Linear::load(store, &name("ff1"), ())?,
            ff2: Linear::load(store, &name("ff2"), ())?,
        })
    }
}

impl Load for Transformer {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let layers = (0..shape.layers)
            .map(|i| Layer::load(store, &format!("{prefix}.layers.{i}"), ()))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(shape, layers)?)
    }
}
