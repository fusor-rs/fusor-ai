//! umT5's encoder (umT5-XXL in Wan2.1 and Wan2.2): token embeddings, pre-RMSNorm layers
//! (`layer`, `LayerSpec::UMT5`) whose attention adds each layer's own relative position bias to
//! the unscaled scores, and a final RMSNorm. A key `d` positions after its query falls in one of
//! `buckets` buckets: half for keys after, half for the others; exact distances below a quarter of
//! the buckets, then logarithmic up to `max_distance`. Every row attends to every other.
//!
//! Tensors under a prefix `p`: `p.embedding` (`[vocabulary, hidden]`), the stack's
//! `p.layers.{i}.{norm1, qkv, o, norm2, gate, up, down}`, each layer's bias `p.layers.{i}.bias`
//! (`buckets · heads` values, bucket-major) and `p.norm`.
//!
//! `forward` runs inside a team run.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::{rms_norm, weigh};
use kernels::{Table, Vector, dot_f32};
use team::{Ctx, SharedBuf};

use crate::layer::{self, Attention, Heads, Layer, LayerSpec, Pass, Stack, Widths};
use crate::rotary::Rotary;

/// The dimensions of one encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub vocabulary: usize,
    pub buckets: usize,
    pub max_distance: usize,
    pub norm_eps: f32,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.heads,
            head_dim: self.head_dim,
        }
    }

    fn widths(&self) -> Widths {
        Widths {
            hidden: self.hidden,
            heads: self.heads(),
            mixer: self.heads().qkv(),
            feed_forward: self.feed_forward,
        }
    }

    /// Activations for calls of up to `rows` tokens.
    pub fn buffers(&self, rows: usize, threads: usize) -> Buffers {
        let reach = rows as isize - 1;
        Buffers {
            stack: layer::Buffers::new(&self.widths(), rows, threads),
            buckets: (-reach..=reach).map(|offset| self.bucket(offset)).collect(),
            states: SharedBuf::new(rows * self.hidden),
        }
    }

    /// T5's bidirectional bucket of a key `offset` positions after its query (before it if
    /// negative).
    fn bucket(&self, offset: isize) -> usize {
        let half = self.buckets / 2;
        let exact = half / 2;
        let base = if offset > 0 { half } else { 0 };
        let distance = offset.unsigned_abs();
        if distance < exact {
            return base + distance;
        }
        let ratio =
            (distance as f64 / exact as f64).ln() / (self.max_distance as f64 / exact as f64).ln();
        base + (exact + (ratio * (half - exact) as f64).floor() as usize).min(half - 1)
    }
}

/// The activations of calls of up to some rows, and the states a call leaves.
pub struct Buffers {
    stack: layer::Buffers,
    /// The bucket of each key offset from `−(rows − 1)` to `rows − 1`, at `offset + rows − 1`.
    buckets: Vec<usize>,
    /// The last hidden states out, `rows × hidden`.
    pub states: SharedBuf,
}

pub struct Encoder {
    shape: Shape,
    embedding: Table,
    stack: Stack,
    biases: Vec<Vector>,
    norm: Vector,
}

impl Encoder {
    /// The last hidden states of `tokens` (at most the buffers' rows, each below the vocabulary)
    /// into `buffers.states`.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, tokens: &[u32]) {
        let (hidden, rows) = (self.shape.hidden, tokens.len());
        let embedding = self.embedding.matrix();
        for row in ctx.split(rows, 1) {
            let at = row * hidden..(row + 1) * hidden;
            embedding.dequantize_row(tokens[row] as usize, buffers.stack.hidden.part(at));
        }
        ctx.barrier();
        let attention = Relative {
            biases: &self.biases,
            buckets: &buffers.buckets,
            rows,
            heads: self.shape.heads,
        };
        self.stack.forward(ctx, &buffers.stack, rows, &attention);
        for row in ctx.split(rows, 1) {
            let at = row * hidden..(row + 1) * hidden;
            let last = &buffers.stack.hidden.read()[at.clone()];
            rms_norm(
                last,
                self.norm.values(),
                self.shape.norm_eps,
                buffers.states.part(at),
            );
        }
        ctx.barrier();
    }
}

/// Attention of every row over every row, plus each layer's bias for the query and key offset.
struct Relative<'call> {
    biases: &'call [Vector],
    buckets: &'call [usize],
    rows: usize,
    heads: usize,
}

impl Attention for Relative<'_> {
    fn position(&self, row: usize) -> usize {
        row
    }

    fn keep(&self, _layer: usize, _unit: usize, _position: usize, _head: &[f32]) {}

    /// Split over (query, head) pairs.
    fn attend(&self, pass: &Pass, layer: usize) {
        for unit in pass.ctx.split(self.heads * self.rows, 1) {
            self.unit(pass, layer, (unit / self.heads, unit % self.heads));
        }
    }
}

impl Relative<'_> {
    /// Head `head` of query row `query`, scores unscaled. Never inlined: V8 optimizes a wasm
    /// function only on a later call, never a loop already running.
    #[inline(never)]
    fn unit(&self, pass: &Pass, layer: usize, (query, head): (usize, usize)) {
        let head_dim = pass.heads.head_dim;
        let (qkv_width, width) = (pass.heads.qkv(), pass.heads.width());
        let qkv = pass.buffers.qkv.read();
        let at = |row: usize, part: usize| row * qkv_width + part * width + head * head_dim;
        let query_head = &qkv[at(query, 0)..][..head_dim];
        let centre = self.buckets.len() / 2;
        let bias = self.biases[layer].values();
        let scores = &mut pass.buffers.scratch.get(pass.ctx).scores;
        scores.resize(self.rows, 0.0);
        for (key, score) in scores.iter_mut().enumerate() {
            let bucket = self.buckets[centre + key - query];
            *score = dot_f32(query_head, &qkv[at(key, 1)..][..head_dim])
                + bias[bucket * self.heads + head];
        }
        let start = query * width + head * head_dim;
        let out = pass.buffers.attention.part(start..start + head_dim);
        weigh(scores, |key| &qkv[at(key, 2)..][..head_dim], out);
    }
}

impl Load for Encoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            heads,
            head_dim,
            ..
        } = shape;
        if heads * head_dim != hidden {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads of {head_dim} for a width of {hidden}"
            )));
        }
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| {
                let eps = shape.norm_eps;
                Layer::load(store, name, &LayerSpec::UMT5, &shape.widths(), eps)
            },
        )?;
        let biases = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| Vector::load(store, &format!("{name}.bias"), shape.buckets * heads),
        )?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            embedding: scope.load("embedding", [shape.vocabulary, hidden])?,
            stack: Stack::new(
                shape.heads(),
                LayerSpec::UMT5.rope,
                Rotary::default(),
                layers,
            ),
            biases,
            norm: scope.load("norm", hidden)?,
        })
    }
}
