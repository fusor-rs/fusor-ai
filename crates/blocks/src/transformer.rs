//! Decoder transformers with a KV cache: pre-norm layers (`layer`) whose attention is
//! multi-head or grouped-query, causal or sliding-window, over a cache of past keys and values.
//! Two architectures share it:
//!
//! - Qwen2: RMSNorm, a fused QKV with biases, rotary embeddings by halves, a SiLU-gated MLP.
//!   Tensors under a prefix `p`: `p.layers.{i}.{input_norm, qkv, o, post_norm, gate, up, down}`
//!   (q, k and v fused into `qkv`). ERNIE 4.5 is the same layer without biases.
//! - Kyutai's streaming transformer (Moshi, Mimi, Pocket TTS, Phonon): LayerNorm, a bias-free
//!   fused QKV, interleaved rotary embeddings, multi-head attention and an exact-GELU MLP without
//!   biases; a checkpoint's layer-scale factors are folded into `out` and `ff2` by its compiler.
//!   Tensors under a prefix `p`: `p.layers.{i}.{norm1, qkv, out, norm2, ff1, ff2}`, the norms as
//!   `weight` and `bias` vectors.
//! - CLIP's text encoder: the Kyutai layer's tensors, with biases on every linear, a quick-GELU
//!   MLP and no rotary embeddings: its positions are learned, added to the input by the model
//!   (`rope_base` is unused).
//!
//! A final norm after the last layer, where a model has one, is the model's.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::attend;
use team::{Ctx, SharedBuf};

pub use crate::layer::Buffers;
use crate::layer::{
    Activation, Attention, FeedForward, Heads, Layer, Norm, Pass, Rope, Stack, Widths,
};
use crate::norm::LayerNorm;
use crate::rotary::{Position, Rotary, Rotation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Kyutai,
    Clip,
}

/// The dimensions of one stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub architecture: Architecture,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub rope_base: f32,
    pub rotation: Rotation,
    pub norm_eps: f32,
}

impl Shape {
    fn heads(&self) -> Heads {
        Heads {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// Activations for calls of up to `rows` positions.
    pub fn buffers(&self, rows: usize, threads: usize) -> Buffers {
        let widths = Widths {
            hidden: self.hidden,
            heads: self.heads(),
            feed_forward: self.feed_forward,
        };
        Buffers::new(&widths, rows, threads)
    }
}

pub struct Transformer {
    stack: Stack,
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
/// `p % capacity`. A causal cache must hold the whole sequence; a sliding-window ring must hold
/// the window plus one call's rows, so no slot is overwritten while still visible.
pub struct KvCache {
    heads: Heads,
    capacity: usize,
    keys: SharedBuf,
    values: SharedBuf,
}

impl KvCache {
    pub fn new(shape: Shape, capacity: usize) -> Self {
        let len = shape.layers * shape.kv_heads * capacity * shape.head_dim;
        Self {
            heads: shape.heads(),
            capacity,
            keys: SharedBuf::new(len),
            values: SharedBuf::new(len),
        }
    }

    fn offset(&self, layer: usize, kv_head: usize, position: usize) -> usize {
        let slot = position % self.capacity;
        let Heads {
            kv_heads, head_dim, ..
        } = self.heads;
        ((layer * kv_heads + kv_head) * self.capacity + slot) * head_dim
    }
}

/// One call: `rows` consecutive positions starting at `first`, attending under `mask`. Their
/// rotary angles turn by their sequence positions, unless `forward_at` gives them their own.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub rows: usize,
    pub first: usize,
    pub mask: Mask,
}

/// A span whose row `r` turns its rotary angles by `positions[r]`.
#[derive(Debug, Clone, Copy)]
pub struct Positioned<'rows> {
    pub span: Span,
    pub positions: &'rows [Position],
}

impl Transformer {
    /// Runs every layer over `buffers.hidden[..span.rows]` (the residual stream, in and out).
    ///
    /// # Panics
    /// When the cache would overwrite a key the span still sees: a causal span past the cache's
    /// capacity, or a window plus the span's rows over it.
    pub fn forward(&self, ctx: &Ctx, cache: &KvCache, buffers: &Buffers, span: Span) {
        self.run(ctx, buffers, &Cached::new(cache, span, None));
    }

    /// `forward` with each row's rotary angles turned by its own position (multimodal RoPE).
    ///
    /// # Panics
    /// As `forward`, and when there is not one position per row.
    pub fn forward_at(&self, ctx: &Ctx, cache: &KvCache, buffers: &Buffers, at: Positioned) {
        let Positioned { span, positions } = at;
        assert_eq!(positions.len(), span.rows, "one position per row");
        self.run(ctx, buffers, &Cached::new(cache, span, Some(positions)));
    }

    fn run(&self, ctx: &Ctx, buffers: &Buffers, attention: &Cached) {
        let Cached { cache, span, .. } = attention;
        let reach = match span.mask {
            Mask::Causal => span.first + span.rows,
            Mask::Window(window) => window + span.rows,
        };
        assert!(
            reach <= cache.capacity,
            "the cache holds every key the span sees"
        );
        self.stack.forward(ctx, buffers, span.rows, attention);
    }
}

/// Attention over the KV cache.
struct Cached<'call> {
    cache: &'call KvCache,
    span: Span,
    positions: Option<&'call [Position]>,
}

impl<'call> Cached<'call> {
    fn new(cache: &'call KvCache, span: Span, positions: Option<&'call [Position]>) -> Self {
        Self {
            cache,
            span,
            positions,
        }
    }
}

impl Attention for Cached<'_> {
    fn position(&self, row: usize) -> usize {
        self.span.first + row
    }

    fn rotary(&self, row: usize) -> Position {
        match self.positions {
            Some(positions) => positions[row],
            None => Position::uniform(self.position(row)),
        }
    }

    fn keep(&self, layer: usize, unit: usize, position: usize, head: &[f32]) {
        let Heads {
            heads, kv_heads, ..
        } = self.cache.heads;
        if unit < heads {
            return;
        }
        let target = if unit < heads + kv_heads {
            &self.cache.keys
        } else {
            &self.cache.values
        };
        let at = self
            .cache
            .offset(layer, (unit - heads) % kv_heads, position);
        target.part(at..at + head.len()).copy_from_slice(head);
    }

    fn attend(&self, pass: &Pass, layer: usize) {
        let Heads {
            heads,
            kv_heads,
            head_dim,
        } = pass.heads;
        let (cache, buffers, span) = (self.cache, pass.buffers, self.span);
        let (qkv_width, width) = (pass.heads.qkv(), pass.heads.width());
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
                let at = row * width + head * head_dim;
                attend(
                    query,
                    position + 1 - first_key,
                    |j| {
                        let slot = cache.offset(layer, kv_head, first_key + j);
                        (&keys[slot..slot + head_dim], &values[slot..slot + head_dim])
                    },
                    &mut scratch.scores,
                    buffers.attention.part(at..at + head_dim),
                );
            }
        }
    }
}

impl Load for Transformer {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            heads, kv_heads, ..
        } = shape;
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads over {kv_heads} KV heads"
            )));
        }
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| layer(store, name, shape),
        )?;
        let rope = match shape.architecture {
            Architecture::Qwen2 => Some(Rope::Half),
            Architecture::Kyutai => Some(Rope::Interleaved),
            Architecture::Clip => None,
        };
        let rotary = Rotary::new(shape.rotation, shape.rope_base, shape.head_dim);
        Ok(Self {
            stack: Stack::new(shape.heads(), rope, rotary, layers),
        })
    }
}

fn layer(store: &mut Store, prefix: &str, shape: Shape) -> Result<Layer, bundle::Error> {
    match shape.architecture {
        Architecture::Qwen2 => qwen2_layer(store, prefix, shape),
        Architecture::Kyutai => layer_norm_layer(store, prefix, shape, Activation::GeluErf),
        Architecture::Clip => layer_norm_layer(store, prefix, shape, Activation::QuickGelu),
    }
}

fn qwen2_layer(store: &mut Store, prefix: &str, shape: Shape) -> Result<Layer, bundle::Error> {
    let Shape {
        hidden,
        feed_forward,
        norm_eps: eps,
        ..
    } = shape;
    let mut scope = Scope::new(store, prefix);
    let rms = |weight| Norm::Rms { weight, eps };
    Ok(Layer {
        norm1: rms(scope.load("input_norm", hidden)?),
        qkv: scope.load("qkv", [shape.heads().qkv(), hidden])?,
        out: scope.load("o", [hidden, shape.heads().width()])?,
        norm2: rms(scope.load("post_norm", hidden)?),
        feed_forward: FeedForward::Gated {
            gate: scope.load("gate", [feed_forward, hidden])?,
            up: scope.load("up", [feed_forward, hidden])?,
            down: scope.load("down", [hidden, feed_forward])?,
        },
    })
}

/// A Kyutai or CLIP layer: LayerNorms, and an MLP with `activation`.
fn layer_norm_layer(
    store: &mut Store,
    prefix: &str,
    shape: Shape,
    activation: Activation,
) -> Result<Layer, bundle::Error> {
    let Shape {
        hidden,
        feed_forward,
        norm_eps: eps,
        ..
    } = shape;
    let mut scope = Scope::new(store, prefix);
    let layer_norm = |norm: LayerNorm| Norm::Layer { norm, eps };
    Ok(Layer {
        norm1: layer_norm(scope.load("norm1", hidden)?),
        qkv: scope.load("qkv", [shape.heads().qkv(), hidden])?,
        out: scope.load("out", [hidden, shape.heads().width()])?,
        norm2: layer_norm(scope.load("norm2", hidden)?),
        feed_forward: FeedForward::Plain {
            up: scope.load("ff1", [feed_forward, hidden])?,
            down: scope.load("ff2", [hidden, feed_forward])?,
            activation,
        },
    })
}
