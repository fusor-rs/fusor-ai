//! Decoder transformers with a KV cache: pre-norm layers (`layer`) whose attention is
//! multi-head or grouped-query, causal or sliding-window, over a cache of past keys and values.
//! Two architectures share it:
//!
//! - Qwen2: RMSNorm, a fused QKV with biases, rotary embeddings by halves, a SiLU-gated MLP.
//!   Tensors under a prefix `p`: `p.layers.{i}.{input_norm, qkv, o, post_norm, gate, up, down}`
//!   (q, k and v fused into `qkv`). ERNIE 4.5 is the same layer without biases.
//! - Qwen3: Qwen2's layer without biases, each query and key head RMS-normalized before its
//!   rotary embeddings, by `p.layers.{i}.{q_norm, k_norm}` (`head_dim` wide).
//! - Kyutai's streaming transformer (Moshi, Mimi, Pocket TTS, Phonon): LayerNorm, a bias-free
//!   fused QKV, interleaved rotary embeddings, multi-head attention and an exact-GELU MLP without
//!   biases; a checkpoint's layer-scale factors are folded into `out` and `ff2` by its compiler.
//!   Tensors under a prefix `p`: `p.layers.{i}.{norm1, qkv, out, norm2, ff1, ff2}`, the norms as
//!   `weight` and `bias` vectors.
//! - CLIP's text encoder: the Kyutai layer's tensors, with biases on every linear, a quick-GELU
//!   MLP and no rotary embeddings: its positions are learned, added to the input by the model
//!   (`rope_base` is unused).
//! - An encoder (Indra OCR's reading-order network): CLIP's layer with an exact-GELU MLP, its
//!   rows attending to each other under `Mask::Full`.
//!
//! A final norm after the last layer, where a model has one, is the model's.

use std::ops::Range;

use bundle::{Load, Store, load_each};
use kernels::ops::attend;
use team::{Ctx, SharedBuf};

pub use crate::layer::Buffers;
use crate::layer::{Attention, Heads, Layer, LayerSpec, Pass, Stack, Widths};
use crate::rotary::{Position, Rotary, Rotation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Qwen3,
    Kyutai,
    Clip,
    Encoder,
}

impl Architecture {
    fn spec(self) -> &'static LayerSpec {
        match self {
            Architecture::Qwen2 => &LayerSpec::QWEN2,
            Architecture::Qwen3 => &LayerSpec::QWEN3,
            Architecture::Kyutai => &LayerSpec::KYUTAI,
            Architecture::Clip => &LayerSpec::CLIP,
            Architecture::Encoder => &LayerSpec::ENCODER,
        }
    }
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

    fn widths(&self) -> Widths {
        Widths {
            hidden: self.hidden,
            heads: self.heads(),
            mixer: self.heads().qkv(),
            feed_forward: self.feed_forward,
        }
    }

    /// Activations for calls of up to `rows` positions.
    pub fn buffers(&self, rows: usize, threads: usize) -> Buffers {
        Buffers::new(&self.widths(), rows, threads)
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
    /// Every position of the call and of the calls before it: an encoder reads its whole
    /// sequence in one call.
    Full,
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

/// A cache's first rows' keys and values, every layer's and head's, copied out to put back.
pub(crate) struct KvRows {
    rows: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    pub fn new(shape: Shape, capacity: usize) -> Self {
        Self::of(shape.layers, shape.heads(), capacity)
    }

    /// The cache of `layers` attention layers of `heads`.
    pub(crate) fn of(layers: usize, heads: Heads, capacity: usize) -> Self {
        let len = layers * heads.kv_heads * capacity * heads.head_dim;
        Self {
            heads,
            capacity,
            keys: SharedBuf::new(len),
            values: SharedBuf::new(len),
        }
    }

    /// Keeps head `unit` (query heads, then key heads, then value heads) of a row at `position`
    /// of layer `layer`: its key and value heads.
    pub(crate) fn keep(&self, layer: usize, unit: usize, position: usize, head: &[f32]) {
        let Heads {
            heads, kv_heads, ..
        } = self.heads;
        if unit < heads {
            return;
        }
        let target = if unit < heads + kv_heads {
            &self.keys
        } else {
            &self.values
        };
        let at = self.offset(layer, (unit - heads) % kv_heads, position);
        target.part(at..at + head.len()).copy_from_slice(head);
    }

    /// This thread's share of the query heads of layer `layer`, attending over the cache under
    /// `span`'s mask into `buffers.attention`; row `r`'s queries lead row `r` of `buffers.qkv`,
    /// `stride` wide.
    pub(crate) fn attend(&self, pass: &Pass, layer: usize, span: Span, stride: usize) {
        let Heads {
            heads,
            kv_heads,
            head_dim,
        } = self.heads;
        let buffers = pass.buffers;
        let width = self.heads.width();
        let scratch = buffers.scratch.get(pass.ctx);
        let qkv = buffers.qkv.read();
        let (keys, values) = (self.keys.read(), self.values.read());
        for head in pass.ctx.split(heads, 1) {
            let kv_head = head / (heads / kv_heads);
            for row in 0..span.rows {
                let position = span.first + row;
                let (first_key, end_key) = match span.mask {
                    Mask::Causal => (0, position + 1),
                    Mask::Window(window) => ((position + 1).saturating_sub(window), position + 1),
                    Mask::Full => (0, span.first + span.rows),
                };
                let query = &qkv[row * stride + head * head_dim..][..head_dim];
                let at = row * width + head * head_dim;
                attend(
                    query,
                    end_key - first_key,
                    |j| {
                        let slot = self.offset(layer, kv_head, first_key + j);
                        (&keys[slot..slot + head_dim], &values[slot..slot + head_dim])
                    },
                    &mut scratch.scores,
                    buffers.attention.part(at..at + head_dim),
                );
            }
        }
    }

    /// The first `rows` rows' keys and values, to put back with `put_rows`.
    pub(crate) fn copy_rows(&self, rows: usize) -> KvRows {
        assert!(
            rows <= self.capacity,
            "{rows} rows don't fit a cache of {}",
            self.capacity
        );
        let (keys, values) = (self.keys.read(), self.values.read());
        let mut copied = KvRows {
            rows,
            keys: Vec::new(),
            values: Vec::new(),
        };
        for block in self.blocks(rows) {
            copied.keys.extend_from_slice(&keys[block.clone()]);
            copied.values.extend_from_slice(&values[block]);
        }
        copied
    }

    /// Puts back the rows `copy_rows` copied, as they were.
    pub(crate) fn put_rows(&mut self, copied: &KvRows) {
        let width = copied.rows * self.heads.head_dim;
        let blocks = self.blocks(copied.rows);
        let (keys, values) = (self.keys.get_mut(), self.values.get_mut());
        for (index, block) in blocks.enumerate() {
            let from = index * width..(index + 1) * width;
            keys[block.clone()].copy_from_slice(&copied.keys[from.clone()]);
            values[block].copy_from_slice(&copied.values[from]);
        }
    }

    // Where the first `rows` rows lie in the cache, a block for each layer's key head, in order.
    fn blocks(&self, rows: usize) -> impl Iterator<Item = Range<usize>> + use<> {
        let (head_dim, capacity) = (self.heads.head_dim, self.capacity);
        let heads = self.keys.read().len() / (capacity * head_dim);
        (0..heads).map(move |head| {
            let first = head * capacity * head_dim;
            first..first + rows * head_dim
        })
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
            Mask::Causal | Mask::Full => span.first + span.rows,
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
        self.cache.keep(layer, unit, position, head);
    }

    fn attend(&self, pass: &Pass, layer: usize) {
        self.cache.attend(pass, layer, self.span, pass.heads.qkv());
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
        let spec = shape.architecture.spec();
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| Layer::load(store, name, spec, &shape.widths(), shape.norm_eps),
        )?;
        let rotary = Rotary::new(shape.rotation, shape.rope_base, shape.head_dim);
        Ok(Self {
            stack: Stack::new(shape.heads(), spec.rope, rotary, layers),
        })
    }
}
