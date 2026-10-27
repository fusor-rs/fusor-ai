//! Decoder stacks whose layers interleave sliding-window and full attention (Spark-X2.6; Gemma 3's
//! and Cohere 2's pattern): pre-norm layers (`layer`) whose attention is `hybrid`'s gated
//! attention, in every `global_interval` layers over every earlier row for the last and over the
//! last `Window::rows` rows for the others, which rotate by a rotary embedding of their own; each
//! followed by a GELU-gated MLP (exact GELU).
//!
//! Tensors under a prefix `p`: `p.layers.{i}.{input_norm, post_norm, gate, up, down}`, and the
//! attention's (`hybrid::attention` lists them). A final norm after the last layer is the
//! model's.
//!
//! `Interleaved::forward` runs inside a team run. Calls read consecutive rows of one sequence
//! from its first: the window layers keep their keys in a ring that holds the window and one
//! call's rows, so a sequence is read on, never from an earlier row again.

use bundle::{Load, Scope, Store, load_each};
use team::Ctx;

use crate::hybrid::attention::{Call, Weights};
use crate::hybrid::{AttentionShape, Rows};
use crate::layer::{
    self, Activation, FeedForward, Norm, Pass, Source, Widths, attended_residual,
    feed_forward_residual, input, prepare,
};
use crate::transformer::{KvCache, Mask, Span};

/// What a window layer's attention has of its own: the rows it sees, up to and including its
/// own, and its rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub rows: usize,
    pub rotary_dim: usize,
    pub rope_base: f32,
}

/// The dimensions of one stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    /// The last of every this many layers attends to every earlier row, the others through the
    /// window.
    pub global_interval: usize,
    /// The full attention layers' attention, and the window layers' but for `window`.
    pub attention: AttentionShape,
    pub window: Window,
    pub feed_forward: usize,
    pub norm_eps: f32,
}

impl Shape {
    fn is_global(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.global_interval)
    }

    /// The layer's index among the stack's layers of its attention.
    fn slot(&self, layer: usize) -> usize {
        let global_through = (layer + 1) / self.global_interval;
        if self.is_global(layer) {
            global_through - 1
        } else {
            layer - global_through
        }
    }

    fn global_layers(&self) -> usize {
        self.layers / self.global_interval
    }

    /// The attention of layer `layer`.
    fn attention(&self, layer: usize) -> AttentionShape {
        if self.is_global(layer) {
            return self.attention;
        }
        AttentionShape {
            rotary_dim: self.window.rotary_dim,
            rope_base: self.window.rope_base,
            ..self.attention
        }
    }

    /// Activations for calls of up to `rows` rows.
    pub fn buffers(&self, rows: usize, threads: usize) -> layer::Buffers {
        let widths = Widths {
            hidden: self.hidden,
            heads: self.attention.heads(),
            mixer: self.attention.qkv(),
            feed_forward: self.feed_forward,
        };
        layer::Buffers::new(&widths, rows, threads)
    }
}

/// The keys and values of every layer: the full attention layers' of every row, the window
/// layers' of the last rows.
pub struct State {
    global: KvCache,
    capacity: usize,
    window: KvCache,
    window_capacity: usize,
}

impl State {
    /// The state of sequences of up to `capacity` rows, read in calls of up to `rows` rows.
    pub fn new(shape: &Shape, capacity: usize, rows: usize) -> Self {
        let heads = shape.attention.heads();
        let window_capacity = shape.window.rows + rows;
        let window_layers = shape.layers - shape.global_layers();
        Self {
            global: KvCache::of(shape.global_layers(), heads, capacity),
            capacity,
            window: KvCache::of(window_layers, heads, window_capacity),
            window_capacity,
        }
    }
}

struct InterleavedLayer {
    norm1: Norm,
    attention: Weights,
    norm2: Norm,
    feed_forward: FeedForward,
}

pub struct Interleaved {
    shape: Shape,
    layers: Vec<InterleavedLayer>,
}

impl Interleaved {
    /// Every layer over `buffers.hidden[..rows.positions.len()]` (the residual stream, in and
    /// out): the full attention layers see keys under `rows.mask`, the window layers through
    /// their window.
    ///
    /// # Panics
    /// When the rows are more than the buffers hold, or the caches would overwrite a key they
    /// still see.
    pub fn forward(&self, ctx: &Ctx, state: &State, buffers: &layer::Buffers, rows: &Rows) {
        let Rows {
            first,
            positions,
            mask,
        } = *rows;
        let rows = positions.len();
        assert!(rows <= buffers.rows, "the buffers hold the rows");
        assert!(
            first + rows <= state.capacity
                && self.shape.window.rows + rows <= state.window_capacity,
            "the caches hold every key the rows see"
        );
        let pass = Pass {
            ctx,
            buffers,
            rows,
            heads: self.shape.attention.heads(),
        };
        for (index, layer) in self.layers.iter().enumerate() {
            let (cache, mask) = if self.shape.is_global(index) {
                (&state.global, mask)
            } else {
                (&state.window, Mask::Window(self.shape.window.rows))
            };
            let call = Call {
                cache,
                slot: self.shape.slot(index),
                span: Span { rows, first, mask },
                positions,
            };
            let projection = &layer.attention.qkv;
            prepare(&pass, Source::Normed(&layer.norm1), projection);
            ctx.barrier();
            let normed = input(&pass, Source::Normed(&layer.norm1), projection);
            layer.attention.mix(&pass, &normed, &call);
            attended_residual(&pass, &layer.attention.out);
            feed_forward_residual(&pass, &layer.norm2, &layer.feed_forward);
        }
    }
}

impl Load for Interleaved {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let AttentionShape {
            heads, kv_heads, ..
        } = shape.attention;
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads over {kv_heads} KV heads"
            )));
        }
        let mut index = 0;
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.layers,
            |store, name| {
                let layer = load_layer(store, name, &shape, index);
                index += 1;
                layer
            },
        )?;
        Ok(Self { shape, layers })
    }
}

/// Layer `index`, under `prefix`.
fn load_layer(
    store: &mut Store,
    prefix: &str,
    shape: &Shape,
    index: usize,
) -> Result<InterleavedLayer, bundle::Error> {
    let (hidden, eps) = (shape.hidden, shape.norm_eps);
    let attention = Weights::load(store, prefix, &shape.attention(index), hidden, eps)?;
    let mut scope = Scope::new(store, prefix);
    let feed_forward =
        FeedForward::gated(&mut scope, hidden, shape.feed_forward, Activation::GeluErf)?;
    Ok(InterleavedLayer {
        norm1: Norm::Rms {
            weight: scope.load("input_norm", hidden)?,
            eps,
        },
        attention,
        norm2: Norm::Rms {
            weight: scope.load("post_norm", hidden)?,
            eps,
        },
        feed_forward,
    })
}
