//! Hybrid decoder stacks (Qwen3-Next, Qwen3.5): pre-norm layers (`layer`) whose token mixer is,
//! in every `attention_interval` layers, gated full attention for the last and a Gated DeltaNet
//! for the others; each followed by a SiLU-gated MLP.
//!
//! - `attention`: grouped-query attention over a KV cache, each query and key head
//!   RMS-normalized (`HeadNorm`), the first `rotary_dim` elements of each rotated, and the output
//!   gated by a sigmoid (`Gate`); `interleaved`'s layers attend with it too.
//! - `delta`: the gated delta rule, a linear attention whose state carries from call to call.
//!
//! Tensors under a prefix `p`: `p.layers.{i}.{input_norm, post_norm, gate, up, down}`, and each
//! mixer's (its module lists them). RMSNorm weights are the multipliers themselves: a
//! checkpoint's zero-centred `1 + w` is folded by its compiler. A final norm after the last
//! layer is the model's.
//!
//! `Hybrid::forward` runs inside a team run. Calls read consecutive rows of one sequence from its
//! first, or from a row whose state was saved (`State::save`, `State::restore`), or from the end
//! of a prefix kept with its attention rows, after other sequences were read (`State::keep_prefix`,
//! `State::restore_prefix`): the delta rule's state is the sum of every row before. The attention layers see keys under each call's mask;
//! under `Mask::Full` a row also sees the later rows of its own call, so a sequence read that way
//! is read in one call.

pub(crate) mod attention;
mod delta;

use bundle::{Load, Scope, Store, load_each};
use team::{Ctx, PerThread, SharedBuf};

pub use crate::layer::HeadNorm;
use crate::layer::{
    self, Activation, FeedForward, Heads, Norm, Pass, Source, Widths, attended_residual,
    feed_forward_residual, input, prepare,
};
use crate::rotary::Position;
use crate::transformer::{KvCache, KvRows, Mask, Span};
pub use attention::{Gate, Shape as AttentionShape};
pub use delta::Shape as DeltaShape;

/// The dimensions of one stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    /// The last of every this many layers mixes by attention, the others by the delta rule.
    pub attention_interval: usize,
    pub attention: AttentionShape,
    pub delta: DeltaShape,
    pub feed_forward: usize,
    pub norm_eps: f32,
}

impl Shape {
    fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.attention_interval)
    }

    /// The layer's index among the stack's layers of its mixer.
    fn slot(&self, layer: usize) -> usize {
        let attention_through = (layer + 1) / self.attention_interval;
        if self.is_attention(layer) {
            attention_through - 1
        } else {
            layer - attention_through
        }
    }

    fn attention_layers(&self) -> usize {
        self.layers / self.attention_interval
    }

    fn delta_layers(&self) -> usize {
        self.layers - self.attention_layers()
    }

    fn widths(&self) -> Widths {
        Widths {
            hidden: self.hidden,
            heads: self.attention.heads(),
            mixer: self.attention.qkv().max(self.delta.input()),
            feed_forward: self.feed_forward,
        }
    }

    /// Activations for calls of up to `rows` rows.
    pub fn buffers(&self, rows: usize, threads: usize) -> Buffers {
        Buffers {
            stack: layer::Buffers::new(&self.widths(), rows, threads),
            convolved: SharedBuf::new(rows * self.delta.channels()),
            scratch: PerThread::new(threads, delta::Scratch::default),
        }
    }
}

/// Activations of one call.
pub struct Buffers {
    /// The layers' activations; `stack.hidden` is the residual stream, in and out.
    pub stack: layer::Buffers,
    /// The delta rule's queries, keys and values after their convolution, `rows × channels`.
    convolved: SharedBuf,
    scratch: PerThread<delta::Scratch>,
}

/// What carries from call to call: the attention layers' keys and values, and the delta rule's
/// convolution inputs and state.
pub struct State {
    cache: KvCache,
    delta: delta::State,
}

impl State {
    /// The state of sequences of up to `capacity` rows.
    pub fn new(shape: &Shape, capacity: usize) -> Self {
        Self {
            cache: KvCache::of(shape.attention_layers(), shape.attention.heads(), capacity),
            delta: delta::State::new(&shape.delta, shape.delta_layers()),
        }
    }

    /// Forgets the sequence read so far, to read a new one from its first row.
    pub fn clear(&mut self) {
        self.delta.clear();
    }

    /// Keeps the state after the rows read so far in `saved`.
    pub fn save(&self, saved: &mut Saved) {
        saved.0.copy_from(&self.delta);
    }

    /// The state after the first `rows` rows read, to read another sequence that begins with
    /// them on from there, whatever was read in between.
    pub fn keep_prefix(&self, shape: &Shape, rows: usize) -> Prefix {
        let mut saved = Saved::new(shape);
        self.save(&mut saved);
        Prefix {
            saved,
            attention: self.cache.copy_rows(rows),
        }
    }

    /// Goes back to the state after `prefix`'s rows, to read on from the row after them.
    pub fn restore_prefix(&mut self, prefix: &Prefix) {
        self.restore(&prefix.saved);
        self.cache.put_rows(&prefix.attention);
    }

    /// Goes back to the state `saved` kept, to read on from the row after the rows read then:
    /// the attention layers' keys of later rows are overwritten as those rows are read again.
    pub fn restore(&mut self, saved: &Saved) {
        self.delta.copy_from(&saved.0);
    }
}

/// The delta rule's part of a `State`, kept to read a sequence on again from where it was.
pub struct Saved(delta::State);

/// The state after a sequence's first rows: the delta rule's, and the attention layers' keys
/// and values of those rows, which a sequence read after them overwrites.
pub struct Prefix {
    saved: Saved,
    attention: KvRows,
}

impl Saved {
    pub fn new(shape: &Shape) -> Self {
        Self(delta::State::new(&shape.delta, shape.delta_layers()))
    }
}

/// Rows of one sequence: from sequence position `first`, row `r` turning its rotary angles by
/// `positions[r]`, the attention layers' keys seen under `mask`.
#[derive(Clone, Copy)]
pub struct Rows<'rows> {
    pub first: usize,
    pub positions: &'rows [Position],
    pub mask: Mask,
}

enum Mixer {
    Attention(attention::Weights),
    Delta(delta::Weights),
}

struct HybridLayer {
    norm1: Norm,
    mixer: Mixer,
    norm2: Norm,
    feed_forward: FeedForward,
}

pub struct Hybrid {
    shape: Shape,
    layers: Vec<HybridLayer>,
}

impl Hybrid {
    /// Every layer over `buffers.stack.hidden[..rows.positions.len()]` (the residual stream, in
    /// and out).
    ///
    /// # Panics
    /// When the rows are more than the buffers hold, or reach past the cache's capacity.
    pub fn forward(&self, ctx: &Ctx, state: &State, buffers: &Buffers, rows: &Rows) {
        let Rows {
            first,
            positions,
            mask,
        } = *rows;
        let rows = positions.len();
        assert!(rows <= buffers.stack.rows, "the buffers hold the rows");
        let span = Span { rows, first, mask };
        let pass = Pass {
            ctx,
            buffers: &buffers.stack,
            rows,
            heads: self.shape.attention.heads(),
        };
        for (index, layer) in self.layers.iter().enumerate() {
            let slot = self.shape.slot(index);
            let (projection, out) = match &layer.mixer {
                Mixer::Attention(weights) => (&weights.qkv, &weights.out),
                Mixer::Delta(weights) => (&weights.input, &weights.out),
            };
            prepare(&pass, Source::Normed(&layer.norm1), projection);
            ctx.barrier();
            let normed = input(&pass, Source::Normed(&layer.norm1), projection);
            match &layer.mixer {
                Mixer::Attention(weights) => {
                    let call = attention::Call {
                        cache: &state.cache,
                        slot,
                        span,
                        positions,
                    };
                    weights.mix(&pass, &normed, &call);
                }
                Mixer::Delta(weights) => {
                    let call = delta::Call {
                        state: &state.delta,
                        slot,
                        convolved: &buffers.convolved,
                        scratch: &buffers.scratch,
                    };
                    weights.mix(&pass, &normed, &call);
                }
            }
            attended_residual(&pass, out);
            feed_forward_residual(&pass, &layer.norm2, &layer.feed_forward);
        }
    }
}

impl Load for Hybrid {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Heads {
            heads, kv_heads, ..
        } = shape.attention.heads();
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads over {kv_heads} KV heads"
            )));
        }
        if shape.delta.values() != shape.attention.heads().width() {
            return Err(bundle::Error::Format(format!(
                "{prefix}: the mixers' outputs differ in width, and share one buffer"
            )));
        }
        if !shape.delta.value_dim.is_multiple_of(kernels::DOTS) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: the delta rule's value width {} is not a multiple of {}",
                shape.delta.value_dim,
                kernels::DOTS
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
) -> Result<HybridLayer, bundle::Error> {
    let (hidden, eps) = (shape.hidden, shape.norm_eps);
    let mixer = if shape.is_attention(index) {
        Mixer::Attention(attention::Weights::load(
            store,
            prefix,
            &shape.attention,
            hidden,
            eps,
        )?)
    } else {
        Mixer::Delta(delta::Weights::load(
            store,
            prefix,
            &shape.delta,
            hidden,
            eps,
        )?)
    };
    let mut scope = Scope::new(store, prefix);
    let feed_forward =
        FeedForward::gated(&mut scope, hidden, shape.feed_forward, Activation::Silu)?;
    Ok(HybridLayer {
        norm1: Norm::Rms {
            weight: scope.load("input_norm", hidden)?,
            eps,
        },
        mixer,
        norm2: Norm::Rms {
            weight: scope.load("post_norm", hidden)?,
            eps,
        },
        feed_forward,
    })
}
