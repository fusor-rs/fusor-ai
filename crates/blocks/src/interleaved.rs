//! Decoder stacks whose layers interleave sliding-window and full attention (Spark-X2.6; Gemma 3's
//! and Cohere 2's pattern), and gated short convolutions (LFM3): pre-norm layers (`layer`), each
//! mixing its rows by its `Mixer`, followed by a gated MLP. The attention is `hybrid`'s gated
//! attention, over every earlier row or over the last `Window::rows` rows, which rotate by a
//! rotary embedding of their own; the convolution is `conv`'s. Under `Norms::Sandwich` each
//! sublayer's output is RMS-normalized too before the residual stream takes it.
//!
//! Tensors under a prefix `p`: `p.layers.{i}.{input_norm, post_norm, gate, up, down}`, under
//! `Norms::Sandwich` `p.layers.{i}.{mixer_out_norm, feed_forward_out_norm}`, and the mixer's
//! (`hybrid::attention` and `conv` list them). A final norm after the last layer is the model's.
//!
//! `Interleaved::forward` runs inside a team run. Calls read consecutive rows of one sequence
//! from its first: the window layers keep their keys in a ring that holds the window and one
//! call's rows, and the conv layers their last inputs, so a sequence is read on, never from an
//! earlier row again.

pub(crate) mod conv;

use bundle::{Load, Scope, Store, load_each};
use team::{Ctx, SharedBuf};

use crate::hybrid::attention::{self, Weights};
use crate::hybrid::{AttentionShape, Rows};
pub use crate::layer::Activation;
use crate::layer::{
    self, FeedForward, Norm, Pass, Source, Widths, attended_residual, feed_forward_residual, input,
    prepare,
};
use crate::rotary::Rotation;
use crate::transformer::{KvCache, Mask, Span};

/// How a layer mixes its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Attention over every earlier row.
    Full,
    /// Attention over the window's rows.
    Window,
    /// The gated short convolution.
    Conv,
}

/// What a window layer's attention has of its own: the rows it sees, up to and including its
/// own, and its rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub rows: usize,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub rotation: Rotation,
}

/// Which outputs of a layer are normed: the sublayers' inputs, or their outputs too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norms {
    Pre,
    Sandwich,
}

/// The dimensions of one stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    /// Each layer's mixer, in order.
    pub mixers: &'static [Mixer],
    /// The full attention layers' attention, and the window layers' but for `window`.
    pub attention: AttentionShape,
    pub window: Window,
    /// Taps of each conv layer's convolution; 0 in a stack without conv layers.
    pub conv_kernel: usize,
    pub feed_forward: usize,
    pub activation: Activation,
    pub norms: Norms,
    pub norm_eps: f32,
}

impl Shape {
    /// The layer's index among the stack's layers of its mixer.
    fn slot(&self, layer: usize) -> usize {
        let mixer = self.mixers[layer];
        self.mixers[..layer]
            .iter()
            .filter(|&&earlier| earlier == mixer)
            .count()
    }

    fn layers_of(&self, mixer: Mixer) -> usize {
        self.mixers.iter().filter(|&&each| each == mixer).count()
    }

    /// The attention of an attention layer of `mixer`.
    fn attention(&self, mixer: Mixer) -> AttentionShape {
        if mixer == Mixer::Full {
            return self.attention;
        }
        AttentionShape {
            rotary_dim: self.window.rotary_dim,
            rope_base: self.window.rope_base,
            rotation: self.window.rotation,
            ..self.attention
        }
    }

    /// Activations for calls of up to `rows` rows.
    pub fn buffers(&self, rows: usize, threads: usize) -> layer::Buffers {
        let conv_input = match self.layers_of(Mixer::Conv) {
            0 => 0,
            _ => conv::input_width(self.hidden),
        };
        let widths = Widths {
            hidden: self.hidden,
            heads: self.attention.heads(),
            mixer: self.attention.qkv().max(conv_input),
            feed_forward: self.feed_forward,
        };
        layer::Buffers::new(&widths, rows, threads)
    }
}

/// What every layer keeps of the rows before: the full attention layers' keys and values of
/// every row, the window layers' of the last rows, the conv layers' last inputs.
pub struct State {
    global: KvCache,
    capacity: usize,
    window: KvCache,
    window_capacity: usize,
    conv: SharedBuf,
}

impl State {
    /// The state of sequences of up to `capacity` rows, read in calls of up to `rows` rows.
    pub fn new(shape: &Shape, capacity: usize, rows: usize) -> Self {
        let heads = shape.attention.heads();
        let window_capacity = shape.window.rows + rows;
        let conv_inputs = match shape.layers_of(Mixer::Conv) {
            0 => 0,
            layers => layers * shape.hidden * conv::history(shape.conv_kernel),
        };
        Self {
            global: KvCache::of(shape.layers_of(Mixer::Full), heads, capacity),
            capacity,
            window: KvCache::of(shape.layers_of(Mixer::Window), heads, window_capacity),
            window_capacity,
            conv: SharedBuf::new(conv_inputs),
        }
    }

    /// Forgets the sequence read so far, to read a new one from its first row: the conv layers'
    /// kept inputs, which a call from the first row would otherwise read as the rows before it.
    pub fn clear(&mut self) {
        self.conv.get_mut().fill(0.0);
    }
}

enum Mixing {
    Attention(Box<Weights>),
    Conv(Box<conv::Weights>),
}

impl Mixing {
    fn input(&self) -> &kernels::Linear {
        match self {
            Mixing::Attention(attention) => &attention.qkv,
            Mixing::Conv(conv) => &conv.input,
        }
    }

    fn out(&self) -> &kernels::Linear {
        match self {
            Mixing::Attention(attention) => &attention.out,
            Mixing::Conv(conv) => &conv.out,
        }
    }
}

/// The norms of a layer's sublayers' outputs, under `Norms::Sandwich`.
struct Sandwich {
    mixer: Norm,
    feed_forward: Norm,
}

struct InterleavedLayer {
    norm1: Norm,
    mixing: Mixing,
    norm2: Norm,
    feed_forward: FeedForward,
    sandwich: Option<Sandwich>,
}

pub struct Interleaved {
    shape: Shape,
    layers: Vec<InterleavedLayer>,
}

impl Interleaved {
    /// Every layer over `buffers.hidden[..rows.positions.len()]` (the residual stream, in and
    /// out): the full attention layers see keys under `rows.mask`, the window layers through
    /// their window, the conv layers their kept inputs.
    ///
    /// # Panics
    /// When the rows are more than the buffers hold, or the caches would overwrite a key they
    /// still see.
    pub fn forward(&self, ctx: &Ctx, state: &State, buffers: &layer::Buffers, rows: &Rows) {
        let count = rows.positions.len();
        assert!(count <= buffers.rows, "the buffers hold the rows");
        assert!(
            rows.first + count <= state.capacity
                && self.shape.window.rows + count <= state.window_capacity,
            "the caches hold every key the rows see"
        );
        let pass = Pass {
            ctx,
            buffers,
            rows: count,
            heads: self.shape.attention.heads(),
        };
        for (index, layer) in self.layers.iter().enumerate() {
            self.mix(&pass, state, rows, index);
            let sandwich = layer.sandwich.as_ref();
            let mixer_out_norm = sandwich.map(|norms| &norms.mixer);
            attended_residual(&pass, layer.mixing.out(), mixer_out_norm);
            let feed_forward_out_norm = sandwich.map(|norms| &norms.feed_forward);
            let (norm, feed_forward) = (&layer.norm2, &layer.feed_forward);
            feed_forward_residual(&pass, norm, feed_forward, feed_forward_out_norm);
        }
    }

    /// Layer `index`'s mixer over its normed rows, into `buffers.attention`.
    fn mix(&self, pass: &Pass, state: &State, rows: &Rows, index: usize) {
        let layer = &self.layers[index];
        let (norm, projection) = (Source::Normed(&layer.norm1), layer.mixing.input());
        prepare(pass, norm, projection);
        pass.ctx.barrier();
        let normed = &input(pass, norm, projection);
        let slot = self.shape.slot(index);
        let (attention, cache, mask) = match (&layer.mixing, self.shape.mixers[index]) {
            (Mixing::Conv(conv), _) => {
                let reach = conv::Reach::Causal {
                    kept: &state.conv,
                    slot,
                };
                return conv.mix(pass, normed, &reach);
            }
            (Mixing::Attention(attention), Mixer::Window) => {
                let window = Mask::Window(self.shape.window.rows);
                (attention, &state.window, window)
            }
            (Mixing::Attention(attention), _) => (attention, &state.global, rows.mask),
        };
        let call = attention::Call {
            cache,
            slot,
            span: Span {
                rows: pass.rows,
                first: rows.first,
                mask,
            },
            positions: rows.positions,
        };
        attention.mix(pass, normed, &call);
    }
}

impl Load for Interleaved {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let AttentionShape {
            heads,
            kv_heads,
            head_dim,
            ..
        } = shape.attention;
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {heads} heads over {kv_heads} KV heads"
            )));
        }
        if shape.mixers.contains(&Mixer::Conv) && heads * head_dim != shape.hidden {
            return Err(bundle::Error::Format(format!(
                "{prefix}: conv layers write their {} channels where {heads} heads of {head_dim} \
                 attend",
                shape.hidden
            )));
        }
        let mut index = 0;
        let layers = load_each(
            store,
            &format!("{prefix}.layers"),
            shape.mixers.len(),
            |store, name| {
                let layer = load_layer(store, name, &shape, shape.mixers[index]);
                index += 1;
                layer
            },
        )?;
        Ok(Self { shape, layers })
    }
}

/// A layer mixing by `mixer`, under `prefix`.
fn load_layer(
    store: &mut Store,
    prefix: &str,
    shape: &Shape,
    mixer: Mixer,
) -> Result<InterleavedLayer, bundle::Error> {
    let (hidden, eps) = (shape.hidden, shape.norm_eps);
    let mixing = match mixer {
        Mixer::Conv => {
            let conv = conv::Weights::load(store, prefix, hidden, shape.conv_kernel)?;
            Mixing::Conv(Box::new(conv))
        }
        Mixer::Full | Mixer::Window => {
            let attention = shape.attention(mixer);
            let attention = Weights::load(store, prefix, &attention, hidden, eps)?;
            Mixing::Attention(Box::new(attention))
        }
    };
    let mut scope = Scope::new(store, prefix);
    let mut norm = |name: &str| -> Result<Norm, bundle::Error> {
        Ok(Norm::Rms {
            weight: scope.load(name, hidden)?,
            eps,
        })
    };
    let (norm1, norm2) = (norm("input_norm")?, norm("post_norm")?);
    let sandwich = match shape.norms {
        Norms::Pre => None,
        Norms::Sandwich => Some(Sandwich {
            mixer: norm("mixer_out_norm")?,
            feed_forward: norm("feed_forward_out_norm")?,
        }),
    };
    let feed_forward =
        FeedForward::gated(&mut scope, hidden, shape.feed_forward, shape.activation)?;
    Ok(InterleavedLayer {
        norm1,
        mixing,
        norm2,
        feed_forward,
        sandwich,
    })
}
