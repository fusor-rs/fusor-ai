//! The model assembled from a loaded bundle: the stack, its final norm and the head load under
//! `llm`, beside the tokenizer; the compiler (`spark-x2-6-compile`) writes exactly these names.

use blocks::hybrid::{AttentionShape, Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Interleaved, Mixer, Norms, Window};
use blocks::rotary::Rotation;
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::{Linear, Vector};

use crate::Error;

pub const HIDDEN: usize = 2048;
pub const VOCABULARY: usize = 131_072;
pub(crate) const NORM_EPS: f32 = 1e-6;
const LAYERS: usize = 28;
/// The last of every this many layers attends to every earlier row.
const GLOBAL_INTERVAL: usize = 4;

const MIXERS: [Mixer; LAYERS] = {
    let mut mixers = [Mixer::Window; LAYERS];
    let mut layer = GLOBAL_INTERVAL - 1;
    while layer < LAYERS {
        mixers[layer] = Mixer::Full;
        layer += GLOBAL_INTERVAL;
    }
    mixers
};

/// Spark's stack: 28 layers, every fourth full attention, the others over a window of 512 rows;
/// 8 query heads over 2 key and value heads of 256, each head's output gated by one sigmoid, a
/// quarter of each full attention head rotated, every element of a window head.
pub const LANGUAGE: interleaved::Shape = interleaved::Shape {
    hidden: HIDDEN,
    mixers: &MIXERS,
    attention: AttentionShape {
        heads: 8,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        rope_base: 5_000_000.0,
        rotation: Rotation::Sequence,
        head_norm: HeadNorm::None,
        gate: Gate::Headwise,
    },
    window: Window {
        rows: 512,
        rotary_dim: 256,
        rope_base: 10_000.0,
        rotation: Rotation::Sequence,
    },
    conv_kernel: 0,
    feed_forward: 6656,
    activation: Activation::GeluErf,
    norms: Norms::Pre,
    norm_eps: NORM_EPS,
};

/// The language model: the stack, its final norm, and the head over the vocabulary, whose rows
/// are the input embeddings too.
pub struct Model {
    pub(crate) stack: Interleaved,
    pub(crate) norm: Vector,
    pub(crate) head: Linear,
    pub(crate) tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let stack = Interleaved::load(store, "llm", LANGUAGE)?;
        let mut scope = Scope::new(store, "llm");
        let norm = scope.load("norm", HIDDEN)?;
        let head = scope.load("head", [VOCABULARY, HIDDEN])?;
        Ok(Self {
            stack,
            norm,
            head,
            tokenizer: Tokenizer::take(store)?,
        })
    }

    /// The embedding of `id` into `out`.
    pub(crate) fn embed(&self, id: u32, out: &mut [f32]) {
        self.head.matrix().dequantize_row(id as usize, out);
    }
}
