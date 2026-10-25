//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `vision`, `merger`, `llm` — beside the tokenizer; the compiler (`qwen3-5-compile`)
//! writes exactly these names.

use blocks::hybrid::{self, AttentionShape, DeltaShape, Hybrid};
use blocks::rotary::Rotation;
use blocks::vit;
use blocks::vit::merger::{self, Merger};
use blocks::vit::tower::{self, Ending, Resample, Tower};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::ops::Exponentials;
use kernels::{Linear, Vector};

use crate::Error;

/// Side of a square patch, in pixels.
pub const PATCH: usize = 16;
pub const CHANNELS: usize = 3;
/// Frames of a patch: a video's consecutive frames, or an image twice.
pub const FRAMES: usize = 2;
/// Values of one patch: each channel's frames' 16 × 16 pixels, channel-major, then frame-major.
pub const PATCH_VALUES: usize = CHANNELS * FRAMES * PATCH * PATCH;
/// Patches merged into one language-model token along each axis.
pub const MERGE: usize = 2;
/// Side of the learned position grid, in patches.
const POSITION_GRID: usize = 48;
pub const HIDDEN: usize = 2048;
pub const VOCABULARY: usize = 248_320;
pub(crate) const NORM_EPS: f32 = 1e-6;

const VISION: vit::Shape = vit::Shape {
    hidden: 1024,
    layers: 24,
    heads: 16,
    feed_forward: 4096,
    rope_base: 10_000.0,
    norm_eps: NORM_EPS,
    exponentials: Exponentials::Fast,
};

/// The vision tower: the patch embedding, the learned positions (resampled with the grid's
/// corners on the image's), and the encoder, read without a final norm.
pub const TOWER: tower::Shape = tower::Shape {
    encoder: VISION,
    patch_values: PATCH_VALUES,
    position_grid: POSITION_GRID,
    resample: Resample::Corners,
    ending: Ending::Encoder,
};

pub const MERGER: merger::Shape = merger::Shape {
    patch: VISION.hidden,
    merge: MERGE,
    hidden: HIDDEN,
    norm_eps: NORM_EPS,
};

/// Qwen3.5's stack: 24 layers, every fourth gated attention (8 query heads over 2 key and value
/// heads of 256, a quarter of each rotated by interleaved multimodal RoPE), the others the gated
/// delta rule (16 heads of 128).
pub const LANGUAGE: hybrid::Shape = hybrid::Shape {
    hidden: HIDDEN,
    layers: 24,
    attention_interval: 4,
    attention: AttentionShape {
        heads: 8,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        rope_base: 10_000_000.0,
        rotation: Rotation::MultimodalInterleaved([11, 11, 10]),
    },
    delta: DeltaShape {
        key_heads: 16,
        value_heads: 16,
        key_dim: 128,
        value_dim: 128,
        kernel: 4,
    },
    feed_forward: 6144,
    norm_eps: NORM_EPS,
};

/// The language model: the stack, its final norm, and the head over the vocabulary, whose rows
/// are the input embeddings too.
pub(crate) struct Language {
    pub(crate) stack: Hybrid,
    pub(crate) norm: Vector,
    pub(crate) head: Linear,
}

impl Language {
    /// The embedding of `id` into `out`.
    pub(crate) fn embed(&self, id: u32, out: &mut [f32]) {
        self.head.matrix().dequantize_row(id as usize, out);
    }
}

pub struct Model {
    pub(crate) vision: Tower,
    pub(crate) merger: Merger,
    pub(crate) language: Language,
    pub(crate) tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let vision = Tower::load(store, "vision", TOWER)?;
        let merger = Merger::load(store, "merger", MERGER)?;
        let stack = Hybrid::load(store, "llm", LANGUAGE)?;
        let mut scope = Scope::new(store, "llm");
        let language = Language {
            stack,
            norm: scope.load("norm", HIDDEN)?,
            head: scope.load("head", [VOCABULARY, HIDDEN])?,
        };
        let tokenizer = Tokenizer::take(store)?;
        Ok(Self {
            vision,
            merger,
            language,
            tokenizer,
        })
    }
}
