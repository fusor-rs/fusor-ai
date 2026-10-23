//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `vision`, `visual`, `llm` — so this is the table of contents; the compiler
//! (`oxr-compile`) writes exactly these names.

use blocks::lm::{self, Embedding, Lm};
use blocks::norm::LayerNorm;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture};
use blocks::vit;
use blocks::vit::tower::{self, Ending, Resample, Tower};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::{Linear, Table};
use media::image::Filter;

use crate::Error;

/// Side of a square patch, in pixels.
pub const PATCH: usize = 16;
pub const CHANNELS: usize = 3;
/// Values of one patch: each channel's 16 × 16 pixels, row-major.
pub const PATCH_VALUES: usize = CHANNELS * PATCH * PATCH;
/// Patches merged into one language-model token along each axis.
pub const MERGE: usize = 2;
/// Side of the learned position grid: a 512-pixel image of 16-pixel patches.
pub const POSITION_GRID: usize = 32;

/// The vision tower: the patch embedding (a 16 × 16 convolution as one matrix), the learned
/// position grid and the encoder. The encoder's final norm is not part of it: the model reads
/// the last layer's output as it is.
pub(crate) const TOWER: tower::Shape = tower::Shape {
    encoder: VISION_SHAPE,
    patch_values: PATCH_VALUES,
    position_grid: POSITION_GRID,
    resample: Resample::Interpolate(Filter::Bicubic),
    ending: Ending::Encoder,
};

pub const VISION_SHAPE: vit::Shape = vit::Shape {
    hidden: 1152,
    layers: 27,
    heads: 16,
    feed_forward: VISION_FEED_FORWARD,
    rope_base: 10_000.0,
    norm_eps: 1e-6,
};
/// SigLIP2's 4304 MLP units, padded with zero units to whole quantization blocks
/// (`gelu(0) = 0`, so they add nothing).
pub const VISION_FEED_FORWARD: usize = 4320;
/// Width of a merged token: four encoded patches side by side.
pub const MERGED: usize = VISION_SHAPE.hidden * MERGE * MERGE;

/// The visual vocabulary: the words an image token is a distribution over, then the indicators
/// that open and close an image (and a video, unused here).
pub const VISUAL_VOCABULARY: usize = 65_536;
pub const INDICATORS: usize = 4;
pub const VISUAL_WORDS: usize = VISUAL_VOCABULARY - INDICATORS;
/// The indicators an image's tokens sit between, as rows of `Visual::indicators`.
pub const IMAGE_START: usize = 0;
pub const IMAGE_END: usize = 1;
/// Rows of `Visual::indicators`: the image's two.
pub const IMAGE_INDICATORS: usize = 2;
pub const VISUAL_NORM_EPS: f32 = 1e-5;

/// Qwen3 0.6B: a Qwen2-style layer without biases, its 16 query heads twice the width of its
/// residual stream, each query and key head RMS-normalized.
pub const LANGUAGE_SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen3,
    hidden: 1024,
    layers: 28,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    feed_forward: 3072,
    rope_base: 1_000_000.0,
    rotation: Rotation::Sequence,
    norm_eps: 1e-6,
};
pub const HIDDEN: usize = LANGUAGE_SHAPE.hidden;
pub const VOCABULARY: usize = 151_936;
/// Qwen3 0.6B ties its input embeddings to its head's rows.
pub(crate) const LANGUAGE: lm::Shape = lm::Shape {
    stack: LANGUAGE_SHAPE,
    vocabulary: VOCABULARY,
    embedding: Embedding::Tied,
};

/// Merged tokens into the language model: each one's logits over the visual words (`head`,
/// padded with zero rows to `VISUAL_VOCABULARY`), their LayerNorm and softmax, then the mixture
/// of the words' embeddings (`mix`, the embedding table transposed: `HIDDEN × VISUAL_VOCABULARY`);
/// and the embeddings of the image's indicators.
pub(crate) struct Visual {
    pub(crate) head: Linear,
    pub(crate) norm: LayerNorm,
    pub(crate) mix: Linear,
    pub(crate) indicators: Table,
}

pub struct Model {
    pub(crate) vision: Tower,
    pub(crate) visual: Visual,
    pub(crate) language: Lm,
    pub(crate) tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let vision = Tower::load(store, "vision", TOWER)?;
        let mut scope = Scope::new(store, "visual");
        let visual = Visual {
            head: scope.load("head", [VISUAL_VOCABULARY, MERGED])?,
            norm: scope.load("norm", VISUAL_WORDS)?,
            mix: scope.load("mix", [HIDDEN, VISUAL_VOCABULARY])?,
            indicators: scope.load("indicators", [IMAGE_INDICATORS, HIDDEN])?,
        };
        let language = Lm::load(store, "llm", LANGUAGE)?;
        let tokenizer = Tokenizer::take(store)?;
        Ok(Self {
            vision,
            visual,
            language,
            tokenizer,
        })
    }
}
