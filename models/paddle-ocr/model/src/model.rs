//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `vision`, `projector`, `llm` — so this is the table of contents; the compiler
//! (`paddle-ocr-compile`) writes exactly these names.

use blocks::lm::{self, Embedding, Lm};
use blocks::norm::LayerNorm;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture};
use blocks::vit;
use blocks::vit::tower::{self, Ending, Tower};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::Linear;
use media::image::Filter;

use crate::Error;

/// Side of a square patch, in pixels.
pub const PATCH: usize = 14;
pub const CHANNELS: usize = 3;
/// Values of one patch: each channel's 14 × 14 pixels, row-major.
pub const PATCH_VALUES: usize = CHANNELS * PATCH * PATCH;
/// Patches merged into one language-model token along each axis.
pub const MERGE: usize = 2;
/// Side of the learned position grid: a 384-pixel image of 14-pixel patches.
pub const POSITION_GRID: usize = 27;

/// The vision tower: the patch embedding (a 14 × 14 convolution as one matrix), the learned
/// position grid, the encoder and its final norm.
pub(crate) const TOWER: tower::Shape = tower::Shape {
    encoder: VISION_SHAPE,
    patch_values: PATCH_VALUES,
    position_grid: POSITION_GRID,
    resample: Filter::Bilinear,
    ending: Ending::Norm,
};

pub const VISION_SHAPE: vit::Shape = vit::Shape {
    hidden: 1152,
    layers: 27,
    heads: 16,
    feed_forward: VISION_FEED_FORWARD,
    rope_base: 10_000.0,
    norm_eps: 1e-6,
};
/// SigLIP's 4304 MLP units, padded with zero units to whole quantization blocks
/// (`gelu(0) = 0`, so they add nothing).
pub const VISION_FEED_FORWARD: usize = 4320;
/// Width of a merged token: four patches side by side.
pub const MERGED: usize = VISION_SHAPE.hidden * MERGE * MERGE;
pub const PROJECTOR_NORM_EPS: f32 = 1e-5;

/// ERNIE 4.5 0.3B: a Qwen2-style layer without biases, its 16 query heads twice the width of
/// its residual stream, and multimodal RoPE over temporal, height and width sections.
pub const LANGUAGE_SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen2,
    hidden: 1024,
    layers: 18,
    heads: 16,
    kv_heads: 2,
    head_dim: 128,
    feed_forward: 3072,
    rope_base: 500_000.0,
    rotation: Rotation::Multimodal([16, 24, 24]),
    norm_eps: 1e-5,
};
pub const HIDDEN: usize = LANGUAGE_SHAPE.hidden;
pub const VOCABULARY: usize = 103_424;
pub(crate) const LANGUAGE: lm::Shape = lm::Shape {
    stack: LANGUAGE_SHAPE,
    vocabulary: VOCABULARY,
    embedding: Embedding::Own,
};

/// Four neighbouring patches into one token: LayerNorm, then `down(gelu(up(·)))` of the four
/// side by side.
pub(crate) struct Projector {
    pub(crate) norm: LayerNorm,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

pub struct Model {
    pub(crate) vision: Tower,
    pub(crate) projector: Projector,
    pub(crate) language: Lm,
    pub(crate) tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let vision = Tower::load(store, "vision", TOWER)?;
        let mut scope = Scope::new(store, "projector");
        let projector = Projector {
            norm: scope.load("norm", VISION_SHAPE.hidden)?,
            up: scope.load("up", [MERGED, MERGED])?,
            down: scope.load("down", [HIDDEN, MERGED])?,
        };
        let language = Lm::load(store, "llm", LANGUAGE)?;
        let tokenizer = Tokenizer::take(store)?;
        Ok(Self {
            vision,
            projector,
            language,
            tokenizer,
        })
    }
}
