//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `vision`, `projector`, `llm` — so this is the table of contents; the compiler
//! (`paddle-ocr-compile`) writes exactly these names.

use blocks::norm::LayerNorm;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture, Transformer};
use blocks::vit::{self, Vit};
use bundle::{Load, Scope, Store};
use kernels::{Linear, Table, Vector};

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

/// The vision tower: the patch embedding (a 14 × 14 convolution as one matrix), the learned
/// position grid, the encoder and its final norm.
pub(crate) struct Vision {
    pub(crate) patch: Linear,
    pub(crate) positions: Vec<f32>,
    pub(crate) encoder: Vit,
    pub(crate) norm: LayerNorm,
}

/// Four neighbouring patches into one token: LayerNorm, then `down(gelu(up(·)))` of the four
/// side by side.
pub(crate) struct Projector {
    pub(crate) norm: LayerNorm,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

pub(crate) struct Language {
    pub(crate) embed: Table,
    pub(crate) body: Transformer,
    pub(crate) norm: Vector,
    pub(crate) head: Linear,
}

pub(crate) struct Model {
    pub(crate) vision: Vision,
    pub(crate) projector: Projector,
    pub(crate) language: Language,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let width = VISION_SHAPE.hidden;
        let positions: Table = Table::load(
            store,
            "vision.positions",
            [POSITION_GRID * POSITION_GRID, width],
        )?;
        let vision = Vision {
            patch: Linear::load(store, "vision.patch", [width, PATCH_VALUES])?,
            positions: positions.dequantize(),
            encoder: Vit::load(store, "vision", VISION_SHAPE)?,
            norm: LayerNorm::load(store, "vision.norm", width)?,
        };
        let mut scope = Scope::new(store, "projector");
        let projector = Projector {
            norm: scope.load("norm", width)?,
            up: scope.load("up", [MERGED, MERGED])?,
            down: scope.load("down", [HIDDEN, MERGED])?,
        };
        let mut scope = Scope::new(store, "llm");
        let language = Language {
            embed: scope.load("embed", [VOCABULARY, HIDDEN])?,
            norm: scope.load("norm", HIDDEN)?,
            head: scope.load("head", [VOCABULARY, HIDDEN])?,
            body: Transformer::load(store, "llm", LANGUAGE_SHAPE)?,
        };
        Ok(Self {
            vision,
            projector,
            language,
        })
    }
}
