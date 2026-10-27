//! The model assembled from a loaded bundle, in either size. Every component loads itself
//! (`bundle::Load`) under its prefix — `vision`, `merger`, `llm` — beside the answer-code
//! `readout` and the checkpoint's `decision_config.json`, which names the size; the compiler
//! (`d3-compile`) writes exactly these names.

use blocks::hybrid::{self, AttentionShape, DeltaShape, Gate, HeadNorm, Hybrid};
use blocks::rotary::Rotation;
use blocks::vit;
use blocks::vit::merger::{self, Merger};
use blocks::vit::tower::{self, Ending, Resample, Tower};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::ops::Exponentials;
use kernels::{Table, Vector};
use serde::Deserialize;

use crate::Error;

/// The bundle's copy of the checkpoint's decision configuration.
pub const DECISION_CONFIG: &str = "decision_config.json";
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
pub const VOCABULARY: usize = 248_320;
/// Answer codes the readout scores: an option's code is its index.
pub const CODES: usize = 255;
const NORM_EPS: f32 = 1e-6;

/// The two sizes of Decision 3.0 this crate runs, on Qwen3.5's 2B and 0.8B backbones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Nano,
    Lite,
}

impl Size {
    pub const ALL: [Size; 2] = [Size::Nano, Size::Lite];

    pub fn name(self) -> &'static str {
        match self {
            Size::Nano => "nano",
            Size::Lite => "lite",
        }
    }

    /// The backbone `decision_config.json` names, which tells the sizes apart.
    fn base_model(self) -> &'static str {
        match self {
            Size::Nano => "Qwen/Qwen3.5-2B",
            Size::Lite => "Qwen/Qwen3.5-0.8B",
        }
    }

    pub fn hidden(self) -> usize {
        match self {
            Size::Nano => 2048,
            Size::Lite => 1024,
        }
    }

    pub fn vision(self) -> vit::Shape {
        match self {
            Size::Nano => vit::Shape {
                hidden: 1024,
                layers: 24,
                heads: 16,
                feed_forward: 4096,
                rope_base: 10_000.0,
                norm_eps: NORM_EPS,
                exponentials: Exponentials::Exact,
            },
            Size::Lite => vit::Shape {
                hidden: 768,
                layers: 12,
                heads: 12,
                feed_forward: 3072,
                rope_base: 10_000.0,
                norm_eps: NORM_EPS,
                exponentials: Exponentials::Exact,
            },
        }
    }

    /// The vision tower: the patch embedding, the learned positions (resampled with the grid's
    /// corners on the image's), and the encoder, read without a final norm.
    pub fn tower(self) -> tower::Shape {
        tower::Shape {
            encoder: self.vision(),
            patch_values: PATCH_VALUES,
            position_grid: POSITION_GRID,
            resample: Resample::Corners,
            ending: Ending::Encoder,
        }
    }

    pub fn merger(self) -> merger::Shape {
        merger::Shape {
            patch: self.vision().hidden,
            merge: MERGE,
            hidden: self.hidden(),
            norm_eps: NORM_EPS,
        }
    }

    /// Qwen3.5's stack: 24 layers, every fourth gated attention (8 query heads over 2 key and
    /// value heads of 256, a quarter of each rotated by interleaved multimodal RoPE), the others
    /// the gated delta rule (16 heads of 128).
    pub fn language(self) -> hybrid::Shape {
        hybrid::Shape {
            hidden: self.hidden(),
            layers: 24,
            attention_interval: 4,
            attention: AttentionShape {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                rope_base: 10_000_000.0,
                rotation: Rotation::MultimodalInterleaved([11, 11, 10]),
                head_norm: HeadNorm::Rms,
                gate: Gate::Elementwise,
            },
            delta: DeltaShape {
                key_heads: 16,
                value_heads: 16,
                key_dim: 128,
                value_dim: 128,
                kernel: 4,
            },
            feed_forward: match self {
                Size::Nano => 6144,
                Size::Lite => 3584,
            },
            norm_eps: NORM_EPS,
        }
    }
}

/// What the checkpoint's `decision_config.json` holds that the model reads: its backbone, the
/// answer codes in order and the softmax temperature.
#[derive(Debug, Clone, Deserialize)]
pub struct Decision {
    base_model: String,
    pub(crate) codes: Vec<String>,
    pub(crate) temperature: f32,
}

impl Decision {
    pub fn size(&self) -> Result<Size, Error> {
        let base = &self.base_model;
        Size::ALL
            .into_iter()
            .find(|size| size.base_model() == base)
            .ok_or_else(|| Error::Format(format!("{DECISION_CONFIG}: no d3 size on {base}")))
    }
}

/// The language model: the token embeddings, the stack and its final norm.
pub(crate) struct Language {
    pub(crate) embed: Table,
    pub(crate) stack: Hybrid,
    pub(crate) norm: Vector,
}

pub struct Model {
    pub(crate) size: Size,
    pub(crate) decision: Decision,
    pub(crate) vision: Tower,
    pub(crate) merger: Merger,
    pub(crate) language: Language,
    /// The answer codes' rows, `CODES × hidden`: a code's logit is its row times the last row's
    /// final-normed state.
    pub(crate) readout: Table,
    pub(crate) tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let config = store.take_blob(DECISION_CONFIG)?;
        let decision: Decision = serde_json::from_slice(&config)
            .map_err(|invalid| Error::Format(format!("{DECISION_CONFIG}: {invalid}")))?;
        if decision.codes.len() != CODES {
            return Err(Error::Format(format!(
                "{DECISION_CONFIG}: {} answer codes, the readout's {CODES}",
                decision.codes.len()
            )));
        }
        let size = decision.size()?;
        let hidden = size.hidden();
        let vision = Tower::load(store, "vision", size.tower())?;
        let merger = Merger::load(store, "merger", size.merger())?;
        let stack = Hybrid::load(store, "llm", size.language())?;
        let mut scope = Scope::new(store, "llm");
        let language = Language {
            embed: scope.load("embed", [VOCABULARY, hidden])?,
            stack,
            norm: scope.load("norm", hidden)?,
        };
        let readout = Table::load(store, "readout", [CODES, hidden])?;
        let tokenizer = Tokenizer::take(store)?;
        Ok(Self {
            size,
            decision,
            vision,
            merger,
            language,
            readout,
            tokenizer,
        })
    }
}
