//! The model assembled from a loaded bundle, in any variant: the stack, its final norm and the
//! head load under `llm`, beside the tokenizer and the checkpoint's `config.json`, which names the
//! variant; the vision tower and projector of LFM3-VL-2B and LFM3-Omni-2B load under `vision` and
//! `projector`, the audio of LFM3-Audio-2B and LFM3-Omni-2B as `audio` lists. The compiler
//! (`lfm3-compile`) writes exactly these names.

use blocks::hybrid::{AttentionShape, Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Interleaved, Mixer, Norms, Window};
use blocks::rotary::Rotation;
use blocks::vit::interleaved::{self as vision, Tower};
use blocks::vit::merger::{self, Merger};
use bundle::{Load, Scope, Store, Tokenizer};
use kernels::ops::Exponentials;
use kernels::{Linear, Vector};
use serde::Deserialize;

use crate::Error;
use crate::prompt::Template;

/// The bundle's copy of the checkpoint's `config.json`, whose `model_type` names the variant.
pub const CONFIG: &str = "config.json";
pub const HIDDEN: usize = 2048;
pub const VOCABULARY: usize = 128_000;
pub(crate) const NORM_EPS: f32 = 1e-5;
const HEAD_DIM: usize = 128;

/// The checkpoint's `layer_types`.
const MIXERS: [Mixer; 24] = {
    use Mixer::{Conv, Full, Window};
    [
        Conv, Conv, Full, Conv, Window, Conv, Full, Conv, Window, Conv, Full, Conv, Window, Conv,
        Full, Conv, Window, Conv, Full, Conv, Window, Conv, Full, Conv,
    ]
};

/// LFM3's stack: 24 layers, gated short convolutions of 7 taps between full attention and
/// attention over a window of 256 rows; 16 query heads over 4 key and value heads of 128, each
/// head RMS-normed and its output gated by one sigmoid; the first quarter of each half of a full
/// attention head rotated, every element of a window head; each sublayer's output normed.
pub const LANGUAGE: interleaved::Shape = interleaved::Shape {
    hidden: HIDDEN,
    mixers: &MIXERS,
    attention: AttentionShape {
        heads: 16,
        kv_heads: 4,
        head_dim: HEAD_DIM,
        rotary_dim: HEAD_DIM,
        rope_base: 1_000_000.0,
        rotation: Rotation::Partial(HEAD_DIM / 4),
        head_norm: HeadNorm::Rms,
        gate: Gate::Headwise,
    },
    window: Window {
        rows: 256,
        rotary_dim: HEAD_DIM,
        rope_base: 10_000.0,
        rotation: Rotation::Sequence,
    },
    conv_kernel: 7,
    feed_forward: 7680,
    activation: Activation::Silu,
    norms: Norms::Sandwich,
    norm_eps: NORM_EPS,
};

/// Side of a square patch, in pixels.
pub const PATCH: usize = 16;
const CHANNELS: usize = 3;
/// Frames of a patch: an image fills both.
const FRAMES: usize = 2;
/// Patches merged into one language-model token along each axis.
pub const MERGE: usize = 2;
const VISION_HIDDEN: usize = 1024;

/// The vision tower's `layer_types`.
const VISION_MIXERS: [vision::Mixer; 14] = {
    use vision::Mixer::{Attention, Conv};
    [
        Conv, Conv, Attention, Conv, Attention, Conv, Attention, Conv, Attention, Conv, Attention,
        Conv, Attention, Conv,
    ]
};

/// The vision tower of LFM3-VL-2B and LFM3-Omni-2B: each patch's two frames of 16 × 16 pixels,
/// channel-major then frame-major, embedded into 1,024 values, then 14 layers, gated 3 × 3
/// convolutions between attention of 16 query heads over 8 key and value heads.
pub const TOWER: vision::Shape = vision::Shape {
    hidden: VISION_HIDDEN,
    mixers: &VISION_MIXERS,
    heads: 16,
    kv_heads: 8,
    feed_forward: 2560,
    kernel: 3,
    rope_base: 1_000_000.0,
    norm_eps: NORM_EPS,
    patch_values: CHANNELS * FRAMES * PATCH * PATCH,
    exponentials: Exponentials::Exact,
};

/// Their projector: each 2 × 2 block of the tower's patches into one language-model token.
pub const PROJECTOR: merger::Shape = merger::Shape {
    patch: VISION_HIDDEN,
    merge: MERGE,
    hidden: HIDDEN,
    norm_eps: 1e-6,
};

/// The variants of LFM3 this crate runs, on the same language model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Variant {
    /// LFM3-1.7B, which reads text.
    #[serde(rename = "lfm3")]
    Text,
    /// LFM3-VL-2B, which reads images too.
    #[serde(rename = "lfm3_vl")]
    Vision,
    /// LFM3-Audio-2B, which hears speech and speaks too.
    #[serde(rename = "lfm3_audio")]
    Audio,
    /// LFM3-Omni-2B, which reads images, hears speech and speaks.
    #[serde(rename = "lfm3_omni")]
    Omni,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "LFM3-1.7B",
            Self::Vision => "LFM3-VL-2B",
            Self::Audio => "LFM3-Audio-2B",
            Self::Omni => "LFM3-Omni-2B",
        }
    }

    pub fn sees(self) -> bool {
        matches!(self, Self::Vision | Self::Omni)
    }

    /// Whether the variant hears speech, and speaks.
    pub fn hears(self) -> bool {
        matches!(self, Self::Audio | Self::Omni)
    }

    /// How its chat template writes a conversation.
    pub fn template(self) -> Template {
        match self {
            Self::Audio => Template::Audio,
            Self::Text | Self::Vision | Self::Omni => Template::Lfm3,
        }
    }
}

/// What the checkpoint's `config.json` holds that the model reads: the variant it names.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: Variant,
}

impl Config {
    /// # Errors
    /// When `config` is not JSON, or names no variant of LFM3 this crate runs.
    pub fn read(config: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(config)
            .map_err(|invalid| Error::Format(format!("{CONFIG}: {invalid}")))
    }
}

/// The vision of LFM3-VL-2B and LFM3-Omni-2B: the tower and the projector of its output into the
/// language model's tokens.
pub struct Vision {
    pub(crate) tower: Tower,
    pub(crate) projector: Merger,
}

impl Vision {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            tower: Tower::load(store, "vision", TOWER)?,
            projector: Merger::load(store, "projector", PROJECTOR)?,
        })
    }
}

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
