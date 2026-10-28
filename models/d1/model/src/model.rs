//! The model assembled from a loaded bundle, in either variant: the stack, its final norm and
//! the head load under `llm`, the vision tower and projector under `vision` and `projector`,
//! d1-Omni-2B's audio encoder under `audio` (`blocks::audio`'s names), beside the tokenizer,
//! the checkpoint's `config.json`, which names the variant and how d1 reads it, and its
//! `calibration.json`. The compiler (`d1-compile`) writes exactly these names.

use blocks::audio;
use blocks::conformer::subsampling;
use blocks::hybrid::{AttentionShape, Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Interleaved, Mixer, Norms, Window};
use blocks::rotary::Rotation;
use blocks::vit::interleaved::{self as vision, Tower};
use blocks::vit::merger::{self, Merger};
use bundle::{Load, Scope, Store};
use kernels::ops::Exponentials;
use kernels::{Linear, Vector};
use serde::Deserialize;

use crate::Error;

/// The bundle's copy of the checkpoint's `config.json`.
pub const CONFIG: &str = "config.json";
/// The bundle's copy of the checkpoint's `calibration.json`.
pub const CALIBRATION: &str = "calibration.json";
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

/// LFM3-VL's vision tower: each patch's two frames of 16 × 16 pixels, channel-major then
/// frame-major, embedded into 1,024 values, then 14 layers, gated 3 × 3 convolutions between
/// attention of 16 query heads over 8 key and value heads.
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

/// The projector: each 2 × 2 block of the tower's patches into one language-model token.
pub const PROJECTOR: merger::Shape = merger::Shape {
    patch: VISION_HIDDEN,
    merge: MERGE,
    hidden: HIDDEN,
    norm_eps: 1e-6,
};

/// Mel bins of the audio encoder's features.
pub const MELS: usize = 128;
const AUDIO_HIDDEN: usize = 768;
const AUDIO_HEAD_DIM: usize = 96;

/// The audio encoder's LFM2 `layer_types`: its attention layers see a window of 32 frames.
const AUDIO_MIXERS: [Mixer; 17] = {
    use Mixer::{Conv, Window};
    [
        Conv, Conv, Window, Conv, Conv, Window, Conv, Conv, Window, Conv, Conv, Window, Conv, Conv,
        Window, Window, Window,
    ]
};

/// d1-Omni-2B's audio encoder: three stride-2 convolutions of 256 channels with a SiLU after
/// each, eight mel frames into one of 768 values; 17 LFM2 layers, gated short convolutions of 3
/// taps between ungated attention of 8 query heads over 4 key and value heads of 96 over the last
/// 32 frames; then each frame into a language-model token.
pub const AUDIO: audio::Shape = audio::Shape {
    subsampling: subsampling::Shape {
        mels: MELS,
        channels: 256,
        kernel: 3,
        stride: 2,
        stages: 3,
        activation: subsampling::Activation::Silu,
        padding_frames: 0,
        hidden: AUDIO_HIDDEN,
    },
    stack: interleaved::Shape {
        hidden: AUDIO_HIDDEN,
        mixers: &AUDIO_MIXERS,
        attention: AttentionShape {
            heads: 8,
            kv_heads: 4,
            head_dim: AUDIO_HEAD_DIM,
            rotary_dim: AUDIO_HEAD_DIM,
            rope_base: 1_000_000.0,
            rotation: Rotation::Sequence,
            head_norm: HeadNorm::Rms,
            gate: Gate::None,
        },
        window: Window {
            rows: 32,
            rotary_dim: AUDIO_HEAD_DIM,
            rope_base: 1_000_000.0,
            rotation: Rotation::Sequence,
        },
        conv_kernel: 3,
        feed_forward: 1792,
        activation: Activation::Silu,
        norms: Norms::Pre,
        norm_eps: NORM_EPS,
    },
    hidden: HIDDEN,
    projector_eps: NORM_EPS,
};

/// The two variants of d1 this crate runs, on the same language model and vision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Variant {
    /// d1-2B, which reads text and images.
    #[serde(rename = "lfm3_vl")]
    Vision,
    /// d1-Omni-2B, which reads audio too.
    #[serde(rename = "lfm3_omni")]
    Omni,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Vision => "d1-2B",
            Variant::Omni => "d1-Omni-2B",
        }
    }
}

/// The option codes of a choice of more than 26 options (`d1_codes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "u8")]
pub enum Codes {
    /// `00`, `01`, … (scheme 1).
    Digits,
    /// `A` to `Z`, then `AA`, `AB`, … (scheme 2).
    Letters,
}

impl TryFrom<u8> for Codes {
    type Error = String;

    fn try_from(scheme: u8) -> Result<Self, String> {
        match scheme {
            1 => Ok(Codes::Digits),
            2 => Ok(Codes::Letters),
            other => Err(format!("d1_codes is 1 or 2, not {other}")),
        }
    }
}

/// How a box's edges are read off their distributions (`d1_box_edges`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Edges {
    /// The median bin's value.
    #[default]
    Median,
    /// Where the distribution's CDF crosses one half, inside the median's bin.
    Interp,
}

/// The prompt version a checkpoint trained with (`d1_prompt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(try_from = "u8")]
pub enum PromptVersion {
    /// Boxes questions say "16 at most" and read at most 16 blocks.
    #[default]
    First,
    /// Boxes questions name no count and read up to 64 blocks.
    Second,
}

impl TryFrom<u8> for PromptVersion {
    type Error = String;

    fn try_from(version: u8) -> Result<Self, String> {
        match version {
            1 => Ok(PromptVersion::First),
            2 => Ok(PromptVersion::Second),
            other => Err(format!("d1_prompt is 1 or 2, not {other}")),
        }
    }
}

/// The attention d1 reads its prompts under (`d1_attention`): this crate reads `set`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Attention {
    Set,
}

/// What the checkpoint's `config.json` holds that the model reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: Variant,
    pub d1_codes: Codes,
    pub d1_numbers: bool,
    pub d1_attention: Attention,
    /// The most questions read together after their state.
    pub d1_set_size: usize,
    /// The most pixels of an image, which is scaled down to it before its processor.
    pub d1_vision_max_pixels: usize,
    #[serde(default)]
    pub d1_box_edges: Edges,
    #[serde(default)]
    pub d1_prompt: PromptVersion,
}

impl Config {
    /// # Errors
    /// When `config` is not JSON, or not a d1 checkpoint's this crate runs.
    pub fn read(config: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(config)
            .map_err(|invalid| Error::Format(format!("{CONFIG}: {invalid}")))
    }
}

/// The vision tower and the projector of its output into the language model's tokens.
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
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let stack = Interleaved::load(store, "llm", LANGUAGE)?;
        let mut scope = Scope::new(store, "llm");
        let norm = scope.load("norm", HIDDEN)?;
        let head = scope.load("head", [VOCABULARY, HIDDEN])?;
        Ok(Self { stack, norm, head })
    }
}
