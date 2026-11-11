//! The model's dimensions and tensor names. The text bundle holds Qwen3-VL's language model
//! (`TEXT`) and the tokenizer; each transformer bundle holds one of the four transformers
//! (`DIT`), a `variant.json` naming it and, for depth and the restorer, the empty prompt's
//! states (`EMPTY_PROMPT`). The compiler (`iris-compile`) writes exactly these names.

use blocks::dit;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture};
use media::image::Size;
use serde::{Deserialize, Serialize};

use crate::Error;

/// The text bundle's directory under a bundle root; the transformers' are `Variant::name`.
pub const TEXT_BUNDLE: &str = "text";
pub const TEXT: &str = "text";
pub const DIT: &str = "dit";
pub const EMPTY_PROMPT: &str = "empty_prompt";
pub const VARIANT_PART: &str = "variant.json";

/// The width of Qwen3-VL's states and of the transformer.
pub const HIDDEN: usize = 2560;
pub const VOCABULARY: usize = 151_936;
/// The text layers whose states the transformer reads: the state after that many layers.
pub const TEXT_LAYERS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
/// Qwen3-VL's language model up to the last layer read (layer 36 and the final norm feed
/// nothing).
pub const TEXT_SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen3,
    hidden: HIDDEN,
    layers: TEXT_LAYERS[TEXT_LAYERS.len() - 1],
    heads: 32,
    kv_heads: 8,
    head_dim: 128,
    feed_forward: 9728,
    rope_base: 5_000_000.0,
    rotation: Rotation::Sequence,
    norm_eps: NORM_EPS,
};
/// Rows of text the transformer reads: the caption and the template's suffix, then zero rows.
pub const TEXT_ROWS: usize = 300;
/// The SwiGLU width, `int(2·int(2560·4)/3)` = 6826 in the checkpoint, padded with zero rows and
/// columns to whole quantization blocks.
pub const FEED_FORWARD: usize = 6848;
/// The two layer-wise blocks over each token's `TEXT_LAYERS` states.
pub const ADAPTER_SHAPE: dit::Shape = dit::Shape {
    hidden: HIDDEN,
    heads: 32,
    head_dim: 80,
    feed_forward: 3328,
    depth: 2,
    norm_eps: NORM_EPS,
    kind: dit::Kind::Plain,
};
/// The text refiner's two layers: Qwen3's without rotary embeddings, as `Partial(0)` turns none
/// of the frequencies.
pub const REFINER_SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen3,
    hidden: HIDDEN,
    layers: 2,
    heads: 20,
    kv_heads: 20,
    head_dim: 128,
    feed_forward: FEED_FORWARD,
    rope_base: 10_000.0,
    rotation: Rotation::Partial(0),
    norm_eps: NORM_EPS,
};
pub(crate) const NORM_EPS: f32 = 1e-6;

/// The trunk's blocks: the first with a stream of their own for the text and one for the image,
/// the others with one set of weights for all rows.
pub const DUAL_BLOCKS: usize = 8;
pub const SINGLE_BLOCKS: usize = 16;

/// The pixel head's blocks, over this many channels per pixel.
pub const PIXEL_BLOCKS: usize = 4;
pub const PIXEL_HIDDEN: usize = 16;

/// Pixels per patch along each side.
pub const PATCH: usize = 16;
pub const PATCH_PIXELS: usize = PATCH * PATCH;
pub const IMAGE_CHANNELS: usize = 3;
pub(crate) const RGBA_CHANNELS: usize = 4;
/// The sizes Iris-3B and Turbo were trained to generate: about one megapixel in seven aspect
/// ratios.
pub const SIZES: [Size; 7] = [
    Size {
        width: 1024,
        height: 1024,
    },
    Size {
        width: 752,
        height: 1344,
    },
    Size {
        width: 1344,
        height: 752,
    },
    Size {
        width: 896,
        height: 1344,
    },
    Size {
        width: 1344,
        height: 896,
    },
    Size {
        width: 896,
        height: 1200,
    },
    Size {
        width: 1200,
        height: 896,
    },
];
/// The largest image the buffers hold: the largest of `SIZES`, its pixels and its sides.
pub(crate) const MAX_PIXELS: usize = 896 * 1344;
pub(crate) const MAX_SIDE: usize = 1344;

/// The noise schedule the transformer was trained on: timesteps over flow-matching levels.
pub const TRAINING_TIMESTEPS: usize = 1000;
/// The base sampler's largest level: the training schedule's, `1 − 0.001`.
pub const LARGEST_LEVEL: f64 = 0.999;
/// The base model's defaults (its card's `--steps` and `--cfg-scale`).
pub const DEFAULT_STEPS: usize = 100;
pub const DEFAULT_GUIDANCE: f32 = 3.0;
pub const DEFAULT_SHIFT: f32 = 4.0;
/// The timestep sinusoids' period (`model.timestep_max_period`).
pub const TIMESTEP_PERIOD: f32 = 10.0;
/// Turbo's levels (`turbo/distill.yaml`), then 0.
pub const TURBO_LEVELS: [f32; 5] = [1.0, 0.938, 0.833, 0.625, 0.0];
/// The depth model's one pass is at the last training timestep.
pub const DEPTH_TIME: f32 = 1000.0;
/// The restorer's level (`upscaler/config.yaml`'s `task.sigma`) and tile side.
pub const RESTORE_LEVEL: f32 = 0.5;
pub const TILE: usize = 1024;

/// The transformers a checkpoint ships, each its own bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    Base,
    Turbo,
    Depth,
    Upscaler,
}

impl Variant {
    /// Its bundle's directory under a bundle root, and its name in `variant.json`.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Base => "base",
            Variant::Turbo => "turbo",
            Variant::Depth => "depth",
            Variant::Upscaler => "upscaler",
        }
    }

    /// Channels of a pixel it predicts: depth's reducer is folded into its last layer.
    pub fn outputs(self) -> usize {
        match self {
            Variant::Depth => 1,
            _ => IMAGE_CHANNELS,
        }
    }

    /// Whether its bundle holds the empty prompt's states: the model reads no prompt of its own.
    pub fn has_empty_prompt(self) -> bool {
        matches!(self, Variant::Depth | Variant::Upscaler)
    }

    pub(crate) fn read(blob: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(blob)
            .map_err(|invalid| Error::Format(format!("{VARIANT_PART}: {invalid}")))
    }
}
