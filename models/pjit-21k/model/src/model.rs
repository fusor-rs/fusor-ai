//! The model's dimensions and tensor names. The text bundle holds Qwen3.5-4B's embeddings and
//! first layers (`TEXT`) and the tokenizer; the image bundle holds P-JiT 21K's transformer
//! (`IMAGE`). The compiler (`pjit-21k-compile`) writes exactly these names.

use blocks::hybrid::{self, AttentionShape, DeltaShape, Gate, HeadNorm, Hybrid};
use blocks::rotary::Rotation;
use blocks::{apg, pyramid_jit};
use bundle::{Load, Scope, Store};
use kernels::Table;

use crate::Error;

/// The bundles' directories under a bundle root.
pub const TEXT_BUNDLE: &str = "text";
pub const IMAGE_BUNDLE: &str = "image";
/// Tensor prefixes.
pub const TEXT: &str = "text";
pub const IMAGE: &str = "image";
pub const HIDDEN: usize = 2560;
pub const VOCABULARY: usize = 248_320;
/// The text's states, side by side: the residual stream after that many decoder layers (after
/// layers 7, 15 and 27, the reference's full-attention `extraction_layers`).
pub const TEXT_LAYERS: [usize; 3] = [8, 16, 28];
/// Tokens the text encoder reads: the templated caption's first.
pub const TEXT_TOKENS: usize = 256;
/// Width of a text state.
pub const TEXT_WIDTH: usize = TEXT_LAYERS.len() * HIDDEN;
const NORM_EPS: f32 = 1e-6;
/// Qwen3.5-4B's stack through its last extracted layer: every fourth layer gated attention (16
/// query heads over 4 key and value heads of 256, a quarter of each rotated by interleaved
/// multimodal RoPE), the others the gated delta rule (16 key heads, 32 value heads, 128 wide).
pub const LANGUAGE: hybrid::Shape = hybrid::Shape {
    hidden: HIDDEN,
    layers: TEXT_LAYERS[TEXT_LAYERS.len() - 1],
    attention_interval: 4,
    attention: AttentionShape {
        heads: 16,
        kv_heads: 4,
        head_dim: 256,
        rotary_dim: 64,
        rope_base: 10_000_000.0,
        rotation: Rotation::MultimodalInterleaved([11, 11, 10]),
        head_norm: HeadNorm::Rms,
        gate: Gate::Elementwise,
    },
    delta: DeltaShape {
        key_heads: 16,
        value_heads: 32,
        key_dim: 128,
        value_dim: 128,
        kernel: 4,
    },
    feed_forward: 9216,
    norm_eps: NORM_EPS,
};
pub const CHANNELS: usize = 3;
/// The sizes P-JiT 21K was trained at, width by height (`ASPECT_RATIOS` of the reference's
/// `config.py`: square, tall, 4:3, 3:2 and 16:9), the only ones a generation takes.
pub(crate) const SIZES: [(usize, usize); 5] = [
    (1024, 1024),
    (864, 1216),
    (1152, 864),
    (1248, 832),
    (1280, 720),
];
/// P-JiT 21K's transformer. Its SwiGLU widths are `ffn_dim`'s two thirds rounded up to 256
/// (7936 → 5376, 4096 → 2816).
pub const DIT: pyramid_jit::Shape = pyramid_jit::Shape {
    hidden: 2944,
    heads: 23,
    depth: 22,
    feed_forward: 5376,
    refiners: 2,
    refiner_feed_forward: 2816,
    rank: 256,
    text: TEXT_WIDTH,
    text_tokens: TEXT_TOKENS,
    channels: CHANNELS,
    patch: 32,
    bottleneck: 1024,
    frequencies: 256,
    rope_sections: [22, 21, 21],
    rope_base: 10_000.0,
    rope_spatial_scale: 0.5,
    eps: 1e-6,
};
/// The sampler starts from this many standard normals (the 1024-pixel noise scale).
pub const NOISE_SCALE: f32 = 4.0;
pub const TRAINING_TIMESTEPS: usize = 1000;
/// The reference's default negative prompt: none.
pub const NEGATIVE: &str = "";
/// The reference's defaults: 50 steps at guidance 15.
pub const DEFAULT_STEPS: usize = 50;
pub const DEFAULT_GUIDANCE: f32 = 15.0;
/// Adaptive projected guidance's defaults (config.json's sampler).
pub const GUIDANCE: apg::Settings = apg::Settings {
    scale: DEFAULT_GUIDANCE,
    momentum: -0.75,
    rescale: 10.0,
    eta: 0.0,
};

/// Qwen3.5-4B's token embeddings and its stack through `TEXT_LAYERS`' last.
pub(crate) struct Language {
    pub(crate) embedding: Table,
    pub(crate) stack: Hybrid,
}

impl Language {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let embedding = Scope::new(store, TEXT).load("embedding", [VOCABULARY, HIDDEN])?;
        let stack = Hybrid::load(store, TEXT, LANGUAGE)?;
        Ok(Self { embedding, stack })
    }
}
