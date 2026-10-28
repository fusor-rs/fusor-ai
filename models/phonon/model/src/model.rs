//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `lm`, `flow`, `mimi`, `voice` — so this is the table of contents; the compiler
//! (`phonon-compile`) writes exactly these names.

use blocks::adaln_mlp::{self, AdalnMlp};
use blocks::norm::LayerNorm;
use blocks::rotary::Rotation;
use blocks::seanet::{self, Direction, Seanet};
use blocks::transformer::{self, Architecture, Transformer};
use bundle::{Load, Store};
use kernels::{Linear, Matrix, Table, Vector};

use crate::Error;
use crate::prompt::VOICES;

pub const SAMPLE_RATE: u32 = 24_000;
/// Samples per 80 ms latent frame.
pub const FRAME_SAMPLES: usize = 1920;
/// Width of the flow LM, of its text embeddings and of the Mimi decoder's transformer.
pub const HIDDEN: usize = 512;
/// One frame's latent.
pub const LATENT: usize = 32;
/// Rows of the text embedding table (the last is the unused classifier-free-guidance row).
const TEXT_ROWS: usize = 4001;
/// Embedded latents of a voice (10 s at 12.5 Hz).
pub const VOICE_FRAMES: usize = 125;

pub const FLOW_LM_SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Kyutai,
    hidden: HIDDEN,
    layers: 6,
    heads: 8,
    kv_heads: 8,
    head_dim: 64,
    feed_forward: 2048,
    rope_base: 1e4,
    rotation: Rotation::Sequence,
    norm_eps: 1e-5,
};
/// The flow LM's final norm.
pub(crate) const OUT_NORM_EPS: f32 = 1e-5;

pub const FLOW_SHAPE: adaln_mlp::Shape = adaln_mlp::Shape {
    condition: HIDDEN,
    latent: LATENT,
    width: 512,
    depth: 6,
    norm_eps: 1e-6,
};

pub const MIMI_TRANSFORMER_SHAPE: transformer::Shape = transformer::Shape {
    layers: 2,
    ..FLOW_LM_SHAPE
};
/// Keys a Mimi transformer query sees: itself and the 250 positions before it.
pub(crate) const MIMI_WINDOW: usize = 251;
/// Mimi transformer positions per latent frame (12.5 Hz → 200 Hz), the depthwise upsampling's
/// stride; its kernel is twice that.
pub(crate) const UPSAMPLE: usize = 16;

pub const SEANET_SHAPE: seanet::Shape = seanet::Shape {
    direction: Direction::Decode,
    dimension: HIDDEN,
    filters: 64,
    ratios: &[6, 5, 4],
    kernel: 7,
    last_kernel: 3,
    residual_kernel: 3,
    compress: 2,
    channels: 1,
};

pub(crate) struct FlowLm {
    embed: Table,
    pub(crate) body: Transformer,
    /// `latent → hidden`, its bias the summed conditions (one speaker, rate unchanged).
    pub(crate) input: Linear,
    pub(crate) bos: Vector,
    pub(crate) norm: LayerNorm,
    pub(crate) eos: Linear,
    pub(crate) flow: AdalnMlp,
}

impl FlowLm {
    pub(crate) fn embeddings(&self) -> Matrix<'_> {
        self.embed.matrix()
    }
}

pub(crate) struct Mimi {
    /// The denormalization and the quantizer's output projection, `latent → hidden`.
    pub(crate) latent: Linear,
    /// The depthwise upsampling kernel, `[hidden, 2 · UPSAMPLE]`.
    pub(crate) upsample: Vec<f32>,
    pub(crate) transformer: Transformer,
    pub(crate) decoder: Seanet,
}

pub(crate) struct Model {
    pub(crate) lm: FlowLm,
    pub(crate) mimi: Mimi,
    /// Each voice of `VOICES`: `[VOICE_FRAMES, HIDDEN]` embedded latents.
    voices: Vec<Table>,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let lm = FlowLm {
            embed: Table::load(store, "lm.embed", [TEXT_ROWS, HIDDEN])?,
            body: Transformer::load(store, "lm", FLOW_LM_SHAPE)?,
            input: Linear::load(store, "lm.input", [HIDDEN, LATENT])?,
            bos: Vector::load(store, "lm.bos", LATENT)?,
            norm: LayerNorm::load(store, "lm.norm", HIDDEN)?,
            eos: Linear::load(store, "lm.eos", [1, HIDDEN])?,
            flow: AdalnMlp::load(store, "flow", FLOW_SHAPE)?,
        };
        let upsample = Table::load(store, "mimi.upsample", [HIDDEN, 2 * UPSAMPLE])?;
        let mimi = Mimi {
            latent: Linear::load(store, "mimi.latent", [HIDDEN, LATENT])?,
            upsample: upsample.dequantize(),
            transformer: Transformer::load(store, "mimi", MIMI_TRANSFORMER_SHAPE)?,
            decoder: Seanet::load(store, "mimi.decoder", SEANET_SHAPE)?,
        };
        let shape = [VOICE_FRAMES, HIDDEN];
        let voices = VOICES
            .iter()
            .map(|(id, _)| Table::load(store, &voice_name(id), shape))
            .collect::<Result<_, _>>()?;
        Ok(Self { lm, mimi, voices })
    }

    /// The embedded latents of voice `index` of `VOICES`, one row per frame.
    pub(crate) fn voice(&self, index: usize) -> &Table {
        &self.voices[index]
    }
}

/// The bundle name of voice `id`.
pub fn voice_name(id: &str) -> String {
    format!("voice.{id}")
}
