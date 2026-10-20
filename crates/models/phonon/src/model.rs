//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `lm`, `flow`, `mimi`, `voice` — so this is the table of contents; the compiler
//! (`phonon-compile`) writes exactly these names.

use blocks::adaln_mlp::{self, AdalnMlp};
use blocks::kyutai::{self, Norm, Transformer};
use blocks::seanet::{self, Seanet};
use bundle::{Load, Store};
use kernels::{Linear, Matrix, Tensor};

use crate::Error;
use crate::prompt::VOICES;

/// The bundle part holding `tokenizer.json`.
pub const TOKENIZER_PART: &str = "tokenizer.json";

/// Width of the flow LM, of its text embeddings and of the Mimi decoder's transformer.
pub const HIDDEN: usize = 512;
/// One frame's latent.
pub const LATENT: usize = 32;
/// Rows of the text embedding table (the last is the unused classifier-free-guidance row).
pub const TEXT_ROWS: usize = 4001;
/// Embedded latents of a voice (10 s at 12.5 Hz).
pub const VOICE_FRAMES: usize = 125;

pub const FLOW_LM_SHAPE: kyutai::Shape = kyutai::Shape {
    hidden: HIDDEN,
    layers: 6,
    heads: 8,
    head_dim: 64,
    feed_forward: 2048,
    rope_base: 1e4,
    norm_eps: 1e-5,
};
/// The flow LM's final norm.
pub const OUT_NORM_EPS: f32 = 1e-5;

pub const FLOW_SHAPE: adaln_mlp::Shape = adaln_mlp::Shape {
    width: 512,
    depth: 6,
    norm_eps: 1e-6,
};

pub const MIMI_TRANSFORMER_SHAPE: kyutai::Shape = kyutai::Shape {
    layers: 2,
    ..FLOW_LM_SHAPE
};
/// Keys a Mimi transformer query sees: itself and the 250 positions before it.
pub const MIMI_WINDOW: usize = 251;
/// Mimi transformer positions per latent frame (12.5 Hz → 200 Hz), the depthwise upsampling's
/// stride; its kernel is twice that.
pub const UPSAMPLE: usize = 16;

pub const SEANET_SHAPE: seanet::Shape = seanet::Shape {
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
    embed: Tensor,
    pub(crate) body: Transformer,
    /// `latent → hidden`, its bias the summed conditions (one speaker, rate unchanged).
    pub(crate) input: Linear,
    pub(crate) bos: Tensor,
    pub(crate) norm: Norm,
    pub(crate) eos: Linear,
    pub(crate) flow: AdalnMlp,
}

impl FlowLm {
    pub(crate) fn embeddings(&self) -> Matrix<'_> {
        self.embed.matrix().expect("checked at load")
    }
}

pub(crate) struct Mimi {
    /// The denormalization and the quantizer's output projection, `latent → hidden`.
    pub(crate) latent: Linear,
    /// The depthwise upsampling kernel, `[hidden, 2 · UPSAMPLE]`.
    pub(crate) upsample: Tensor,
    pub(crate) transformer: Transformer,
    pub(crate) decoder: Seanet,
}

pub(crate) struct Model {
    pub(crate) lm: FlowLm,
    pub(crate) mimi: Mimi,
    /// Each voice of `VOICES`: `[VOICE_FRAMES, HIDDEN]` embedded latents.
    voices: Vec<Tensor>,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let embed = Tensor::load(store, "lm.embed", ())?;
        embed.matrix()?;
        let lm = FlowLm {
            embed,
            body: Transformer::load(store, "lm", FLOW_LM_SHAPE)?,
            input: Linear::load(store, "lm.input", ())?,
            bos: f32_tensor(store, "lm.bos", &[LATENT])?,
            norm: Norm::load(store, "lm.norm", ())?,
            eos: Linear::load(store, "lm.eos", ())?,
            flow: AdalnMlp::load(store, "flow", FLOW_SHAPE)?,
        };
        let mimi = Mimi {
            latent: Linear::load(store, "mimi.latent", ())?,
            upsample: f32_tensor(store, "mimi.upsample", &[HIDDEN, 2 * UPSAMPLE])?,
            transformer: Transformer::load(store, "mimi", MIMI_TRANSFORMER_SHAPE)?,
            decoder: Seanet::load(store, "mimi.decoder", SEANET_SHAPE)?,
        };
        let voices = VOICES
            .iter()
            .map(|(id, _)| voice(store, id))
            .collect::<Result<_, _>>()?;
        Ok(Self { lm, mimi, voices })
    }
}

impl Model {
    /// The embedded latents of voice `index` of `VOICES`, one row per frame.
    pub(crate) fn voice(&self, index: usize) -> Matrix<'_> {
        self.voices[index].matrix().expect("checked at load")
    }
}

/// Voice `id`'s `[VOICE_FRAMES, HIDDEN]` matrix.
fn voice(store: &mut Store, id: &str) -> Result<Tensor, Error> {
    let name = voice_name(id);
    let tensor = Tensor::load(store, &name, ())?;
    let matrix = tensor.matrix()?;
    if [matrix.rows, matrix.cols] != [VOICE_FRAMES, HIDDEN] {
        return Err(Error::Format(format!("{name} of shape {:?}", tensor.shape)));
    }
    Ok(tensor)
}

/// Tensor `name`, which must be f32 of `shape`.
fn f32_tensor(store: &mut Store, name: &str, shape: &[usize]) -> Result<Tensor, Error> {
    let tensor = Tensor::load(store, name, ())?;
    tensor.values()?;
    if tensor.shape != shape {
        return Err(Error::Format(format!(
            "{name} of shape {:?}, not {shape:?}",
            tensor.shape
        )));
    }
    Ok(tensor)
}

/// The bundle name of voice `id`.
pub fn voice_name(id: &str) -> String {
    format!("voice.{id}")
}
