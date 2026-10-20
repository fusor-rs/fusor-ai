//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `llm`, `flow`, `agg`, `stop`, `vae` — so this is the table of contents; the
//! compiler (`ming-compile`) writes exactly these names.

use blocks::qwen2::{Qwen2, Shape};
use bundle::{Load, Store};
use kernels::{Linear, Matrix, Tensor};

use crate::Error;
use crate::flow_head::{Aggregator, FlowHead};
use crate::vocoder::Vocoder;

/// The bundle part holding `tokenizer.json`.
pub const TOKENIZER_PART: &str = "tokenizer.json";

/// Qwen2.5-0.5B's dimensions, shared by the LLM and the VAE decoder's backbone.
pub const QWEN2_SHAPE: Shape = Shape {
    hidden: 896,
    layers: 24,
    heads: 14,
    kv_heads: 2,
    head_dim: 64,
    intermediate: 4864,
    rope_base: 1e6,
};
/// Width of the LLM's hidden states (and of its input embeddings).
pub const HIDDEN: usize = QWEN2_SHAPE.hidden;

pub(crate) struct Llm {
    embed: Tensor,
    pub(crate) body: Qwen2,
}

impl Llm {
    pub(crate) fn embeddings(&self) -> Matrix<'_> {
        self.embed.matrix().expect("checked at load")
    }
}

pub(crate) struct Model {
    pub(crate) llm: Llm,
    pub(crate) flow: FlowHead,
    pub(crate) aggregator: Aggregator,
    pub(crate) stop: Linear,
    pub(crate) vocoder: Vocoder,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let embed = Tensor::load(store, "llm.embed", ())?;
        embed.matrix()?;
        Ok(Self {
            llm: Llm {
                embed,
                body: Qwen2::load(store, "llm", QWEN2_SHAPE)?,
            },
            flow: FlowHead::load(store, "flow", ())?,
            aggregator: Aggregator::load(store, "agg", ())?,
            stop: Linear::load(store, "stop", ())?,
            vocoder: Vocoder::load(store, "vae", ())?,
        })
    }
}
