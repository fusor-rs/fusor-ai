//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `llm`, `flow`, `agg`, `stop`, `vae` — so this is the table of contents; the
//! compiler (`ming-compile`) writes exactly these names.

use blocks::lm::Decoder;
use blocks::rotary::Rotation;
use blocks::transformer::{Architecture, Shape};
use bundle::{Load, Store};
use kernels::{Linear, Matrix, Table};

use crate::Error;
use crate::flow_head::{Aggregator, FlowHead};
use crate::vocoder::Vocoder;

pub const SAMPLE_RATE: u32 = 44_100;

/// Qwen2.5-0.5B's dimensions, shared by the LLM and the VAE decoder's backbone.
pub const QWEN2_SHAPE: Shape = Shape {
    architecture: Architecture::Qwen2,
    hidden: 896,
    layers: 24,
    heads: 14,
    kv_heads: 2,
    head_dim: 64,
    feed_forward: 4864,
    rope_base: 1e6,
    rotation: Rotation::Sequence,
    norm_eps: 1e-6,
};
/// Width of the LLM's hidden states (and of its input embeddings).
pub const HIDDEN: usize = QWEN2_SHAPE.hidden;
/// Rows of the LLM's token embeddings.
const VOCABULARY: usize = 151_936;
/// The stop head's logits: continue, stop.
pub(crate) const STOP_LOGITS: usize = 2;

pub(crate) struct Llm {
    embed: Table,
    pub(crate) qwen2: Decoder,
}

impl Llm {
    pub(crate) fn embeddings(&self) -> Matrix<'_> {
        self.embed.matrix()
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
        Ok(Self {
            llm: Llm {
                embed: Table::load(store, "llm.embed", [VOCABULARY, HIDDEN])?,
                qwen2: Decoder::load(store, "llm", QWEN2_SHAPE)?,
            },
            flow: FlowHead::load(store, "flow", ())?,
            aggregator: Aggregator::load(store, "agg", ())?,
            stop: Linear::load(store, "stop", [STOP_LOGITS, HIDDEN])?,
            vocoder: Vocoder::load(store, "vae", ())?,
        })
    }
}
