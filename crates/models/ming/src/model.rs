//! The model assembled from a loaded bundle, under the runtime's tensor names (plan §13.1):
//! `llm.*`, `flow.*`, `agg.*`, `stop.*`, `vae.*`. The compiler writes exactly these names.

use bundle::Store;
use kernels::{Linear, Matrix, Tensor};
use nn::qwen2::{Layer, Qwen2, Shape};

use crate::Error;
use crate::dit::{Aggregator, Block, Dit, FlowHead};
use crate::vocoder::Vocoder;

pub const DIT_BLOCKS: usize = 8;
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

pub struct Llm {
    embed: Tensor,
    pub body: Qwen2,
}

impl Llm {
    pub fn embeddings(&self) -> Matrix<'_> {
        self.embed.matrix().expect("checked at load")
    }
}

pub struct Model {
    pub llm: Llm,
    pub flow: FlowHead,
    pub aggregator: Aggregator,
    pub stop: Linear,
    pub vocoder: Vocoder,
}

impl Model {
    pub fn load(store: &mut Store) -> Result<Self, Error> {
        let mut load = Loader { store };
        let embed = load.tensor("llm.embed")?;
        embed.matrix()?;
        Ok(Self {
            llm: Llm {
                embed,
                body: load.qwen2("llm")?,
            },
            flow: FlowHead {
                x_embed: load.linear("flow.x_embed")?,
                c_embed: load.linear("flow.c_embed")?,
                t_mlp0: load.linear("flow.t_mlp0")?,
                t_mlp2: load.linear("flow.t_mlp2")?,
                dit: load.dit("flow")?,
                final_norm: load.tensor("flow.final_norm")?,
                final_linear: load.linear("flow.final")?,
            },
            aggregator: Aggregator {
                cls: load.tensor("agg.cls")?,
                x_embed: load.linear("agg.x_embed")?,
                dit: load.dit("agg")?,
                final_norm: load.tensor("agg.final_norm")?,
                final_linear: load.linear("agg.final")?,
            },
            stop: load.linear("stop")?,
            vocoder: Vocoder {
                fc1: load.linear("vae.fc1")?,
                body: load.qwen2("vae")?,
                head: load.linear("vae.head")?,
                window: load.tensor("vae.window")?,
            },
        })
    }
}

struct Loader<'a> {
    store: &'a mut Store,
}

impl Loader<'_> {
    fn tensor(&mut self, name: &str) -> Result<Tensor, Error> {
        Ok(self.store.take(name)?)
    }

    /// `{prefix}.weight` and, when the bundle has one, `{prefix}.bias`.
    fn linear(&mut self, prefix: &str) -> Result<Linear, Error> {
        let weight = self.store.take(&format!("{prefix}.weight"))?;
        let bias = self.store.take_optional(&format!("{prefix}.bias"));
        Ok(Linear::new(weight, bias)?)
    }

    fn qwen2(&mut self, prefix: &str) -> Result<Qwen2, Error> {
        let layers = (0..QWEN2_SHAPE.layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                Ok(Layer {
                    input_norm: self.tensor(&format!("{p}.input_norm"))?,
                    qkv: self.linear(&format!("{p}.qkv"))?,
                    output: self.linear(&format!("{p}.o"))?,
                    post_norm: self.tensor(&format!("{p}.post_norm"))?,
                    gate: self.linear(&format!("{p}.gate"))?,
                    up: self.linear(&format!("{p}.up"))?,
                    down: self.linear(&format!("{p}.down"))?,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Qwen2::new(
            QWEN2_SHAPE,
            layers,
            self.tensor(&format!("{prefix}.norm"))?,
        )?)
    }

    fn dit(&mut self, prefix: &str) -> Result<Dit, Error> {
        let blocks = (0..DIT_BLOCKS)
            .map(|i| {
                let p = format!("{prefix}.blocks.{i}");
                Ok(Block {
                    norm1: self.tensor(&format!("{p}.norm1"))?,
                    qkv: self.linear(&format!("{p}.qkv"))?,
                    out: self.linear(&format!("{p}.out"))?,
                    norm2: self.tensor(&format!("{p}.norm2"))?,
                    ff1: self.linear(&format!("{p}.ff1"))?,
                    ff2: self.linear(&format!("{p}.ff2"))?,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Dit::new(blocks))
    }
}
