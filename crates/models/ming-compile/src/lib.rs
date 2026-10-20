//! `ming compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps only what synthesis with built-in or described voices needs (no VAE encoder, no speaker
//! projection, no tied LM head), renames to the runtime's vocabulary (`ming::model`), fuses
//! q/k/v into one matrix per layer and quantizes per profile; fusor-ai's `BundleWriter` packs
//! and hashes the parts and writes `manifest.json`.

use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error};
use kernels::{BLOCK, DType, Tensor};
use ming::flow_head::DIT_SHAPE;
use ming::model::{QWEN2_SHAPE, TOKENIZER_PART};

pub const REPO: &str = "inclusionAI/Ming-3.0-tts-0.5B";
pub const REVISION: &str = "e5bd8da31ecf1b9276cdbdda821fa891068f1368";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity tests only: ~4.3 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// 4-bit LLM layers and VAE decoder, 7-bit flow head, aggregator and embeddings.
    Mix,
}

impl Profile {
    pub const ALL: [Self; 3] = [Self::F32, Self::Q7, Self::Mix];

    /// The name in `manifest.json` and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
            Self::Mix => "mix",
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.name() == name)
            .ok_or_else(|| Error::Format(format!("no profile {name:?} (f32, q7 or mix)")))
    }
}

/// Small or precision-critical matrices kept in f32 in every profile.
const ALWAYS_F32: [&str; 6] = [
    "flow.x_embed.",
    "agg.x_embed.",
    "vae.fc1.",
    "flow.t_mlp",
    "flow.final.",
    "stop.",
];

fn dtype(profile: Profile, name: &str, shape: &[usize]) -> DType {
    let matrix = shape.len() == 2 && shape[1].is_multiple_of(BLOCK);
    let kept = ALWAYS_F32.iter().any(|prefix| name.starts_with(prefix));
    if !matrix || profile == Profile::F32 || kept {
        return DType::F32;
    }
    let four_bit = name.starts_with("llm.layers.") || name.starts_with("vae.");
    if profile == Profile::Mix && four_bit {
        DType::Q4b32
    } else {
        DType::Q7b32
    }
}

/// `(runtime name, checkpoint tensors concatenated along rows)`, in bundle order.
#[derive(Default)]
struct Table(Vec<(String, Vec<String>)>);

const PARTS: [&str; 2] = ["weight", "bias"];

impl Table {
    fn add(&mut self, name: String, sources: Vec<String>) {
        self.0.push((name, sources));
    }

    fn one(&mut self, name: String, source: String) {
        self.add(name, vec![source]);
    }

    /// A linear layer's weight and bias.
    fn linear(&mut self, name: &str, source: &str) {
        for part in PARTS {
            self.one(format!("{name}.{part}"), format!("{source}.{part}"));
        }
    }

    /// A Qwen2 stack: `blocks::qwen2`'s names under `prefix`.
    fn qwen2(&mut self, prefix: &str, source: &str) {
        for i in 0..QWEN2_SHAPE.layers {
            let (layer, from) = (
                format!("{prefix}.layers.{i}"),
                format!("{source}.layers.{i}"),
            );
            self.one(
                format!("{layer}.input_norm"),
                format!("{from}.input_layernorm.weight"),
            );
            for part in PARTS {
                let qkv =
                    ["q", "k", "v"].map(|head| format!("{from}.self_attn.{head}_proj.{part}"));
                self.add(format!("{layer}.qkv.{part}"), qkv.to_vec());
            }
            self.one(
                format!("{layer}.o.weight"),
                format!("{from}.self_attn.o_proj.weight"),
            );
            self.one(
                format!("{layer}.post_norm"),
                format!("{from}.post_attention_layernorm.weight"),
            );
            for projection in ["gate", "up", "down"] {
                self.one(
                    format!("{layer}.{projection}.weight"),
                    format!("{from}.mlp.{projection}_proj.weight"),
                );
            }
        }
        self.one(format!("{prefix}.norm"), format!("{source}.norm.weight"));
    }

    /// A DiT stack: `blocks::dit`'s block names under `prefix`.
    fn dit_blocks(&mut self, prefix: &str, source: &str) {
        for i in 0..DIT_SHAPE.depth {
            let (block, from) = (
                format!("{prefix}.blocks.{i}"),
                format!("{source}.blocks.{i}"),
            );
            for part in PARTS {
                let qkv = ["q", "k", "v"].map(|head| format!("{from}.attn.to_{head}.{part}"));
                self.add(format!("{block}.qkv.{part}"), qkv.to_vec());
                self.one(
                    format!("{block}.out.{part}"),
                    format!("{from}.attn.to_out.0.{part}"),
                );
                self.one(
                    format!("{block}.ff1.{part}"),
                    format!("{from}.mlp.ff.0.0.{part}"),
                );
                self.one(
                    format!("{block}.ff2.{part}"),
                    format!("{from}.mlp.ff.2.{part}"),
                );
            }
            self.one(format!("{block}.norm1"), format!("{from}.norm1.weight"));
            self.one(format!("{block}.norm2"), format!("{from}.norm2.weight"));
        }
    }

    /// The flow head (`flow`) or the aggregator (`agg`): `ming::flow_head`'s names.
    fn flow_head(&mut self, prefix: &str, source: &str) {
        self.linear(
            &format!("{prefix}.x_embed"),
            &format!("{source}.x_embedder"),
        );
        self.linear(
            &format!("{prefix}.final"),
            &format!("{source}.final_layer.linear"),
        );
        if prefix == "flow" {
            self.linear(
                "flow.c_embed",
                &format!("{source}.c_embedder.cond_embedder"),
            );
            self.linear("flow.t_mlp0", &format!("{source}.t_embedder.time_mlp.0"));
            self.linear("flow.t_mlp2", &format!("{source}.t_embedder.time_mlp.2"));
        }
        self.dit_blocks(prefix, source);
        self.one(
            format!("{prefix}.final_norm"),
            format!("{source}.final_layer.norm_final.weight"),
        );
    }
}

fn rename_table() -> Vec<(String, Vec<String>)> {
    let mut table = Table::default();
    table.one("llm.embed".into(), "model.model.embed_tokens.weight".into());
    table.qwen2("llm", "model.model");
    table.qwen2("vae", "audio.decoder.decoder");
    table.flow_head("flow", "flowloss.cfm.model");
    table.flow_head("agg", "linear_proj_audio");
    table.one(
        "agg.cls".into(),
        "linear_proj_audio.word_embedder.weight".into(),
    );
    for part in PARTS {
        table.one(format!("stop.{part}"), format!("stop_head.{part}"));
        table.one(
            format!("vae.fc1.{part}"),
            format!("audio.decoder.fc1.{part}"),
        );
        table.one(
            format!("vae.head.{part}"),
            format!("audio.decoder.head.out.{part}"),
        );
    }
    table.one(
        "vae.window".into(),
        "audio.decoder.head.istft.window".into(),
    );
    table.0
}

/// The shape of the row-concatenation of `sources` (vectors stay vectors; `[1, n]` rows of a
/// single-row table become `[n]`).
fn shape_of(checkpoint: &Checkpoint, sources: &[String]) -> Result<Vec<usize>, Error> {
    let first = &checkpoint.entry(&sources[0])?.shape;
    if first.len() == 1 || first[0] == 1 && sources.len() == 1 {
        let len = sources
            .iter()
            .map(|source| Ok(checkpoint.entry(source)?.shape.iter().product::<usize>()))
            .sum::<Result<_, Error>>()?;
        return Ok(vec![len]);
    }
    let rows = sources
        .iter()
        .map(|source| Ok(checkpoint.entry(source)?.shape[0]))
        .sum::<Result<_, Error>>()?;
    Ok(vec![rows, first[1]])
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join("model.safetensors"))?;
    let mut writer = BundleWriter::create(out)?;
    for (name, sources) in rename_table() {
        let shape = shape_of(&checkpoint, &sources)?;
        let values: Vec<f32> = sources
            .iter()
            .map(|source| checkpoint.f32(source))
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        if values.len() != shape.iter().product::<usize>() {
            return Err(Error::Format(format!("{name}: shape {shape:?}")));
        }
        let dtype = dtype(profile, &name, &shape);
        writer.add_tensor(&name, &Tensor::encode(dtype, shape, &values))?;
    }
    let path = checkpoint_dir.join(TOKENIZER_PART);
    let tokenizer = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
    writer.add_blob(TOKENIZER_PART, &tokenizer)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
