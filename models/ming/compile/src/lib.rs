//! `ming compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps only what synthesis with built-in or described voices needs (no VAE encoder, no speaker
//! projection, no tied LM head), renames to the runtime's vocabulary (`ming::model`), fuses
//! q/k/v into one matrix per layer and quantizes per profile; fusor-ai's `BundleWriter` packs
//! and hashes the parts and writes `manifest.json`.

use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;
use ming::flow_head::DIT_SHAPE;
use ming::model::QWEN2_SHAPE;

const REPO: &str = "inclusionAI/Ming-3.0-tts-0.5B";
const REVISION: &str = "e5bd8da31ecf1b9276cdbdda821fa891068f1368";
const WEIGHTS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity tests only: ~4.3 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// 4-bit LLM layers and VAE decoder, 7-bit flow head, aggregator and embeddings.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7, Self::Mix];
    const ALWAYS_F32: &'static [&'static str] = &[
        "flow.x_embed.",
        "agg.x_embed.",
        "vae.fc1.",
        "flow.t_mlp",
        "flow.final.",
        "stop.",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
            Self::Mix => "mix",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        let four_bit = name.starts_with("llm.layers.") || name.starts_with("vae.");
        match self {
            Self::F32 => DType::F32,
            Self::Mix if four_bit => DType::Q4b32,
            Self::Q7 | Self::Mix => DType::Q7b32,
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        compile::profile(name)
    }
}

/// `(runtime name, checkpoint tensors concatenated along rows)`, in bundle order.
#[derive(Default)]
struct Table(Vec<(String, Vec<String>)>);

const PARTS: [&str; 2] = ["weight", "bias"];
const QKV: [&str; 3] = ["q", "k", "v"];

impl Table {
    fn one(&mut self, name: String, source: String) {
        self.0.push((name, vec![source]));
    }

    /// A linear layer's weight and bias, each the row-concatenation of those of `sources` (q, k
    /// and v fused).
    fn linear(&mut self, name: &str, sources: &[String]) {
        for part in PARTS {
            let parts = sources.iter().map(|source| format!("{source}.{part}"));
            self.0.push((format!("{name}.{part}"), parts.collect()));
        }
    }

    /// A Qwen2 stack: `blocks::transformer`'s Qwen2 names under `prefix`, and its final norm.
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
            let qkv = QKV.map(|head| format!("{from}.self_attn.{head}_proj"));
            self.linear(&format!("{layer}.qkv"), &qkv);
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

    /// The flow head (`flow`) or the aggregator (`agg`) as `ming::flow_head` names them: its
    /// input embedding, its DiT blocks and its final layer.
    fn dit(&mut self, prefix: &str, source: &str) {
        self.linear(
            &format!("{prefix}.x_embed"),
            &[format!("{source}.x_embedder")],
        );
        for i in 0..DIT_SHAPE.depth {
            let (block, from) = (
                format!("{prefix}.blocks.{i}"),
                format!("{source}.blocks.{i}"),
            );
            let qkv = QKV.map(|head| format!("{from}.attn.to_{head}"));
            self.linear(&format!("{block}.qkv"), &qkv);
            let linears = [
                ("out", "attn.to_out.0"),
                ("ff1", "mlp.ff.0.0"),
                ("ff2", "mlp.ff.2"),
            ];
            for (name, part) in linears {
                self.linear(&format!("{block}.{name}"), &[format!("{from}.{part}")]);
            }
            self.one(format!("{block}.norm1"), format!("{from}.norm1.weight"));
            self.one(format!("{block}.norm2"), format!("{from}.norm2.weight"));
        }
        let final_layer = format!("{source}.final_layer");
        self.linear(
            &format!("{prefix}.final"),
            &[format!("{final_layer}.linear")],
        );
        self.one(
            format!("{prefix}.final_norm"),
            format!("{final_layer}.norm_final.weight"),
        );
    }
}

fn rename_table() -> Vec<(String, Vec<String>)> {
    const FLOW: &str = "flowloss.cfm.model";
    let mut table = Table::default();
    table.one("llm.embed".into(), "model.model.embed_tokens.weight".into());
    table.qwen2("llm", "model.model");
    table.qwen2("vae", "audio.decoder.decoder");
    table.dit("flow", FLOW);
    let flow_linears = [
        ("flow.c_embed", "c_embedder.cond_embedder"),
        ("flow.t_mlp0", "t_embedder.time_mlp.0"),
        ("flow.t_mlp2", "t_embedder.time_mlp.2"),
    ];
    for (name, source) in flow_linears {
        table.linear(name, &[format!("{FLOW}.{source}")]);
    }
    table.dit("agg", "linear_proj_audio");
    table.one(
        "agg.cls".into(),
        "linear_proj_audio.word_embedder.weight".into(),
    );
    let linears = [
        ("stop", "stop_head"),
        ("vae.fc1", "audio.decoder.fc1"),
        ("vae.head", "audio.decoder.head.out"),
    ];
    for (name, source) in linears {
        table.linear(name, &[source.to_string()]);
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
    let first = checkpoint.shape(&sources[0])?;
    if first.len() == 1 || first[0] == 1 && sources.len() == 1 {
        let len = sources
            .iter()
            .map(|source| Ok(checkpoint.shape(source)?.iter().product::<usize>()))
            .sum::<Result<_, Error>>()?;
        return Ok(vec![len]);
    }
    let rows = sources
        .iter()
        .map(|source| Ok(checkpoint.shape(source)?[0]))
        .sum::<Result<_, Error>>()?;
    Ok(vec![rows, first[1]])
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    for (name, sources) in rename_table() {
        let shape = shape_of(&checkpoint, &sources)?;
        let values: Vec<f32> = sources
            .iter()
            .map(|source| checkpoint.f32(source))
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        tensors.add(&name, shape, &values)?;
    }
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
