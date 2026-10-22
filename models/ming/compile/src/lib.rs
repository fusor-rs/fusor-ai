//! `ming compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps only what synthesis with built-in or described voices needs (no VAE encoder, no speaker
//! projection, no tied LM head), renames to the runtime's vocabulary (`ming::model`), fuses
//! q/k/v into one matrix per layer and quantizes per profile; fusor-ai's `BundleWriter` packs
//! and hashes the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
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

const QKV: [&str; 3] = ["q", "k", "v"];

/// A Qwen2 stack: `blocks::transformer`'s Qwen2 names under `prefix`, and its final norm.
fn qwen2(table: &mut Table, prefix: &str, source: &str) {
    for i in 0..QWEN2_SHAPE.layers {
        let (layer, from) = (
            format!("{prefix}.layers.{i}"),
            format!("{source}.layers.{i}"),
        );
        table.one(
            format!("{layer}.input_norm"),
            format!("{from}.input_layernorm.weight"),
        );
        let qkv = QKV.map(|head| format!("{from}.self_attn.{head}_proj"));
        table.linear(&format!("{layer}.qkv"), &qkv, &Layout::AsIs);
        table.one(
            format!("{layer}.o.weight"),
            format!("{from}.self_attn.o_proj.weight"),
        );
        table.one(
            format!("{layer}.post_norm"),
            format!("{from}.post_attention_layernorm.weight"),
        );
        for projection in ["gate", "up", "down"] {
            table.one(
                format!("{layer}.{projection}.weight"),
                format!("{from}.mlp.{projection}_proj.weight"),
            );
        }
    }
    table.one(format!("{prefix}.norm"), format!("{source}.norm.weight"));
}

/// The flow head (`flow`) or the aggregator (`agg`) as `ming::flow_head` names them: its
/// input embedding, its DiT blocks and its final layer.
fn dit(table: &mut Table, prefix: &str, source: &str) {
    table.linear(
        &format!("{prefix}.x_embed"),
        &[format!("{source}.x_embedder")],
        &Layout::AsIs,
    );
    for i in 0..DIT_SHAPE.depth {
        let (block, from) = (
            format!("{prefix}.blocks.{i}"),
            format!("{source}.blocks.{i}"),
        );
        let qkv = QKV.map(|head| format!("{from}.attn.to_{head}"));
        table.linear(&format!("{block}.qkv"), &qkv, &Layout::AsIs);
        let linears = [
            ("out", "attn.to_out.0"),
            ("ff1", "mlp.ff.0.0"),
            ("ff2", "mlp.ff.2"),
        ];
        for (name, part) in linears {
            table.linear(
                &format!("{block}.{name}"),
                &[format!("{from}.{part}")],
                &Layout::AsIs,
            );
        }
        table.one(format!("{block}.norm1"), format!("{from}.norm1.weight"));
        table.one(format!("{block}.norm2"), format!("{from}.norm2.weight"));
    }
    let final_layer = format!("{source}.final_layer");
    table.linear(
        &format!("{prefix}.final"),
        &[format!("{final_layer}.linear")],
        &Layout::AsIs,
    );
    table.one(
        format!("{prefix}.final_norm"),
        format!("{final_layer}.norm_final.weight"),
    );
}

fn rename_table() -> Table {
    const FLOW: &str = "flowloss.cfm.model";
    let mut table = Table::default();
    table.one("llm.embed".into(), "model.model.embed_tokens.weight".into());
    qwen2(&mut table, "llm", "model.model");
    qwen2(&mut table, "vae", "audio.decoder.decoder");
    dit(&mut table, "flow", FLOW);
    let flow_linears = [
        ("flow.c_embed", "c_embedder.cond_embedder"),
        ("flow.t_mlp0", "t_embedder.time_mlp.0"),
        ("flow.t_mlp2", "t_embedder.time_mlp.2"),
    ];
    for (name, source) in flow_linears {
        table.linear(name, &[format!("{FLOW}.{source}")], &Layout::AsIs);
    }
    dit(&mut table, "agg", "linear_proj_audio");
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
        table.linear(name, &[source.to_string()], &Layout::AsIs);
    }
    table.one(
        "vae.window".into(),
        "audio.decoder.head.istft.window".into(),
    );
    table
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    rename_table().write(&checkpoint, &mut tensors)?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
