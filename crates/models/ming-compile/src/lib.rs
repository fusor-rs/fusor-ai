//! `ming compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle (plan §5).
//!
//! Keeps only what synthesis with built-in or described voices needs (no VAE encoder, no speaker
//! projection, no tied LM head), renames to the runtime's vocabulary (`ming::model`), fuses
//! q/k/v into one matrix per layer and quantizes per profile; fusor-ai's `BundleWriter` packs
//! and hashes the parts and writes `manifest.json`.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use bundle::{Manifest, Source};
use clap::ValueEnum;
use compile::{BundleWriter, Checkpoint};
use kernels::{BLOCK, DType, Tensor};
use ming::model::{DIT_BLOCKS, QWEN2_SHAPE, TOKENIZER_PART};

pub const REPO: &str = "inclusionAI/Ming-3.0-tts-0.5B";
pub const REVISION: &str = "e5bd8da31ecf1b9276cdbdda821fa891068f1368";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Profile {
    /// Every weight f32 (native parity tests only: ~4.3 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// 4-bit LLM layers and VAE decoder, 7-bit flow head, aggregator and embeddings (plan §3).
    Mix,
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
    if !matrix || profile == Profile::F32 || ALWAYS_F32.iter().any(|p| name.starts_with(p)) {
        return DType::F32;
    }
    let four_bit = name.starts_with("llm.layers.") || name.starts_with("vae.");
    if profile == Profile::Mix && four_bit {
        DType::Q4b32
    } else {
        DType::Q7b32
    }
}

/// `(runtime name, checkpoint tensors concatenated along rows)`.
fn rename_table() -> Vec<(String, Vec<String>)> {
    let mut table: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |name: String, sources: &[String]| table.push((name, sources.to_vec()));
    add(
        "llm.embed".into(),
        &["model.model.embed_tokens.weight".into()],
    );
    for (prefix, source) in [("llm", "model.model"), ("vae", "audio.decoder.decoder")] {
        for i in 0..QWEN2_SHAPE.layers {
            let (p, s) = (
                format!("{prefix}.layers.{i}"),
                format!("{source}.layers.{i}"),
            );
            add(
                format!("{p}.input_norm"),
                &[format!("{s}.input_layernorm.weight")],
            );
            for part in ["weight", "bias"] {
                let qkv = ["q", "k", "v"].map(|x| format!("{s}.self_attn.{x}_proj.{part}"));
                add(format!("{p}.qkv.{part}"), &qkv);
            }
            add(
                format!("{p}.o.weight"),
                &[format!("{s}.self_attn.o_proj.weight")],
            );
            add(
                format!("{p}.post_norm"),
                &[format!("{s}.post_attention_layernorm.weight")],
            );
            for m in ["gate", "up", "down"] {
                add(
                    format!("{p}.{m}.weight"),
                    &[format!("{s}.mlp.{m}_proj.weight")],
                );
            }
        }
        add(format!("{prefix}.norm"), &[format!("{source}.norm.weight")]);
    }
    for (prefix, source) in [("flow", "flowloss.cfm.model"), ("agg", "linear_proj_audio")] {
        let mut linear = |name: &str, from: &str| {
            for part in ["weight", "bias"] {
                add(
                    format!("{prefix}.{name}.{part}"),
                    &[format!("{source}.{from}.{part}")],
                );
            }
        };
        linear("x_embed", "x_embedder");
        linear("final", "final_layer.linear");
        if prefix == "flow" {
            linear("c_embed", "c_embedder.cond_embedder");
            linear("t_mlp0", "t_embedder.time_mlp.0");
            linear("t_mlp2", "t_embedder.time_mlp.2");
        }
        for i in 0..DIT_BLOCKS {
            let (p, s) = (
                format!("{prefix}.blocks.{i}"),
                format!("{source}.blocks.{i}"),
            );
            for part in ["weight", "bias"] {
                let qkv = ["q", "k", "v"].map(|x| format!("{s}.attn.to_{x}.{part}"));
                add(format!("{p}.qkv.{part}"), &qkv);
                add(
                    format!("{p}.out.{part}"),
                    &[format!("{s}.attn.to_out.0.{part}")],
                );
                add(
                    format!("{p}.ff1.{part}"),
                    &[format!("{s}.mlp.ff.0.0.{part}")],
                );
                add(format!("{p}.ff2.{part}"), &[format!("{s}.mlp.ff.2.{part}")]);
            }
            add(format!("{p}.norm1"), &[format!("{s}.norm1.weight")]);
            add(format!("{p}.norm2"), &[format!("{s}.norm2.weight")]);
        }
        add(
            format!("{prefix}.final_norm"),
            &[format!("{source}.final_layer.norm_final.weight")],
        );
    }
    add(
        "agg.cls".into(),
        &["linear_proj_audio.word_embedder.weight".into()],
    );
    for part in ["weight", "bias"] {
        add(format!("stop.{part}"), &[format!("stop_head.{part}")]);
        add(
            format!("vae.fc1.{part}"),
            &[format!("audio.decoder.fc1.{part}")],
        );
        add(
            format!("vae.head.{part}"),
            &[format!("audio.decoder.head.out.{part}")],
        );
    }
    add(
        "vae.window".into(),
        &["audio.decoder.head.istft.window".into()],
    );
    table
}

/// The shape of the row-concatenation of `sources` (vectors stay vectors; `[1, n]` rows of a
/// single-row table become `[n]`).
fn shape_of(checkpoint: &Checkpoint, sources: &[String]) -> Result<Vec<usize>> {
    let first = &checkpoint.entry(&sources[0])?.shape;
    if first.len() == 1 || first[0] == 1 && sources.len() == 1 {
        let len = sources
            .iter()
            .map(|s| Ok(checkpoint.entry(s)?.shape.iter().product::<usize>()))
            .sum::<Result<_>>()?;
        return Ok(vec![len]);
    }
    let rows = sources
        .iter()
        .map(|s| Ok(checkpoint.entry(s)?.shape[0]))
        .sum::<Result<_>>()?;
    Ok(vec![rows, first[1]])
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join("model.safetensors"))?;
    let mut writer = BundleWriter::create(out)?;
    for (name, sources) in rename_table() {
        let shape = shape_of(&checkpoint, &sources)?;
        let values: Vec<f32> = sources
            .iter()
            .map(|s| checkpoint.f32(s))
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        ensure!(
            values.len() == shape.iter().product::<usize>(),
            "{name}: shape {shape:?}"
        );
        let dtype = dtype(profile, &name, &shape);
        writer.add_tensor(&name, &Tensor::encode(dtype, shape, &values))?;
    }
    let tokenizer = std::fs::read(checkpoint_dir.join(TOKENIZER_PART)).context("tokenizer.json")?;
    writer.add_blob(TOKENIZER_PART, &tokenizer)?;
    Ok(writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: format!("{profile:?}").to_lowercase(),
    })?)
}
