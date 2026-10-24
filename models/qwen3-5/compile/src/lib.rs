//! `qwen3-5 compile`: the Qwen3.5-2B checkpoint (its safetensors, bf16, and its tokenizer) → a
//! browser bundle.
//!
//! Renames to the runtime's vocabulary (`qwen3_5::model`), folds the zero-centred RMSNorms'
//! `1 + w` into their weights, splits each attention layer's interleaved query and gate heads
//! apart, fuses each delta layer's input projections into one matrix, keeps the patch
//! convolution's two frames (a video's consecutive frames; an image fills both), and quantizes
//! per profile; fusor-ai's `BundleWriter` packs and hashes the parts and writes
//! `manifest.json`. The multi-token prediction layer (`mtp.*`), which only drafts tokens for
//! speculative decoding, is left out, as the reference leaves it.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use qwen3_5::model::{LANGUAGE, TOWER};

const REPO: &str = "Qwen/Qwen3.5-2B";
const REVISION: &str = "15852e8c16360a2fea060d615a32b45270f8a8fc";
const WEIGHTS: &str = "model.safetensors-00001-of-00001.safetensors";
const VISION: &str = "model.visual";
const LANGUAGE_MODEL: &str = "model.language_model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity checks only).
    F32,
    /// 7-bit weights for every large matrix: 4-bit feed-forwards move the language model's states
    /// five times as far from the reference's, and a reply carries its errors token to token.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    const ALWAYS_F32: &'static [&'static str] = &["vision.patch.", "vision.positions"];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
        }
    }

    fn matrix_dtype(self, _name: &str) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Q7 => DType::Q7b32,
        }
    }
}

/// The language model's feed-forward projections.
const FEED_FORWARD: [&str; 3] = ["gate", "up", "down"];

/// The vision tower: its patch embedding, the learned positions and `blocks::vit`'s names under
/// `vision`; and the merger, `blocks::vit::merger`'s names under `merger`.
fn vision_table() -> Table {
    let mut table = Table::default();
    table.linear(
        "vision.patch",
        &[format!("{VISION}.patch_embed.proj")],
        &Layout::AsIs,
    );
    table.one(
        "vision.positions".into(),
        format!("{VISION}.pos_embed.weight"),
    );
    let block_parts = [
        ("qkv", "attn.qkv"),
        ("out", "attn.proj"),
        ("ff1", "mlp.linear_fc1"),
        ("ff2", "mlp.linear_fc2"),
        ("norm1", "norm1"),
        ("norm2", "norm2"),
    ];
    for i in 0..TOWER.encoder.layers {
        let (layer, from) = (format!("vision.layers.{i}"), format!("{VISION}.blocks.{i}"));
        for (name, part) in block_parts {
            let sources = [format!("{from}.{part}")];
            table.linear(&format!("{layer}.{name}"), &sources, &Layout::AsIs);
        }
    }
    for (name, part) in [
        ("norm", "norm"),
        ("up", "linear_fc1"),
        ("down", "linear_fc2"),
    ] {
        let sources = [format!("{VISION}.merger.{part}")];
        table.linear(&format!("merger.{name}"), &sources, &Layout::AsIs);
    }
    table
}

/// The language model's tensors taken as they are: the head (the input embeddings, tied), the
/// feed-forwards, the attention layers' output projections and the delta layers' every tensor
/// but their fused input projection, whose order is the checkpoint's: queries, keys, values,
/// `z`, `b`, `a`.
fn language_table() -> Table {
    let mut table = Table::default();
    table.one(
        "llm.head.weight".into(),
        format!("{LANGUAGE_MODEL}.embed_tokens.weight"),
    );
    for i in 0..LANGUAGE.layers {
        let (layer, from) = (
            format!("llm.layers.{i}"),
            format!("{LANGUAGE_MODEL}.layers.{i}"),
        );
        let weights = |source: &str| format!("{from}.{source}.weight");
        for projection in FEED_FORWARD {
            table.one(
                format!("{layer}.{projection}.weight"),
                weights(&format!("mlp.{projection}_proj")),
            );
        }
        if is_attention(i) {
            table.one(format!("{layer}.o.weight"), weights("self_attn.o_proj"));
            continue;
        }
        let delta = format!("{from}.linear_attn");
        let inputs = ["qkv", "z", "b", "a"].map(|part| format!("{delta}.in_proj_{part}.weight"));
        table.push(
            format!("{layer}.input.weight"),
            inputs.to_vec(),
            Layout::AsIs,
        );
        table.one(format!("{layer}.conv"), format!("{delta}.conv1d.weight"));
        table.one(format!("{layer}.a_log"), format!("{delta}.A_log"));
        table.one(format!("{layer}.dt_bias"), format!("{delta}.dt_bias"));
        table.one(format!("{layer}.norm"), format!("{delta}.norm.weight"));
        table.one(
            format!("{layer}.out.weight"),
            format!("{delta}.out_proj.weight"),
        );
    }
    table
}

fn is_attention(layer: usize) -> bool {
    (layer + 1).is_multiple_of(LANGUAGE.attention_interval)
}

/// The language model's tensors that are not the checkpoint's as they are: every RMSNorm as
/// `1 + w`, and each attention layer's fused projection.
fn language(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    for i in 0..LANGUAGE.layers {
        let (layer, from) = (
            format!("llm.layers.{i}"),
            format!("{LANGUAGE_MODEL}.layers.{i}"),
        );
        let mut norms = vec![
            ("input_norm", format!("{from}.input_layernorm")),
            ("post_norm", format!("{from}.post_attention_layernorm")),
        ];
        if is_attention(i) {
            gated_qkv(checkpoint, tensors, &layer, &format!("{from}.self_attn"))?;
            norms.push(("q_norm", format!("{from}.self_attn.q_norm")));
            norms.push(("k_norm", format!("{from}.self_attn.k_norm")));
        }
        for (name, source) in norms {
            shifted_norm(checkpoint, tensors, &format!("{layer}.{name}"), &source)?;
        }
    }
    shifted_norm(
        checkpoint,
        tensors,
        "llm.norm",
        &format!("{LANGUAGE_MODEL}.norm"),
    )
}

/// A zero-centred RMSNorm's multipliers, `1 + w`.
fn shifted_norm(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    name: &str,
    source: &str,
) -> Result<(), Error> {
    let weight = checkpoint.f32(&format!("{source}.weight"))?;
    let shifted: Vec<f32> = weight.iter().map(|value| 1.0 + value).collect();
    tensors.vector(name, &shifted)
}

/// An attention layer's projection `layer.qkv`: the query heads, their gate heads, the key heads
/// and the value heads. The checkpoint's query projection holds each head's query then its gate.
fn gated_qkv(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    layer: &str,
    from: &str,
) -> Result<(), Error> {
    let projection = |part: &str| checkpoint.f32(&format!("{from}.{part}_proj.weight"));
    let queries = projection("q")?;
    let head = LANGUAGE.attention.head_dim * LANGUAGE.hidden;
    let heads = queries.chunks_exact(head);
    let mut weight: Vec<f32> = heads.clone().step_by(2).flatten().copied().collect();
    weight.extend(heads.skip(1).step_by(2).flatten());
    weight.extend(projection("k")?);
    weight.extend(projection("v")?);
    let rows = weight.len() / LANGUAGE.hidden;
    tensors.linear(&format!("{layer}.qkv"), rows, &weight, None)
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    vision_table().write(&checkpoint, &mut tensors)?;
    language_table().write(&checkpoint, &mut tensors)?;
    language(&checkpoint, &mut tensors)?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
