//! `d3 compile`: a d3 checkpoint (its backbone's safetensors, bf16, its answer-code readout,
//! f32, and its decision configuration and tokenizer) → a browser bundle, for either size.
//!
//! Renames to the runtime's vocabulary (`d3::model`), folds the zero-centred RMSNorms' `1 + w`
//! into their weights, splits each attention layer's interleaved query and gate heads apart,
//! fuses each layer's input projections into one matrix, keeps the patch convolution's two frames
//! (a video's consecutive frames; an image fills both), and quantizes per profile; fusor-ai's
//! `BundleWriter` packs and hashes the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use d3::model::{DECISION_CONFIG, Decision, Size};
use kernels::DType;

const VISION: &str = "visual";
const LANGUAGE: &str = "language_model";
const READOUT: &str = "readout.safetensors";

/// Where each size comes from: its repository, revision and backbone weights.
fn source(size: Size) -> (&'static str, &'static str, &'static str) {
    match size {
        Size::Nano => (
            "vllm-sr/d3-nano",
            "462efc366229ea268cf4eed3738b3fc4415e4817",
            "model.safetensors",
        ),
        Size::Lite => (
            "vllm-sr/d3-lite",
            "03560c26f5199c3c04129a2ccb38960c913635e0",
            "model-00001-of-00001.safetensors",
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity checks only).
    F32,
    /// 7-bit weights for every large matrix (d3-lite's: its 4-bit feed-forwards move an
    /// uncertain yes-or-no across one half).
    Q7,
    /// 4-bit feed-forwards in the language model, every other large matrix 7-bit (d3-nano's).
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7, Self::Mix];
    const ALWAYS_F32: &'static [&'static str] = &["vision.patch.", "vision.positions", "readout"];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
            Self::Mix => "mix",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Mix if is_feed_forward(name) => DType::Q4b32,
            Self::Q7 | Self::Mix => DType::Q7b32,
        }
    }
}

/// The language model's feed-forward projections.
const FEED_FORWARD: [&str; 3] = ["gate", "up", "down"];

fn is_feed_forward(name: &str) -> bool {
    let Some(layer) = name.strip_prefix("llm.layers.") else {
        return false;
    };
    FEED_FORWARD
        .iter()
        .any(|projection| layer.ends_with(&format!(".{projection}.weight")))
}

/// The vision tower: its patch embedding, the learned positions and `blocks::vit`'s names under
/// `vision`; and the merger, `blocks::vit::merger`'s names under `merger`.
fn vision_table(size: Size) -> Table {
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
    for i in 0..size.vision().layers {
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

/// The language model's tensors taken as they are: the embeddings, the feed-forwards, the
/// attention layers' output projections and the delta layers' every tensor but their fused
/// input projection, whose order is the checkpoint's: queries, keys, values, `z`, `b`, `a`.
fn language_table(size: Size) -> Table {
    let mut table = Table::default();
    table.one(
        "llm.embed".into(),
        format!("{LANGUAGE}.embed_tokens.weight"),
    );
    for i in 0..size.language().layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
        let weights = |source: &str| format!("{from}.{source}.weight");
        for projection in FEED_FORWARD {
            table.one(
                format!("{layer}.{projection}.weight"),
                weights(&format!("mlp.{projection}_proj")),
            );
        }
        if is_attention(size, i) {
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

fn is_attention(size: Size, layer: usize) -> bool {
    (layer + 1).is_multiple_of(size.language().attention_interval)
}

/// The language model's tensors that are not the checkpoint's as they are: every RMSNorm as
/// `1 + w`, and each attention layer's fused projection.
fn language(
    size: Size,
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    for i in 0..size.language().layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
        let mut norms = vec![
            ("input_norm", format!("{from}.input_layernorm")),
            ("post_norm", format!("{from}.post_attention_layernorm")),
        ];
        if is_attention(size, i) {
            gated_qkv(
                size,
                checkpoint,
                tensors,
                &layer,
                &format!("{from}.self_attn"),
            )?;
            norms.push(("q_norm", format!("{from}.self_attn.q_norm")));
            norms.push(("k_norm", format!("{from}.self_attn.k_norm")));
        }
        for (name, source) in norms {
            shifted_norm(checkpoint, tensors, &format!("{layer}.{name}"), &source)?;
        }
    }
    shifted_norm(checkpoint, tensors, "llm.norm", &format!("{LANGUAGE}.norm"))
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
    size: Size,
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    layer: &str,
    from: &str,
) -> Result<(), Error> {
    let language = size.language();
    let projection = |part: &str| checkpoint.f32(&format!("{from}.{part}_proj.weight"));
    let queries = projection("q")?;
    let head = language.attention.head_dim * language.hidden;
    let heads = queries.chunks_exact(head);
    let mut weight: Vec<f32> = heads.clone().step_by(2).flatten().copied().collect();
    weight.extend(heads.skip(1).step_by(2).flatten());
    weight.extend(projection("k")?);
    weight.extend(projection("v")?);
    let rows = weight.len() / language.hidden;
    tensors.linear(&format!("{layer}.qkv"), rows, &weight, None)
}

/// The checkpoint's size, from its decision configuration.
fn size(checkpoint_dir: &Path) -> Result<Size, Error> {
    let decision: Decision = compile::read_json(&checkpoint_dir.join(DECISION_CONFIG))?;
    decision
        .size()
        .map_err(|unknown| Error::Format(unknown.to_string()))
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let size = size(checkpoint_dir)?;
    let (repo, revision, weights) = source(size);
    let backbone = Checkpoint::read(&checkpoint_dir.join(weights))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    vision_table(size).write(&backbone, &mut tensors)?;
    language_table(size).write(&backbone, &mut tensors)?;
    language(size, &backbone, &mut tensors)?;
    let readout = Checkpoint::read(&checkpoint_dir.join(READOUT))?;
    let mut table = Table::default();
    table.one("readout".into(), "weight".into());
    table.write(&readout, &mut tensors)?;
    writer.add_blob_file(DECISION_CONFIG, &checkpoint_dir.join(DECISION_CONFIG))?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: repo.into(),
        revision: revision.into(),
        profile: profile.name().into(),
    })
}
