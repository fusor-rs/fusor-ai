//! `intern-decision compile`: the checkpoint (its vision, projector and language safetensors,
//! bf16) → a browser bundle.
//!
//! Renames to the runtime's vocabulary (`intern_decision::model`), sums the patch convolution's
//! two temporal kernels into one matrix (an image fills both frames alike), folds the
//! zero-centred RMSNorms' `1 + w` into their weights, splits each attention layer's interleaved
//! query and gate heads apart, fuses each layer's input projections into one matrix, and
//! quantizes per profile; fusor-ai's `BundleWriter` packs and hashes the parts and writes
//! `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use intern_decision::model::{CHANNELS, LANGUAGE_SHAPE, PATCH_VALUES, VISION_SHAPE};
use kernels::DType;

const REPO: &str = "internlm/Intern-Decision-2B";
const REVISION: &str = "85c5657708a09884ff9fd8d7f57789b410a97b33";
const VISION_WEIGHTS: &str = "model-vision-0001-save_rank0.safetensors";
const PROJECTOR_WEIGHTS: &str = "model-projector-0001-save_rank0.safetensors";
const LANGUAGE_WEIGHTS: &str = "model-language-0001-save_rank0.safetensors";
const VISION: &str = "model.visual";
const LANGUAGE: &str = "model.language_model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity checks only: 8.9 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// 4-bit feed-forwards in the language model, every other large matrix 7-bit: its mixers or
    /// its embeddings in 4 bits too flip a near-even decision of the fixtures.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7, Self::Mix];
    const ALWAYS_F32: &'static [&'static str] = &["vision.patch.", "vision.positions"];

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

/// The vision tower: `blocks::vit`'s names under `vision`, and the learned positions.
fn vision_table() -> Table {
    let mut table = Table::default();
    table.one(
        "vision.positions".into(),
        format!("{VISION}.pos_embed.weight"),
    );
    for i in 0..VISION_SHAPE.layers {
        let (layer, from) = (format!("vision.layers.{i}"), format!("{VISION}.blocks.{i}"));
        let linears = [
            ("qkv", "attn.qkv"),
            ("out", "attn.proj"),
            ("ff1", "mlp.linear_fc1"),
            ("ff2", "mlp.linear_fc2"),
            ("norm1", "norm1"),
            ("norm2", "norm2"),
        ];
        for (name, part) in linears {
            table.linear(
                &format!("{layer}.{name}"),
                &[format!("{from}.{part}")],
                &Layout::AsIs,
            );
        }
    }
    table
}

/// The merger: `blocks::vit::merger`'s names under `merger`.
fn merger_table() -> Table {
    let mut table = Table::default();
    for (name, part) in [
        ("norm", "norm"),
        ("up", "linear_fc1"),
        ("down", "linear_fc2"),
    ] {
        table.linear(
            &format!("merger.{name}"),
            &[format!("{VISION}.merger.{part}")],
            &Layout::AsIs,
        );
    }
    table
}

/// The language model's tensors taken as they are: the embeddings, the feed-forwards, the
/// attention layers' output projections and the delta layers' every tensor but their fused
/// input projection's order is the checkpoint's: queries, keys, values, `z`, `b`, `a`.
fn language_table() -> Table {
    let mut table = Table::default();
    table.one(
        "llm.embed".into(),
        format!("{LANGUAGE}.embed_tokens.weight"),
    );
    for i in 0..LANGUAGE_SHAPE.layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
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
    (layer + 1).is_multiple_of(LANGUAGE_SHAPE.attention_interval)
}

/// The patch embedding: the convolution's kernel over both frames summed into one over an
/// image's channel-major patch.
fn patch(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let source = format!("{VISION}.patch_embed.proj");
    let kernel = checkpoint.f32(&format!("{source}.weight"))?;
    let pixels = PATCH_VALUES / CHANNELS;
    let weight: Vec<f32> = kernel
        .chunks_exact(kernel.len() / (VISION_SHAPE.hidden * CHANNELS))
        .flat_map(|channel| {
            let frames: Vec<&[f32]> = channel.chunks_exact(pixels).collect();
            (0..pixels).map(move |at| frames.iter().map(|frame| frame[at]).sum::<f32>())
        })
        .collect();
    let bias = checkpoint.f32(&format!("{source}.bias"))?;
    tensors.linear("vision.patch", VISION_SHAPE.hidden, &weight, Some(&bias))
}

/// The language model's tensors that are not the checkpoint's as they are: every RMSNorm as
/// `1 + w`, and each attention layer's fused projection.
fn language(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    for i in 0..LANGUAGE_SHAPE.layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
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
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    layer: &str,
    from: &str,
) -> Result<(), Error> {
    let projection = |part: &str| checkpoint.f32(&format!("{from}.{part}_proj.weight"));
    let queries = projection("q")?;
    let head = LANGUAGE_SHAPE.attention.head_dim * LANGUAGE_SHAPE.hidden;
    let heads = queries.chunks_exact(head);
    let mut weight: Vec<f32> = heads.clone().step_by(2).flatten().copied().collect();
    weight.extend(heads.skip(1).step_by(2).flatten());
    weight.extend(projection("k")?);
    weight.extend(projection("v")?);
    let rows = weight.len() / LANGUAGE_SHAPE.hidden;
    tensors.linear(&format!("{layer}.qkv"), rows, &weight, None)
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let vision = Checkpoint::read(&checkpoint_dir.join(VISION_WEIGHTS))?;
    let projector = Checkpoint::read(&checkpoint_dir.join(PROJECTOR_WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    patch(&vision, &mut tensors)?;
    vision_table().write(&vision, &mut tensors)?;
    merger_table().write(&projector, &mut tensors)?;
    let language_weights = Checkpoint::read(&checkpoint_dir.join(LANGUAGE_WEIGHTS))?;
    language_table().write(&language_weights, &mut tensors)?;
    language(&language_weights, &mut tensors)?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
