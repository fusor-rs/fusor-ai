//! `paddle-ocr compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps what reading needs (not the vision tower's pooling head, nor its packed position table),
//! renames to the runtime's vocabulary (`paddle_ocr::model`), fuses q/k/v into one matrix per
//! layer, flattens the patch convolution into a matrix, pads the vision MLP with zero units to
//! whole quantization blocks and quantizes per profile; fusor-ai's `BundleWriter` packs and
//! hashes the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use paddle_ocr::model::{LANGUAGE_SHAPE, VISION_FEED_FORWARD, VISION_SHAPE};

const REPO: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const REVISION: &str = "c5630abae1d940eafe0697512a0325494b02ab42";
const WEIGHTS: &str = "model.safetensors";
const VISION: &str = "visual.vision_model";
const LANGUAGE: &str = "model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity checks only: ~3.6 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// 4-bit vision layers; a 7-bit language model, projector, embeddings and head (the
    /// language model's layers in 4 bits fall into repeating themselves).
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
            Self::Mix if name.starts_with("vision.layers.") => DType::Q4b32,
            Self::Q7 | Self::Mix => DType::Q7b32,
        }
    }
}

const QKV: [&str; 3] = ["q", "k", "v"];

/// The vision tower: `blocks::vit`'s names under `vision`, the patch embedding, the
/// position grid and the final norm.
fn vision(table: &mut Table) {
    let embeddings = format!("{VISION}.embeddings");
    table.linear(
        "vision.patch",
        &[format!("{embeddings}.patch_embedding")],
        &Layout::AsIs,
    );
    table.one(
        "vision.positions".into(),
        format!("{embeddings}.position_embedding.weight"),
    );
    for i in 0..VISION_SHAPE.layers {
        let (layer, from) = (
            format!("vision.layers.{i}"),
            format!("{VISION}.encoder.layers.{i}"),
        );
        let qkv = QKV.map(|head| format!("{from}.self_attn.{head}_proj"));
        table.linear(&format!("{layer}.qkv"), &qkv, &Layout::AsIs);
        let linears = [
            ("out", "self_attn.out_proj", Layout::AsIs),
            ("ff1", "mlp.fc1", Layout::PadRows(VISION_FEED_FORWARD)),
            ("ff2", "mlp.fc2", Layout::PadColumns(VISION_FEED_FORWARD)),
            ("norm1", "layer_norm1", Layout::AsIs),
            ("norm2", "layer_norm2", Layout::AsIs),
        ];
        for (name, part, layout) in linears {
            table.linear(
                &format!("{layer}.{name}"),
                &[format!("{from}.{part}")],
                &layout,
            );
        }
    }
    table.linear(
        "vision.norm",
        &[format!("{VISION}.post_layernorm")],
        &Layout::AsIs,
    );
}

/// The language model: `blocks::transformer`'s names under `llm`, its embeddings, final
/// norm and head.
fn language(table: &mut Table) {
    table.one(
        "llm.embed".into(),
        format!("{LANGUAGE}.embed_tokens.weight"),
    );
    for i in 0..LANGUAGE_SHAPE.layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
        let weights = |source: &str| format!("{from}.{source}.weight");
        table.one(format!("{layer}.input_norm"), weights("input_layernorm"));
        let qkv = QKV.map(|head| weights(&format!("self_attn.{head}_proj")));
        table.push(format!("{layer}.qkv.weight"), qkv.to_vec(), Layout::AsIs);
        table.one(format!("{layer}.o.weight"), weights("self_attn.o_proj"));
        table.one(
            format!("{layer}.post_norm"),
            weights("post_attention_layernorm"),
        );
        for projection in ["gate", "up", "down"] {
            table.one(
                format!("{layer}.{projection}.weight"),
                weights(&format!("mlp.{projection}_proj")),
            );
        }
    }
    table.one("llm.norm".into(), format!("{LANGUAGE}.norm.weight"));
    table.one("llm.head.weight".into(), "lm_head.weight".into());
}

fn rename_table() -> Table {
    let mut table = Table::default();
    vision(&mut table);
    let projector = [
        ("projector.norm", "mlp_AR.pre_norm"),
        ("projector.up", "mlp_AR.linear_1"),
        ("projector.down", "mlp_AR.linear_2"),
    ];
    for (name, source) in projector {
        table.linear(name, &[source.to_string()], &Layout::AsIs);
    }
    language(&mut table);
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
