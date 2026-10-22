//! `oxr compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps what reading needs (not the vision tower's final norm, which the model never reads, nor
//! the input embeddings, which are the head's), renames to the runtime's vocabulary
//! (`oxr::model`), fuses q/k/v into one matrix per layer, flattens the patch convolution into a
//! matrix, pads the vision MLP with zero units and the visual head with zero words to whole
//! blocks, transposes the visual embeddings into the matrix that mixes them (the image's
//! indicators kept apart) and quantizes per profile; fusor-ai's `BundleWriter` packs and hashes
//! the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use oxr::model::{
    IMAGE_INDICATORS, LANGUAGE_SHAPE, VISION_FEED_FORWARD, VISION_SHAPE, VISUAL_VOCABULARY,
    VISUAL_WORDS,
};

const REPO: &str = "SoMarkAI/OXR-1.0";
const REVISION: &str = "520bed7e90f8f4ffa110db47bdb64d5cf80ccfba";
const WEIGHTS: &str = "model.safetensors";
const VISION: &str = "visual_tokenizer.vit.vision_model";
const HEAD: &str = "visual_tokenizer.head";
const EMBEDDINGS: &str = "vte.weight";
const LANGUAGE: &str = "llm.model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (native parity checks only: 5.5 GB).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
    /// A 4-bit visual head (its logits only rank the visual words), every other large matrix
    /// 7-bit: the vision layers and the language model in 4 bits each lose a dense table.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7, Self::Mix];
    const ALWAYS_F32: &'static [&'static str] =
        &["vision.patch.", "vision.positions", "visual.indicators"];

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
            Self::Mix if name.starts_with("visual.head.") => DType::Q4b32,
            Self::Q7 | Self::Mix => DType::Q7b32,
        }
    }
}

const QKV: [&str; 3] = ["q", "k", "v"];

/// The vision tower: `blocks::vit`'s names under `vision`, the patch embedding and the
/// position grid.
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
}

/// The visual vocabulary: the head over the words, padded to the whole vocabulary, its
/// norm, the embeddings transposed to mix them, and the image's indicators' embeddings.
fn visual(table: &mut Table) {
    let head = vec![format!("{HEAD}.0.weight")];
    let embeddings = vec![EMBEDDINGS.to_string()];
    table.push(
        "visual.head.weight".into(),
        head,
        Layout::PadRows(VISUAL_VOCABULARY),
    );
    table.linear("visual.norm", &[format!("{HEAD}.1")], &Layout::AsIs);
    table.push(
        "visual.mix.weight".into(),
        embeddings.clone(),
        Layout::Transpose,
    );
    let indicators = VISUAL_WORDS..VISUAL_WORDS + IMAGE_INDICATORS;
    table.push(
        "visual.indicators".into(),
        embeddings,
        Layout::Rows(indicators),
    );
}

/// The language model: `blocks::transformer`'s names under `llm`, its final norm and head.
fn language(table: &mut Table) {
    for i in 0..LANGUAGE_SHAPE.layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
        let weights = |source: &str| format!("{from}.{source}.weight");
        table.one(format!("{layer}.input_norm"), weights("input_layernorm"));
        let qkv = QKV.map(|head| weights(&format!("self_attn.{head}_proj")));
        table.push(format!("{layer}.qkv.weight"), qkv.to_vec(), Layout::AsIs);
        table.one(format!("{layer}.q_norm"), weights("self_attn.q_norm"));
        table.one(format!("{layer}.k_norm"), weights("self_attn.k_norm"));
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
    table.one("llm.head.weight".into(), "llm.lm_head.weight".into());
}

fn rename_table() -> Table {
    let mut table = Table::default();
    vision(&mut table);
    visual(&mut table);
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
