//! `oxr compile`: the checkpoint (`model.safetensors`, bf16) → a browser bundle.
//!
//! Keeps what reading needs (not the vision tower's final norm, which the model never reads, nor
//! the input embeddings, which are the head's), renames to the runtime's vocabulary
//! (`oxr::model`), fuses q/k/v into one matrix per layer, flattens the patch convolution into a
//! matrix, pads the vision MLP with zero units and the visual head with zero words to whole
//! blocks, transposes the visual embeddings into the matrix that mixes them (the image's
//! indicators kept apart) and quantizes per profile; fusor-ai's `BundleWriter` packs and hashes
//! the parts and writes `manifest.json`.

use std::ops::Range;
use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
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

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        compile::profile(name)
    }
}

/// How a tensor's values are laid out in the bundle, from the checkpoint's.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Layout {
    AsIs,
    /// Zero rows (a vector: zero values) after the last, up to this many.
    PadRows(usize),
    /// Zero columns after each row's last, up to this many.
    PadColumns(usize),
    /// Rows and columns swapped.
    Transpose,
    /// Only these rows.
    Rows(Range<usize>),
}

/// A bundle tensor: the checkpoint tensors concatenated along rows, laid out as `layout`.
struct Entry {
    name: String,
    sources: Vec<String>,
    layout: Layout,
}

const PARTS: [&str; 2] = ["weight", "bias"];
const QKV: [&str; 3] = ["q", "k", "v"];

/// `(runtime name, checkpoint sources, layout)`, in bundle order.
#[derive(Default)]
struct Table(Vec<Entry>);

impl Table {
    fn push(&mut self, name: String, sources: Vec<String>, layout: Layout) {
        self.0.push(Entry {
            name,
            sources,
            layout,
        });
    }

    fn one(&mut self, name: String, source: String) {
        self.push(name, vec![source], Layout::AsIs);
    }

    /// A layer's weight and bias, each the row-concatenation of those of `sources`.
    fn linear(&mut self, name: &str, sources: &[String], layout: &Layout) {
        for part in PARTS {
            let parts = sources.iter().map(|source| format!("{source}.{part}"));
            let part_layout = match (part, layout) {
                ("bias", Layout::PadColumns(_)) => Layout::AsIs,
                _ => layout.clone(),
            };
            self.push(format!("{name}.{part}"), parts.collect(), part_layout);
        }
    }

    /// The vision tower: `blocks::vit`'s names under `vision`, the patch embedding and the
    /// position grid.
    fn vision(&mut self) {
        let embeddings = format!("{VISION}.embeddings");
        self.linear(
            "vision.patch",
            &[format!("{embeddings}.patch_embedding")],
            &Layout::AsIs,
        );
        self.one(
            "vision.positions".into(),
            format!("{embeddings}.position_embedding.weight"),
        );
        for i in 0..VISION_SHAPE.layers {
            let (layer, from) = (
                format!("vision.layers.{i}"),
                format!("{VISION}.encoder.layers.{i}"),
            );
            let qkv = QKV.map(|head| format!("{from}.self_attn.{head}_proj"));
            self.linear(&format!("{layer}.qkv"), &qkv, &Layout::AsIs);
            let linears = [
                ("out", "self_attn.out_proj", Layout::AsIs),
                ("ff1", "mlp.fc1", Layout::PadRows(VISION_FEED_FORWARD)),
                ("ff2", "mlp.fc2", Layout::PadColumns(VISION_FEED_FORWARD)),
                ("norm1", "layer_norm1", Layout::AsIs),
                ("norm2", "layer_norm2", Layout::AsIs),
            ];
            for (name, part, layout) in linears {
                self.linear(
                    &format!("{layer}.{name}"),
                    &[format!("{from}.{part}")],
                    &layout,
                );
            }
        }
    }

    /// The visual vocabulary: the head over the words, padded to the whole vocabulary, its
    /// norm, the embeddings transposed to mix them, and the image's indicators' embeddings.
    fn visual(&mut self) {
        let head = vec![format!("{HEAD}.0.weight")];
        let embeddings = vec![EMBEDDINGS.to_string()];
        self.push(
            "visual.head.weight".into(),
            head,
            Layout::PadRows(VISUAL_VOCABULARY),
        );
        self.linear("visual.norm", &[format!("{HEAD}.1")], &Layout::AsIs);
        self.push(
            "visual.mix.weight".into(),
            embeddings.clone(),
            Layout::Transpose,
        );
        let indicators = VISUAL_WORDS..VISUAL_WORDS + IMAGE_INDICATORS;
        self.push(
            "visual.indicators".into(),
            embeddings,
            Layout::Rows(indicators),
        );
    }

    /// The language model: `blocks::transformer`'s names under `llm`, its final norm and head.
    fn language(&mut self) {
        for i in 0..LANGUAGE_SHAPE.layers {
            let (layer, from) = (format!("llm.layers.{i}"), format!("{LANGUAGE}.layers.{i}"));
            let weights = |source: &str| format!("{from}.{source}.weight");
            self.one(format!("{layer}.input_norm"), weights("input_layernorm"));
            let qkv = QKV.map(|head| weights(&format!("self_attn.{head}_proj")));
            self.push(format!("{layer}.qkv.weight"), qkv.to_vec(), Layout::AsIs);
            self.one(format!("{layer}.q_norm"), weights("self_attn.q_norm"));
            self.one(format!("{layer}.k_norm"), weights("self_attn.k_norm"));
            self.one(format!("{layer}.o.weight"), weights("self_attn.o_proj"));
            self.one(
                format!("{layer}.post_norm"),
                weights("post_attention_layernorm"),
            );
            for projection in ["gate", "up", "down"] {
                self.one(
                    format!("{layer}.{projection}.weight"),
                    weights(&format!("mlp.{projection}_proj")),
                );
            }
        }
        self.one("llm.norm".into(), format!("{LANGUAGE}.norm.weight"));
        self.one("llm.head.weight".into(), "llm.lm_head.weight".into());
    }
}

fn rename_table() -> Vec<Entry> {
    let mut table = Table::default();
    table.vision();
    table.visual();
    table.language();
    table.0
}

/// The entry's values and bundle shape: vectors stay vectors, anything else is rows of its
/// first dimension (a convolution's kernel flattened), laid out as its layout says.
fn read(checkpoint: &Checkpoint, entry: &Entry) -> Result<(Vec<usize>, Vec<f32>), Error> {
    let mut values = Vec::new();
    let mut rows = 0;
    for source in &entry.sources {
        let shape = checkpoint.shape(source)?;
        rows += shape[0];
        values.extend(checkpoint.f32(source)?);
    }
    let vector = checkpoint.shape(&entry.sources[0])?.len() == 1;
    let cols = values.len() / rows;
    match &entry.layout {
        Layout::AsIs if vector => Ok((vec![rows], values)),
        Layout::AsIs => Ok((vec![rows, cols], values)),
        Layout::PadRows(padded) => {
            values.resize(padded * cols, 0.0);
            let shape = if vector {
                vec![*padded]
            } else {
                vec![*padded, cols]
            };
            Ok((shape, values))
        }
        Layout::PadColumns(padded) => {
            let rows_of = values.chunks_exact(cols);
            let padded_rows = rows_of.flat_map(|row| {
                let zeros = std::iter::repeat_n(0.0, padded - cols);
                row.iter().copied().chain(zeros)
            });
            Ok((vec![rows, *padded], padded_rows.collect()))
        }
        Layout::Transpose => {
            let transposed = (0..cols).flat_map(|col| (0..rows).map(move |row| (row, col)));
            let values = transposed.map(|(row, col)| values[row * cols + col]);
            Ok((vec![cols, rows], values.collect()))
        }
        Layout::Rows(kept) => {
            let values = values[kept.start * cols..kept.end * cols].to_vec();
            Ok((vec![kept.len(), cols], values))
        }
    }
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    for entry in rename_table() {
        let (shape, values) = read(&checkpoint, &entry)?;
        tensors.add(&entry.name, shape, &values)?;
    }
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
