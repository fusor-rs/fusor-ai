//! `spark-x2-6 compile`: the Spark-X2.6-1.7B checkpoint (its two safetensors shards, bf16, and its
//! tokenizer) → a browser bundle.
//!
//! Renames to the runtime's vocabulary (`spark_x2_6::model`), fuses each layer's gate projection
//! (one row per query head) after its query, key and value projection, and quantizes per profile;
//! fusor-ai's `BundleWriter` packs and hashes the parts and writes `manifest.json`. The shards
//! are read one at a time, each for the tensors it holds.

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use spark_x2_6::model::LANGUAGE;

const REPO: &str = "XHToken/Spark-X2.6-1.7B";
const REVISION: &str = "f18c032c7675f807c0fdffe281dbe124edba2605";
const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The large matrices 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    const ALWAYS_F32: &'static [&'static str] = &[];

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

/// Each layer's tensors taken as they are: the runtime's name and the checkpoint's, under the
/// layer.
const LAYER_PARTS: [(&str, &str); 6] = [
    ("input_norm", "input_layernorm.weight"),
    ("post_norm", "post_attention_layernorm.weight"),
    ("o.weight", "self_attn.out_proj.weight"),
    ("gate.weight", "mlp.gate_proj.weight"),
    ("up.weight", "mlp.up_proj.weight"),
    ("down.weight", "mlp.down_proj.weight"),
];

/// The model's tensors that `shard` holds, as the runtime names them: the head (the input
/// embeddings, tied), the final norm, and each layer's.
fn rename_table(shard: &Checkpoint) -> Table {
    let mut table = Table::default();
    let mut add = |name: String, sources: Vec<String>| {
        if shard.contains(&sources[0]) {
            table.push(name, sources, Layout::AsIs);
        }
    };
    add(
        "llm.head.weight".into(),
        vec!["model.embedding.weight".into()],
    );
    add("llm.norm".into(), vec!["model.norm.weight".into()]);
    for i in 0..LANGUAGE.layers {
        let (layer, from) = (format!("llm.layers.{i}"), format!("model.layers.{i}"));
        for (name, source) in LAYER_PARTS {
            add(format!("{layer}.{name}"), vec![format!("{from}.{source}")]);
        }
        let attention =
            ["q_k_v_proj", "g_proj"].map(|part| format!("{from}.self_attn.{part}.weight"));
        add(format!("{layer}.qkv.weight"), attention.to_vec());
    }
    table
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    for shard in SHARDS {
        let checkpoint = Checkpoint::read(&checkpoint_dir.join(shard))?;
        rename_table(&checkpoint).write(&checkpoint, &mut tensors)?;
    }
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
