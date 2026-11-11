//! Qwen3-VL's language model (two safetensors shards, bf16) under `iris::model::TEXT`.

use std::path::Path;

use bundle::TOKENIZER_PART;
use compile::{BundleWriter, Checkpoint, Error, Layout, Table, Tensors};
use iris::model::{TEXT, TEXT_SHAPE};

use crate::Profile;

const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
/// The language model's tensors, under the multimodal checkpoint's prefix.
const LANGUAGE: &str = "model.language_model";

pub(crate) fn write(dir: &Path, writer: &mut BundleWriter, profile: Profile) -> Result<(), Error> {
    let mut tensors = Tensors::new(writer, profile);
    for shard in SHARDS {
        let checkpoint = Checkpoint::read(&dir.join(shard))?;
        rename_table(&checkpoint).write(&checkpoint, &mut tensors)?;
    }
    writer.add_blob_file(TOKENIZER_PART, &dir.join(TOKENIZER_PART))
}

/// The tensors that `shard` holds, as the runtime names them: the embeddings, and the layers
/// the transformer reads.
fn rename_table(shard: &Checkpoint) -> Table {
    let mut table = Table::default();
    let mut add = |name: String, sources: Vec<String>| {
        if shard.contains(&sources[0]) {
            table.push(name, sources, Layout::AsIs);
        }
    };
    add(
        format!("{TEXT}.embed"),
        vec![format!("{LANGUAGE}.embed_tokens.weight")],
    );
    for layer in 0..TEXT_SHAPE.layers {
        let name = |part: &str| format!("{TEXT}.layers.{layer}.{part}");
        let source = |part: &str| format!("{LANGUAGE}.layers.{layer}.{part}.weight");
        let qkv = ["q", "k", "v"].map(|part| source(&format!("self_attn.{part}_proj")));
        add(name("qkv.weight"), qkv.to_vec());
        let parts = [
            ("input_norm", "input_layernorm"),
            ("post_norm", "post_attention_layernorm"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("o.weight", "self_attn.o_proj"),
            ("gate.weight", "mlp.gate_proj"),
            ("up.weight", "mlp.up_proj"),
            ("down.weight", "mlp.down_proj"),
        ];
        for (bundle_part, checkpoint_part) in parts {
            add(name(bundle_part), vec![source(checkpoint_part)]);
        }
    }
    table
}
