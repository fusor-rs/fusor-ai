//! The text bundle's tensors: Qwen3.5-4B's embeddings and decoder layers 0 to 27 under `text`,
//! read shard by shard.

use std::path::Path;

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use pjit_21k::model::{LANGUAGE, TEXT};

use crate::Profile;

const SHARDS: [&str; 2] = [
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
];
const LANGUAGE_MODEL: &str = "model.language_model";
/// The language model's feed-forward projections.
const FEED_FORWARD: [&str; 3] = ["gate", "up", "down"];

/// The text bundle's tensors of the shards in `dir`, one shard in memory at a time.
pub(crate) fn write(dir: &Path, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    for name in SHARDS {
        let shard = Checkpoint::read(&dir.join(name))?;
        table(&shard).write(&shard, tensors)?;
        norms_and_attention(&shard, tensors)?;
    }
    Ok(())
}

fn is_attention(layer: usize) -> bool {
    (layer + 1).is_multiple_of(LANGUAGE.attention_interval)
}

/// The tensors of `shard` that are the checkpoint's as they are: the embeddings, the
/// feed-forwards, the attention layers' output projections and the delta layers' every tensor but
/// their fused input projection, whose order is the checkpoint's: queries, keys, values, `z`,
/// `b`, `a`. An entry is added only when the shard holds its first source.
fn table(shard: &Checkpoint) -> Table {
    let mut table = Table::default();
    let mut add = |name: String, sources: Vec<String>| {
        if shard.contains(&sources[0]) {
            table.push(name, sources, Layout::AsIs);
        }
    };
    add(
        format!("{TEXT}.embedding"),
        vec![format!("{LANGUAGE_MODEL}.embed_tokens.weight")],
    );
    for layer in 0..LANGUAGE.layers {
        let name = |part: &str| format!("{TEXT}.layers.{layer}.{part}");
        let source = |part: &str| format!("{LANGUAGE_MODEL}.layers.{layer}.{part}");
        for projection in FEED_FORWARD {
            let from = source(&format!("mlp.{projection}_proj.weight"));
            add(name(&format!("{projection}.weight")), vec![from]);
        }
        if is_attention(layer) {
            add(name("o.weight"), vec![source("self_attn.o_proj.weight")]);
            continue;
        }
        let delta = |part: &str| source(&format!("linear_attn.{part}"));
        let inputs = ["qkv", "z", "b", "a"].map(|part| delta(&format!("in_proj_{part}.weight")));
        add(name("input.weight"), inputs.to_vec());
        add(name("conv"), vec![delta("conv1d.weight")]);
        add(name("a_log"), vec![delta("A_log")]);
        add(name("dt_bias"), vec![delta("dt_bias")]);
        add(name("norm"), vec![delta("norm.weight")]);
        add(name("out.weight"), vec![delta("out_proj.weight")]);
    }
    table
}

/// The tensors of the layers `shard` holds that are not the checkpoint's as they are: every
/// RMSNorm as `1 + w`, and each attention layer's fused projection.
fn norms_and_attention(shard: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    for layer in 0..LANGUAGE.layers {
        let name = format!("{TEXT}.layers.{layer}");
        let from = format!("{LANGUAGE_MODEL}.layers.{layer}");
        if !shard.contains(&format!("{from}.input_layernorm.weight")) {
            continue;
        }
        let mut norms = vec![
            ("input_norm", format!("{from}.input_layernorm")),
            ("post_norm", format!("{from}.post_attention_layernorm")),
        ];
        if is_attention(layer) {
            gated_qkv(shard, tensors, &name, &format!("{from}.self_attn"))?;
            norms.push(("q_norm", format!("{from}.self_attn.q_norm")));
            norms.push(("k_norm", format!("{from}.self_attn.k_norm")));
        }
        for (part, source) in norms {
            tensors.shifted_norm(shard, &format!("{name}.{part}"), &source)?;
        }
    }
    Ok(())
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
