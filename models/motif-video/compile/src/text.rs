//! T5Gemma2's text encoder (`text_encoder/model.safetensors`) under `motif_video::model::TEXT`.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use motif_video::model::{TEXT, TEXT_SHAPE};

use crate::Profile;

/// Each zero-centred norm of a layer: its bundle name, its checkpoint name.
const NORMS: [(&str, &str); 6] = [
    ("input_norm", "pre_self_attn_layernorm"),
    ("mixer_out_norm", "post_self_attn_layernorm"),
    ("post_norm", "pre_feedforward_layernorm"),
    ("feed_forward_out_norm", "post_feedforward_layernorm"),
    ("q_norm", "self_attn.q_norm"),
    ("k_norm", "self_attn.k_norm"),
];

/// The gated MLP's projections: its bundle name, its checkpoint name.
const PROJECTIONS: [(&str, &str); 3] = [
    ("gate", "gate_proj"),
    ("up", "up_proj"),
    ("down", "down_proj"),
];

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut table = Table::default();
    table.one(format!("{TEXT}.embedding"), "embed_tokens.weight".into());
    table.one(
        format!("{TEXT}.end_of_image"),
        "embed_tokens.eoi_embedding".into(),
    );
    let layers = TEXT_SHAPE.mixers.len();
    for layer in 0..layers {
        add_layer(&mut table, layer);
    }
    table.write(checkpoint, tensors)?;
    for layer in 0..layers {
        for (name, source) in NORMS {
            let name = format!("{TEXT}.layers.{layer}.{name}");
            tensors.shifted_norm(checkpoint, &name, &format!("layers.{layer}.{source}"))?;
        }
    }
    tensors.shifted_norm(checkpoint, &format!("{TEXT}.norm"), "norm")
}

/// A layer's matrices: queries, keys and values fused, the output and the MLP.
fn add_layer(table: &mut Table, layer: usize) {
    let name = |part: &str| format!("{TEXT}.layers.{layer}.{part}");
    let source = |part: &str| format!("layers.{layer}.{part}");
    let qkv = ["q", "k", "v"].map(|head| source(&format!("self_attn.{head}_proj.weight")));
    table.push(name("qkv.weight"), qkv.to_vec(), Layout::AsIs);
    table.one(name("o.weight"), source("self_attn.o_proj.weight"));
    for (part, projection) in PROJECTIONS {
        table.one(
            name(&format!("{part}.weight")),
            source(&format!("mlp.{projection}.weight")),
        );
    }
}
