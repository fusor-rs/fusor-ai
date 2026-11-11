//! SigLIP's vision tower (`vision_tower` of the text encoder's file) under
//! `motif_video::model::VISION`; the multi-modal projector is not read.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use motif_video::model::{VISION, VISION_FEED_FORWARD, VISION_SHAPE};

use crate::Profile;

const SOURCE: &str = "vision_tower.vision_model";

/// A layer's linears and norms: bundle name, checkpoint name, layout. Queries, keys and values
/// are fused apart.
const LAYER_PARTS: [(&str, &str, Layout); 5] = [
    ("out", "self_attn.out_proj", Layout::AsIs),
    ("ff1", "mlp.fc1", Layout::PadRows(VISION_FEED_FORWARD)),
    ("ff2", "mlp.fc2", Layout::PadColumns(VISION_FEED_FORWARD)),
    ("norm1", "layer_norm1", Layout::AsIs),
    ("norm2", "layer_norm2", Layout::AsIs),
];

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut table = Table::default();
    let embeddings = format!("{SOURCE}.embeddings");
    table.linear(
        &format!("{VISION}.patch"),
        &[format!("{embeddings}.patch_embedding")],
        &Layout::AsIs,
    );
    table.one(
        format!("{VISION}.positions"),
        format!("{embeddings}.position_embedding.weight"),
    );
    for layer in 0..VISION_SHAPE.layers {
        let name = format!("{VISION}.layers.{layer}");
        let source = format!("{SOURCE}.encoder.layers.{layer}");
        let qkv = ["q", "k", "v"].map(|head| format!("{source}.self_attn.{head}_proj"));
        table.linear(&format!("{name}.qkv"), &qkv, &Layout::AsIs);
        for (part, from, layout) in &LAYER_PARTS {
            table.linear(
                &format!("{name}.{part}"),
                &[format!("{source}.{from}")],
                layout,
            );
        }
    }
    table.linear(
        &format!("{VISION}.norm"),
        &[format!("{SOURCE}.post_layernorm")],
        &Layout::AsIs,
    );
    table.write(checkpoint, tensors)
}
