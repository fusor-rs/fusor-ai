//! Motif-Video's transformer (`transformer/diffusion_pytorch_model.safetensors`) under
//! `motif_video::model::DIT`.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use motif_video::model::{
    DIT, DUAL_BLOCKS, ENCODER_SINGLE_BLOCKS, LATENT_CHANNELS, PATCH, SINGLE_BLOCKS,
};

use crate::Profile;

/// A block's linears: bundle name, then the checkpoint names whose rows it fuses in order.
type Linears<'table> = &'table [(&'table str, &'table [&'table str])];
/// A block's vectors: bundle name, checkpoint name.
type Vectors<'table> = &'table [(&'table str, &'table str)];

const DUAL_LINEARS: Linears = &[
    ("modulation", &["norm1.linear", "norm1_context.linear"]),
    ("qkv", &["attn.to_q", "attn.to_k", "attn.to_v"]),
    (
        "context_qkv",
        &["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
    ),
    ("out", &["attn.to_out.0"]),
    ("context_out", &["attn.to_add_out"]),
    ("ff1", &["ff.net.0.proj"]),
    ("ff2", &["ff.net.2"]),
    ("context_ff1", &["ff_context.net.0.proj"]),
    ("context_ff2", &["ff_context.net.2"]),
];
const DUAL_VECTORS: Vectors = &[
    ("q_norm", "attn.norm_q.weight"),
    ("k_norm", "attn.norm_k.weight"),
    ("context_q_norm", "attn.norm_added_q.weight"),
    ("context_k_norm", "attn.norm_added_k.weight"),
];
const SINGLE_LINEARS: Linears = &[
    ("modulation", &["norm.linear"]),
    ("qkv", &["attn.to_q", "attn.to_k", "attn.to_v"]),
    ("mlp", &["proj_mlp"]),
    ("out", &["proj_out"]),
];
const SINGLE_VECTORS: Vectors = &[
    ("q_norm", "attn.norm_q.weight"),
    ("k_norm", "attn.norm_k.weight"),
];
/// The shared cross-attention of the first single-stream blocks, under `cross_attn`.
const CROSS_LINEARS: Linears = &[
    ("cross.query", &["cross_attn.to_q"]),
    ("cross.context", &["cross_attn.to_k", "cross_attn.to_v"]),
    ("cross.out", &["cross_attn.to_out.0"]),
];
const CROSS_VECTORS: Vectors = &[
    ("cross.query_norm", "cross_attn.norm_q.weight"),
    ("cross.key_norm", "cross_attn.norm_k.weight"),
];
/// The linears outside the blocks: bundle name, checkpoint name.
const PROJECTIONS: [(&str, &str); 9] = [
    ("context.0", "context_embedder.linear_1"),
    ("context.1", "context_embedder.linear_2"),
    ("image.norm_in", "image_embedder.norm_in"),
    ("image.norm_out", "image_embedder.norm_out"),
    ("image.0", "image_embedder.linear_1"),
    ("image.1", "image_embedder.linear_2"),
    ("time.0", "time_text_embed.timestep_embedder.linear_1"),
    ("time.1", "time_text_embed.timestep_embedder.linear_2"),
    ("head.modulation", "norm_out.linear"),
];

/// `linears` and `vectors` of the checkpoint block `source` as the bundle block `name`.
fn add_block(table: &mut Table, name: &str, source: &str, (linears, vectors): (Linears, Vectors)) {
    for (part, from) in linears {
        let sources: Vec<String> = from.iter().map(|part| format!("{source}.{part}")).collect();
        table.linear(&format!("{name}.{part}"), &sources, &Layout::AsIs);
    }
    for (part, from) in vectors {
        table.one(format!("{name}.{part}"), format!("{source}.{from}"));
    }
}

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    tensors.conv(checkpoint, &format!("{DIT}.patch"), "x_embedder.proj")?;
    let mut table = Table::default();
    for (name, source) in PROJECTIONS {
        table.linear(&format!("{DIT}.{name}"), &[source.into()], &Layout::AsIs);
    }
    for block in 0..DUAL_BLOCKS {
        let name = format!("{DIT}.dual.{block}");
        let source = format!("transformer_blocks.{block}");
        add_block(&mut table, &name, &source, (DUAL_LINEARS, DUAL_VECTORS));
    }
    for block in 0..SINGLE_BLOCKS {
        let name = format!("{DIT}.single.{block}");
        let source = format!("single_transformer_blocks.{block}");
        add_block(&mut table, &name, &source, (SINGLE_LINEARS, SINGLE_VECTORS));
        if block < ENCODER_SINGLE_BLOCKS {
            add_block(&mut table, &name, &source, (CROSS_LINEARS, CROSS_VECTORS));
        }
    }
    table.write(checkpoint, tensors)?;
    head(checkpoint, tensors)
}

/// The output projection `proj_out`, whose row `c · 4 + dy · 2 + dx` is channel `c` of the
/// patch's pixel `(dy, dx)`, with its rows in the order `blocks::video::Patching::scatter`
/// reads: `(dy · 2 + dx) · 16 + c`.
fn head(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let pixels = PATCH * PATCH;
    let rows = pixels * LATENT_CHANNELS;
    let weight = checkpoint.f32("proj_out.weight")?;
    let bias = checkpoint.f32("proj_out.bias")?;
    let columns = weight.len() / rows;
    let sources: Vec<usize> = (0..rows)
        .map(|row| (row % LATENT_CHANNELS) * pixels + row / LATENT_CHANNELS)
        .collect();
    let permuted_weight: Vec<f32> = sources
        .iter()
        .flat_map(|&source| {
            weight[source * columns..(source + 1) * columns]
                .iter()
                .copied()
        })
        .collect();
    let permuted_bias: Vec<f32> = sources.iter().map(|&source| bias[source]).collect();
    tensors.linear(
        &format!("{DIT}.head.linear"),
        rows,
        &permuted_weight,
        Some(&permuted_bias),
    )
}
