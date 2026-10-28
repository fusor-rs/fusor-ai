//! The encoder (Mimi's): SEANet's convs, the transformer with its rotary pairs permuted and its
//! layer scales folded, the downsampling conv, the quantizer's projections and codebooks.

use compile::{Checkpoint, Error, Tensors};
use qwen3_tts::codec::encoder::{SEANET, TRANSFORMER};
use qwen3_tts::model::CODES;

use super::{centroids, scaled};
use crate::Profile;

const SOURCE: &str = "encoder";
/// The quantizer's two halves: the semantic codebook alone, and the acoustic residual's.
const SEMANTIC: &str = "quantizer.semantic_residual_vector_quantizer";
const ACOUSTIC: &str = "quantizer.acoustic_residual_vector_quantizer";
/// The acoustic codebooks the runtime keeps.
const ACOUSTIC_BOOKS: usize = qwen3_tts::model::CODE_GROUPS - 1;
/// A head's width: its rotary pairs are `i` with `i + HEAD / 2`, which the runtime pairs `2i`
/// with `2i + 1`.
const HEAD: usize = TRANSFORMER.head_dim;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    seanet(checkpoint, tensors)?;
    for layer in 0..TRANSFORMER.layers {
        transformer_layer(checkpoint, tensors, layer)?;
    }
    let downsample = format!("{SOURCE}.downsample.conv");
    tensors.conv1d(checkpoint, "codec.encoder.downsample", &downsample)?;
    for (name, half) in [("semantic", SEMANTIC), ("acoustic", ACOUSTIC)] {
        let source = format!("{SOURCE}.{half}.input_proj");
        tensors.conv1d(checkpoint, &format!("codec.encoder.{name}"), &source)?;
    }
    codebooks(checkpoint, tensors)
}

/// SEANet's layers: the first conv, per stage a residual unit and the downsampling conv
/// (`nn.Sequential` numbering: three layers a stage), and the last conv.
fn seanet(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let layer = |index: usize| format!("{SOURCE}.encoder.layers.{index}");
    let target = |part: &str| format!("codec.encoder.seanet.{part}");
    tensors.conv1d(checkpoint, &target("init"), &format!("{}.conv", layer(0)))?;
    for stage in 0..SEANET.ratios.len() {
        let unit = layer(1 + 3 * stage);
        let name = |part: &str| target(&format!("stages.{stage}.{part}"));
        tensors.conv1d(
            checkpoint,
            &name("residual1"),
            &format!("{unit}.block.1.conv"),
        )?;
        tensors.conv1d(
            checkpoint,
            &name("residual2"),
            &format!("{unit}.block.3.conv"),
        )?;
        let downsample = layer(3 + 3 * stage);
        tensors.conv1d(
            checkpoint,
            &name("downsample"),
            &format!("{downsample}.conv"),
        )?;
    }
    let last = layer(1 + 3 * SEANET.ratios.len() + 1);
    tensors.conv1d(checkpoint, &target("final"), &format!("{last}.conv"))
}

fn transformer_layer(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    layer: usize,
) -> Result<(), Error> {
    let to = format!("codec.encoder.transformer.layers.{layer}");
    let from = format!("{SOURCE}.encoder_transformer.layers.{layer}");
    tensors.copy_norm(
        checkpoint,
        &format!("{to}.norm1"),
        &format!("{from}.input_layernorm"),
    )?;
    tensors.copy_norm(
        checkpoint,
        &format!("{to}.norm2"),
        &format!("{from}.post_attention_layernorm"),
    )?;
    let attention = |part: &str| format!("{from}.self_attn.{part}");
    let mut qkv = Vec::new();
    for part in ["q_proj", "k_proj"] {
        let rows = checkpoint.f32(&attention(&format!("{part}.weight")))?;
        qkv.extend(pair_neighbours(&rows));
    }
    qkv.extend(checkpoint.f32(&attention("v_proj.weight"))?);
    tensors.linear(&format!("{to}.qkv"), 3 * TRANSFORMER.hidden, &qkv, None)?;
    let out = scaled(
        checkpoint,
        &attention("o_proj.weight"),
        &format!("{from}.self_attn_layer_scale.scale"),
    )?;
    tensors.linear(&format!("{to}.out"), TRANSFORMER.hidden, &out, None)?;
    let up = checkpoint.f32(&format!("{from}.mlp.fc1.weight"))?;
    tensors.linear(&format!("{to}.ff1"), TRANSFORMER.feed_forward, &up, None)?;
    let down = scaled(
        checkpoint,
        &format!("{from}.mlp.fc2.weight"),
        &format!("{from}.mlp_layer_scale.scale"),
    )?;
    tensors.linear(&format!("{to}.ff2"), TRANSFORMER.hidden, &down, None)
}

/// A query or key projection's rows with each head's pairs `(i, i + HEAD / 2)` made neighbours:
/// new row `2i + j` of a head is the old row `i + HEAD / 2 · j`.
fn pair_neighbours(rows: &[f32]) -> Vec<f32> {
    let width = rows.len() / TRANSFORMER.hidden;
    let mut permuted = Vec::with_capacity(rows.len());
    for head in rows.chunks_exact(HEAD * width) {
        for pair in 0..HEAD / 2 {
            for half in 0..2 {
                let row = pair + HEAD / 2 * half;
                permuted.extend_from_slice(&head[row * width..(row + 1) * width]);
            }
        }
    }
    permuted
}

/// `[16 · 2048, 256]`: the semantic codebook, then the acoustic residual's.
fn codebooks(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let sources =
        std::iter::once((SEMANTIC, 0)).chain((0..ACOUSTIC_BOOKS).map(|book| (ACOUSTIC, book)));
    let mut values = Vec::new();
    for (half, book) in sources {
        let layer = format!("{SOURCE}.{half}.layers.{book}.codebook");
        let names = (
            format!("{layer}.embed_sum"),
            format!("{layer}.cluster_usage"),
        );
        values.extend(centroids(checkpoint, (&names.0, &names.1))?);
    }
    let channels = values.len() / (CODES * (ACOUSTIC_BOOKS + 1));
    tensors.add(
        "codec.encoder.codebooks",
        vec![(ACOUSTIC_BOOKS + 1) * CODES, channels],
        &values,
    )
}
