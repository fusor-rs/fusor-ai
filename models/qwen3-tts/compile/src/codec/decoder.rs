//! The decoder: the quantizer's codebooks through its output projections, the conv, the
//! transformer with its layer scales folded, and the norm and projections around it.

use compile::{Checkpoint, Error, Tensors};
use qwen3_tts::codec::decoder::TRANSFORMER;
use qwen3_tts::model::{CODE_GROUPS, CODES};

use super::{centroids, scaled};
use crate::Profile;

const SOURCE: &str = "decoder";
/// The quantizer's first codebook, and the other 15 under one output projection.
const FIRST: &str = "decoder.quantizer.rvq_first";
const REST: &str = "decoder.quantizer.rvq_rest";

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    codebooks(checkpoint, tensors)?;
    tensors.conv1d(
        checkpoint,
        "codec.decoder.pre_conv",
        &format!("{SOURCE}.pre_conv.conv"),
    )?;
    let transformer = format!("{SOURCE}.pre_transformer");
    tensors.copy_linear(
        checkpoint,
        "codec.decoder.input",
        &format!("{transformer}.input_proj"),
    )?;
    for layer in 0..TRANSFORMER.layers {
        transformer_layer(checkpoint, tensors, layer)?;
    }
    let norm = checkpoint.f32(&format!("{transformer}.norm.weight"))?;
    tensors.vector("codec.decoder.norm", &norm)?;
    tensors.copy_linear(
        checkpoint,
        "codec.decoder.output",
        &format!("{transformer}.output_proj"),
    )
}

fn transformer_layer(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    layer: usize,
) -> Result<(), Error> {
    let to = format!("codec.decoder.transformer.layers.{layer}");
    let from = format!("{SOURCE}.pre_transformer.layers.{layer}");
    for (name, source) in [
        ("input_norm", "input_layernorm"),
        ("post_norm", "post_attention_layernorm"),
    ] {
        let weight = checkpoint.f32(&format!("{from}.{source}.weight"))?;
        tensors.vector(&format!("{to}.{name}"), &weight)?;
    }
    let mut qkv = Vec::new();
    for part in ["q_proj", "k_proj", "v_proj"] {
        qkv.extend(checkpoint.f32(&format!("{from}.self_attn.{part}.weight"))?);
    }
    let width = TRANSFORMER.heads * TRANSFORMER.head_dim;
    tensors.linear(&format!("{to}.qkv"), 3 * width, &qkv, None)?;
    let out = scaled(
        checkpoint,
        &format!("{from}.self_attn.o_proj.weight"),
        &format!("{from}.self_attn_layer_scale.scale"),
    )?;
    tensors.linear(&format!("{to}.o"), TRANSFORMER.hidden, &out, None)?;
    for part in ["gate", "up"] {
        let weight = checkpoint.f32(&format!("{from}.mlp.{part}_proj.weight"))?;
        tensors.linear(
            &format!("{to}.{part}"),
            TRANSFORMER.feed_forward,
            &weight,
            None,
        )?;
    }
    let down = scaled(
        checkpoint,
        &format!("{from}.mlp.down_proj.weight"),
        &format!("{from}.mlp_layer_scale.scale"),
    )?;
    tensors.linear(&format!("{to}.down"), TRANSFORMER.hidden, &down, None)
}

/// `[16 · 2048, 512]`: each codebook's centroids through the output projection of its half of
/// the quantizer, so that the decoder sums 16 rows.
fn codebooks(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut values = Vec::new();
    for book in 0..CODE_GROUPS {
        let (half, layer) = match book {
            0 => (FIRST, 0),
            _ => (REST, book - 1),
        };
        let codebook = format!("{half}.vq.layers.{layer}._codebook");
        let names = (
            format!("{codebook}.embedding_sum"),
            format!("{codebook}.cluster_usage"),
        );
        let entries = centroids(checkpoint, (&names.0, &names.1))?;
        let projection = checkpoint.f32(&format!("{half}.output_proj.weight"))?;
        values.extend(project(&entries, &projection));
    }
    let channels = values.len() / (CODE_GROUPS * CODES);
    tensors.add(
        "codec.decoder.codebooks",
        vec![CODE_GROUPS * CODES, channels],
        &values,
    )
}

/// Each row of `entries` through `projection` (`[outputs, inputs]`), accumulated in f64.
fn project(entries: &[f32], projection: &[f32]) -> Vec<f32> {
    let inputs = entries.len() / CODES;
    let mut projected = Vec::with_capacity(entries.len() / inputs * projection.len() / inputs);
    for entry in entries.chunks_exact(inputs) {
        for output in projection.chunks_exact(inputs) {
            let dot: f64 = entry
                .iter()
                .zip(output)
                .map(|(value, weight)| f64::from(*value) * f64::from(*weight))
                .sum();
            projected.push(dot as f32);
        }
    }
    projected
}
