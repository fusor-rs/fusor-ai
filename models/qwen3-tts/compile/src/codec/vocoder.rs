//! The vocoder's convs as `blocks::code2wav`'s matrices, its ConvNeXt blocks' `gamma` folded into
//! their last layer.

use compile::{Checkpoint, Error, Tensors, scale_rows};
use qwen3_tts::codec::decoder::CODE2WAV;

use crate::Profile;

const SOURCE: &str = "decoder";
const TARGET: &str = "codec.vocoder";
/// `nn.Sequential` numbering of the decoder: the first conv, then the blocks, then the last
/// snake and conv.
const FIRST_BLOCK: usize = 1;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    for (index, &ratio) in CODE2WAV.upsampling.iter().enumerate() {
        let upsample = format!("{SOURCE}.upsample.{index}");
        let target = format!("{TARGET}.upsample.{index}");
        let transpose = format!("{upsample}.0.conv");
        tensors.transposed_conv1d(
            checkpoint,
            &format!("{target}.transpose"),
            &transpose,
            ratio,
        )?;
        convnext(
            checkpoint,
            tensors,
            (&format!("{upsample}.1"), &format!("{target}.convnext")),
        )?;
    }
    let decoder = format!("{SOURCE}.decoder");
    tensors.conv1d(
        checkpoint,
        &format!("{TARGET}.input"),
        &format!("{decoder}.0.conv"),
    )?;
    for (block, &rate) in CODE2WAV.rates.iter().enumerate() {
        let source = format!("{decoder}.{}.block", FIRST_BLOCK + block);
        let target = format!("{TARGET}.blocks.{block}");
        snake(
            checkpoint,
            tensors,
            (&format!("{source}.0"), &format!("{target}.snake")),
        )?;
        tensors.transposed_conv1d(
            checkpoint,
            &format!("{target}.upsample"),
            &format!("{source}.1.conv"),
            rate,
        )?;
        for unit in 0..CODE2WAV.dilations.len() {
            residual_unit(
                checkpoint,
                tensors,
                (
                    &format!("{source}.{}", 2 + unit),
                    &format!("{target}.units.{unit}"),
                ),
            )?;
        }
    }
    let last = FIRST_BLOCK + CODE2WAV.rates.len();
    snake(
        checkpoint,
        tensors,
        (&format!("{decoder}.{last}"), &format!("{TARGET}.snake")),
    )?;
    tensors.conv1d(
        checkpoint,
        &format!("{TARGET}.output"),
        &format!("{decoder}.{}.conv", last + 1),
    )
}

/// A residual unit: two snakes, a dilated conv and a 1-wide conv.
fn residual_unit(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    (source, target): (&str, &str),
) -> Result<(), Error> {
    for (name, part) in [("snake1", "act1"), ("snake2", "act2")] {
        snake(
            checkpoint,
            tensors,
            (&format!("{source}.{part}"), &format!("{target}.{name}")),
        )?;
    }
    for conv in ["conv1", "conv2"] {
        tensors.conv1d(
            checkpoint,
            &format!("{target}.{conv}"),
            &format!("{source}.{conv}.conv"),
        )?;
    }
    Ok(())
}

/// A SnakeBeta's `alpha` and `beta`, as the checkpoint has them: logarithms.
fn snake(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    (source, target): (&str, &str),
) -> Result<(), Error> {
    for part in ["alpha", "beta"] {
        let values = checkpoint.f32(&format!("{source}.{part}"))?;
        tensors.vector(&format!("{target}.{part}"), &values)?;
    }
    Ok(())
}

/// A ConvNeXt block: the depthwise conv's taps (one row a channel), the LayerNorm, the
/// feed-forward, its last layer's rows and bias scaled by `gamma`.
fn convnext(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    (source, target): (&str, &str),
) -> Result<(), Error> {
    let taps = checkpoint.f32(&format!("{source}.dwconv.conv.weight"))?;
    let channels = CODE2WAV.latent;
    tensors.add(
        &format!("{target}.depthwise"),
        vec![channels, taps.len() / channels],
        &taps,
    )?;
    let bias = checkpoint.f32(&format!("{source}.dwconv.conv.bias"))?;
    tensors.vector(&format!("{target}.depthwise_bias"), &bias)?;
    tensors.copy_norm(
        checkpoint,
        &format!("{target}.norm"),
        &format!("{source}.norm"),
    )?;
    tensors.copy_linear(
        checkpoint,
        &format!("{target}.up"),
        &format!("{source}.pwconv1"),
    )?;
    let gamma = checkpoint.f32(&format!("{source}.gamma"))?;
    let mut weight = checkpoint.f32(&format!("{source}.pwconv2.weight"))?;
    scale_rows(&mut weight, &gamma);
    let mut bias = checkpoint.f32(&format!("{source}.pwconv2.bias"))?;
    bias.iter_mut()
        .zip(&gamma)
        .for_each(|(value, scale)| *value *= scale);
    tensors.linear(&format!("{target}.down"), channels, &weight, Some(&bias))
}
