//! The runtime tensors (`phonon::model`'s names) computed from the checkpoint's, with the folds
//! the crate docs list.

use std::path::Path;

use compile::{Checkpoint, Error, Tensors};
use kernels::ops::{rms_norm, silu};
use phonon::model::{
    FLOW_LM_SHAPE, FLOW_SHAPE, HIDDEN, LATENT, MIMI_TRANSFORMER_SHAPE, SEANET_SHAPE, VOICE_FRAMES,
    voice_name,
};
use phonon::prompt::VOICES;

use crate::{Profile, VOICES_REVISION};

/// The voice file of each voice of `VOICES`, in order, in the checkpoint folder: the checkpoint's
/// own, then voices of the `gradium/phonon-voices` catalog (at `VOICES_REVISION`) under their
/// catalog path.
const VOICE_FILES: [&str; VOICES.len()] = [
    "default-voice.safetensors",
    "voices/4SZHfMpw-p46Ywgs.safetensors",
    "voices/d9XCyhjo4JhkHF1d.safetensors",
    "voices/nGzMHMPAWGaJFHOo.safetensors",
    "voices/qIxlZ7Ei8KvWlyQH.safetensors",
    "voices/6PWnV0Nq4wu7RVBT.safetensors",
    "voices/zUCxzrOrugTSis03.safetensors",
];
const VOICE_TENSOR: &str = "speaker_wavs";

const LM: &str = "flow_lm";
const CONDITIONERS: &str = "flow_lm.condition_provider.conditioners";
/// The `num_speakers` value of every request, as a row of its lookup table.
const ONE_SPEAKER: usize = 1;
/// The `duration_delta` sinusoid's width; at delta 0 it is `[1; half] ++ [0; half]`
/// (cosines first).
const DURATION_DIM: usize = 128;
/// The time embedding's uncentred RMSNorm epsilon.
const TIME_NORM_EPS: f32 = 1e-5;
/// `(s, t)` of the flow head's single LSD step, the inputs of its two time embeddings.
const STEP_TIMES: [f32; 2] = [0.0, 1.0];
const SEANET: &str = "mimi.decoder.model";
/// SEANet's `nn.Sequential` numbering: the first conv, then per stage an ELU, the transposed
/// conv and the residual unit (its convs after its ELUs, at 1 and 3), then an ELU and the last
/// conv.
const FIRST_CONV: usize = 0;
const STAGE_LAYERS: usize = 3;
const RESIDUAL_CONVS: [&str; 2] = ["block.1.conv", "block.3.conv"];

/// The flow LM: text embeddings, the transformer, the input projection with the conditions as
/// its bias, BOS, the final norm and the end-of-speech head.
pub(crate) fn flow_lm(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    let embed = checkpoint.f32(&format!(
        "{CONDITIONERS}.transcript_in_segment.embed.weight"
    ))?;
    let rows = embed.len() / HIDDEN;
    tensors.add("lm.embed", vec![rows, HIDDEN], &embed)?;
    let transformer = Transformer {
        prefix: "lm",
        source: &format!("{LM}.transformer"),
        layers: FLOW_LM_SHAPE.layers,
    };
    transformer.write(checkpoint, tensors)?;
    let input = checkpoint.f32(&format!("{LM}.input_linear.weight"))?;
    tensors.linear("lm.input", HIDDEN, &input, Some(&conditions(checkpoint)?))?;
    tensors.vector("lm.bos", &checkpoint.f32(&format!("{LM}.bos_emb"))?)?;
    tensors.copy_norm(checkpoint, "lm.norm", &format!("{LM}.out_norm"))?;
    tensors.copy_linear(checkpoint, "lm.eos", &format!("{LM}.out_eos"))
}

/// The summed conditioners for one speaker and an unchanged speaking rate.
fn conditions(checkpoint: &Checkpoint) -> Result<Vec<f32>, Error> {
    let speakers = format!("{CONDITIONERS}.num_speakers");
    let table_name = format!("{speakers}.embed.weight");
    let width = checkpoint.shape(&table_name)?[1];
    let table = checkpoint.f32(&table_name)?;
    let speaker = &table[ONE_SPEAKER * width..(ONE_SPEAKER + 1) * width];
    let speaker = matvec(
        &checkpoint.f32(&format!("{speakers}.output_proj.weight"))?,
        speaker,
    );
    let rate_input: Vec<f32> = (0..DURATION_DIM)
        .map(|i| if i < DURATION_DIM / 2 { 1.0 } else { 0.0 })
        .collect();
    let rate_proj = checkpoint.f32(&format!("{CONDITIONERS}.duration_delta.output_proj.weight"))?;
    let rate = matvec(&rate_proj, &rate_input);
    Ok(speaker
        .iter()
        .zip(rate)
        .map(|(speaker, rate)| speaker + rate)
        .collect())
}

/// A Kyutai transformer of the checkpoint, as `blocks::transformer` names it.
struct Transformer<'names> {
    prefix: &'names str,
    source: &'names str,
    layers: usize,
}

impl Transformer<'_> {
    /// Every layer, its layer scales (when it has them) folded into the output rows of both
    /// residual branches.
    fn write(&self, checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
        for i in 0..self.layers {
            let (layer, from) = (
                format!("{}.layers.{i}", self.prefix),
                format!("{}.layers.{i}", self.source),
            );
            for norm in ["norm1", "norm2"] {
                tensors.copy_norm(
                    checkpoint,
                    &format!("{layer}.{norm}"),
                    &format!("{from}.{norm}"),
                )?;
            }
            let linears = [("qkv", "self_attn.in_proj"), ("ff1", "linear1")];
            for (name, source) in linears {
                tensors.copy_linear(
                    checkpoint,
                    &format!("{layer}.{name}"),
                    &format!("{from}.{source}"),
                )?;
            }
            let scaled = [("out", "self_attn.out_proj", 1), ("ff2", "linear2", 2)];
            for (name, weight, scale) in scaled {
                let mut values = checkpoint.f32(&format!("{from}.{weight}.weight"))?;
                let layer_scale = format!("{from}.layer_scale_{scale}.scale");
                if checkpoint.contains(&layer_scale) {
                    scale_rows(&mut values, &checkpoint.f32(&layer_scale)?);
                }
                tensors.linear(&format!("{layer}.{name}"), HIDDEN, &values, None)?;
            }
        }
        Ok(())
    }
}

/// The flow head as `blocks::adaln_mlp`: the constant time embedding in the condition bias, the
/// seven adaLN projections and the LayerNorm affines in one modulation matrix.
pub(crate) fn flow_head(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    let source = format!("{LM}.flow_net");
    let read = |name: &str| checkpoint.f32(&format!("{source}.{name}"));
    let width = FLOW_SHAPE.width;
    let time = time_embedding(checkpoint, &source)?;
    let cond_bias: Vec<f32> = read("cond_embed.bias")?
        .iter()
        .zip(&time)
        .map(|(bias, time)| bias + time)
        .collect();
    tensors.linear(
        "flow.cond",
        width,
        &read("cond_embed.weight")?,
        Some(&cond_bias),
    )?;
    tensors.copy_linear(checkpoint, "flow.input", &format!("{source}.input_proj"))?;
    let mut modulation = Modulation::default();
    for i in 0..FLOW_SHAPE.depth {
        let block = format!("res_blocks.{i}");
        let norm = [
            read(&format!("{block}.in_ln.weight"))?,
            read(&format!("{block}.in_ln.bias"))?,
        ];
        modulation.add(
            checkpoint,
            &format!("{source}.{block}.adaLN_modulation.1"),
            &norm,
        )?;
        for (name, part) in [("up", "mlp.0"), ("down", "mlp.2")] {
            let from = format!("{source}.{block}.{part}");
            tensors.copy_linear(checkpoint, &format!("flow.blocks.{i}.{name}"), &from)?;
        }
    }
    let unit = [vec![1.0; width], vec![0.0; width]];
    modulation.add(
        checkpoint,
        &format!("{source}.final_layer.adaLN_modulation.1"),
        &unit,
    )?;
    let rows = modulation.bias.len();
    tensors.linear(
        "flow.modulation",
        rows,
        &modulation.weight,
        Some(&modulation.bias),
    )?;
    tensors.copy_linear(
        checkpoint,
        "flow.final",
        &format!("{source}.final_layer.linear"),
    )
}

/// The rows of `blocks::adaln_mlp`'s modulation matrix, one norm at a time.
#[derive(Default)]
struct Modulation {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Modulation {
    /// One adaLN projection (`width` rows each of shift, scale and, for a block, gate) after a
    /// LayerNorm of weight `norm[0]` and bias `norm[1]`: rows for `w ⊙ (1 + scale)`, then for
    /// `b ⊙ (1 + scale) + shift`, then the gate's unchanged.
    fn add(
        &mut self,
        checkpoint: &Checkpoint,
        source: &str,
        norm: &[Vec<f32>; 2],
    ) -> Result<(), Error> {
        let weight = checkpoint.f32(&format!("{source}.weight"))?;
        let bias = checkpoint.f32(&format!("{source}.bias"))?;
        let width = FLOW_SHAPE.width;
        let cols = weight.len() / bias.len();
        let row = |chunk: usize, index: usize| &weight[(chunk * width + index) * cols..][..cols];
        let [norm_weight, norm_bias] = norm;
        for (index, factor) in norm_weight.iter().enumerate() {
            self.weight
                .extend(row(1, index).iter().map(|value| factor * value));
            self.bias.push(factor * (1.0 + bias[width + index]));
        }
        for (index, factor) in norm_bias.iter().enumerate() {
            let rows = row(1, index).iter().zip(row(0, index));
            self.weight
                .extend(rows.map(|(scale, shift)| factor * scale + shift));
            self.bias
                .push(factor * (1.0 + bias[width + index]) + bias[index]);
        }
        self.weight.extend_from_slice(&weight[2 * width * cols..]);
        self.bias.extend_from_slice(&bias[2 * width..]);
        Ok(())
    }
}

/// The mean of the head's two time embeddings at its single LSD step.
fn time_embedding(checkpoint: &Checkpoint, source: &str) -> Result<Vec<f32>, Error> {
    let mut sum = vec![0.0; FLOW_SHAPE.width];
    let mut normed = vec![0.0; FLOW_SHAPE.width];
    for (index, time) in STEP_TIMES.into_iter().enumerate() {
        let read = |name: &str| checkpoint.f32(&format!("{source}.time_embed.{index}.{name}"));
        let frequencies = read("freqs")?;
        let angles = frequencies.iter().map(|frequency| time * frequency);
        let sinusoid: Vec<f32> = angles
            .clone()
            .map(f32::cos)
            .chain(angles.map(f32::sin))
            .collect();
        let mut hidden = matvec(&read("mlp.0.weight")?, &sinusoid);
        for (value, bias) in hidden.iter_mut().zip(read("mlp.0.bias")?) {
            *value = silu(*value + bias);
        }
        let mut embedding = matvec(&read("mlp.2.weight")?, &hidden);
        for (value, bias) in embedding.iter_mut().zip(read("mlp.2.bias")?) {
            *value += bias;
        }
        rms_norm(
            &embedding,
            &read("mlp.3.alpha")?,
            TIME_NORM_EPS,
            &mut normed,
        );
        for (total, value) in sum.iter_mut().zip(&normed) {
            *total += value / STEP_TIMES.len() as f32;
        }
    }
    Ok(sum)
}

/// Mimi's decoder: the latent denormalization folded into the quantizer projection, the
/// depthwise upsampling kernel, the transformer and SEANet.
pub(crate) fn mimi(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let projection = checkpoint.f32("mimi.quantizer.output_proj.weight")?;
    let deviation = checkpoint.f32(&format!("{LM}.emb_std"))?;
    let bias = matvec(&projection, &checkpoint.f32(&format!("{LM}.emb_mean"))?);
    let mut weight = projection;
    for row in weight.as_chunks_mut::<LATENT>().0 {
        for (value, scale) in row.iter_mut().zip(&deviation) {
            *value *= scale;
        }
    }
    tensors.linear("mimi.latent", HIDDEN, &weight, Some(&bias))?;
    let upsample = checkpoint.f32("mimi.upsample.convtr.convtr.weight")?;
    tensors.add(
        "mimi.upsample",
        vec![HIDDEN, upsample.len() / HIDDEN],
        &upsample,
    )?;
    let transformer = Transformer {
        prefix: "mimi",
        source: "mimi.decoder_transformer.transformer",
        layers: MIMI_TRANSFORMER_SHAPE.layers,
    };
    transformer.write(checkpoint, tensors)?;
    seanet(checkpoint, tensors)
}

/// SEANet's layers as `blocks::seanet` causal-conv matrices.
fn seanet(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    conv(
        checkpoint,
        tensors,
        "init",
        &format!("{SEANET}.{FIRST_CONV}.conv"),
    )?;
    for (stage, &ratio) in SEANET_SHAPE.ratios.iter().enumerate() {
        let elu = FIRST_CONV + 1 + stage * STAGE_LAYERS;
        let name = |part: &str| format!("stages.{stage}.{part}");
        let upsample = format!("{SEANET}.{}.convtr", elu + 1);
        transposed_conv(checkpoint, tensors, &name("upsample"), &upsample, ratio)?;
        let [first, second] = RESIDUAL_CONVS.map(|conv| format!("{SEANET}.{}.{conv}", elu + 2));
        conv(checkpoint, tensors, &name("residual1"), &first)?;
        conv(checkpoint, tensors, &name("residual2"), &second)?;
    }
    let last = FIRST_CONV + 1 + SEANET_SHAPE.ratios.len() * STAGE_LAYERS + 1;
    conv(
        checkpoint,
        tensors,
        "final",
        &format!("{SEANET}.{last}.conv"),
    )
}

/// A conv layer of the checkpoint.
struct Conv {
    /// `[out, in, kernel]`, or `[in, out, kernel]` for a transposed conv.
    shape: [usize; 3],
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv {
    fn read(checkpoint: &Checkpoint, source: &str) -> Result<Self, Error> {
        let weight_name = format!("{source}.weight");
        let [first, second, kernel] = checkpoint.shape(&weight_name)?[..] else {
            return Err(Error::Format(format!("{weight_name} is not a conv weight")));
        };
        Ok(Self {
            shape: [first, second, kernel],
            weight: checkpoint.f32(&weight_name)?,
            bias: checkpoint.f32(&format!("{source}.bias"))?,
        })
    }
}

/// A conv `[out, in, k]` as `[out, k · in]`, oldest input first.
fn conv(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    name: &str,
    source: &str,
) -> Result<(), Error> {
    let Conv {
        shape: [outputs, inputs, kernel],
        weight,
        bias,
    } = Conv::read(checkpoint, source)?;
    let mut matrix = Vec::with_capacity(weight.len());
    for output in 0..outputs {
        for tap in 0..kernel {
            matrix
                .extend((0..inputs).map(|input| weight[(output * inputs + input) * kernel + tap]));
        }
    }
    tensors.linear(
        &format!("mimi.decoder.{name}"),
        outputs,
        &matrix,
        Some(&bias),
    )
}

/// A transposed conv `[in, out, 2s]` of stride `s` as a window-2 conv of `s` phases,
/// `[s · out, 2 · in]`: output step `s·p + j` is `W[:, :, s + j]` times input `p − 1` plus
/// `W[:, :, j]` times input `p`.
fn transposed_conv(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    name: &str,
    source: &str,
    stride: usize,
) -> Result<(), Error> {
    let Conv {
        shape: [inputs, outputs, kernel],
        weight,
        bias,
    } = Conv::read(checkpoint, source)?;
    if kernel != 2 * stride {
        return Err(Error::Format(format!(
            "{source}: kernel {kernel} for stride {stride}"
        )));
    }
    let at =
        |input: usize, output: usize, tap: usize| weight[(input * outputs + output) * kernel + tap];
    let mut matrix = Vec::with_capacity(weight.len());
    for phase in 0..stride {
        for output in 0..outputs {
            matrix.extend((0..inputs).map(|input| at(input, output, stride + phase)));
            matrix.extend((0..inputs).map(|input| at(input, output, phase)));
        }
    }
    let rows = stride * outputs;
    tensors.linear(
        &format!("mimi.decoder.{name}"),
        rows,
        &matrix,
        Some(&bias.repeat(stride)),
    )
}

/// The voices of `VOICES` through the speaker projection, `[VOICE_FRAMES, HIDDEN]` each.
pub(crate) fn voices(
    dir: &Path,
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    let projection = checkpoint.f32(&format!("{CONDITIONERS}.speaker_wavs.output_proj.weight"))?;
    for ((id, _), file) in VOICES.iter().zip(VOICE_FILES) {
        let path = dir.join(file);
        if !path.exists() {
            return Err(Error::Format(format!(
                "{}: missing; voice files come from gradium/phonon-voices at {VOICES_REVISION}",
                path.display()
            )));
        }
        let latents = Checkpoint::read(&path)?.f32(VOICE_TENSOR)?;
        let mut embedded = Vec::with_capacity(VOICE_FRAMES * HIDDEN);
        for frame in 0..VOICE_FRAMES {
            let column: Vec<f32> = (0..HIDDEN)
                .map(|channel| latents[channel * VOICE_FRAMES + frame])
                .collect();
            embedded.extend(matvec(&projection, &column));
        }
        tensors.add(&voice_name(id), vec![VOICE_FRAMES, HIDDEN], &embedded)?;
    }
    Ok(())
}

/// `matrix · x` for a row-major matrix of `x.len()` columns, accumulated in f64.
fn matvec(matrix: &[f32], x: &[f32]) -> Vec<f32> {
    let dot = |row: &[f32]| {
        row.iter()
            .zip(x)
            .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
            .sum::<f64>()
    };
    matrix
        .chunks_exact(x.len())
        .map(|row| dot(row) as f32)
        .collect()
}

/// Multiplies row `i` of a row-major matrix by `scales[i]`.
fn scale_rows(matrix: &mut [f32], scales: &[f32]) {
    let cols = matrix.len() / scales.len();
    for (row, scale) in matrix.chunks_exact_mut(cols).zip(scales) {
        row.iter_mut().for_each(|value| *value *= scale);
    }
}
