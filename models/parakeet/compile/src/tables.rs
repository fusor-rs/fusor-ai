//! The runtime tensors (`parakeet::model`'s names) computed from the checkpoint's, with the folds
//! the crate docs list.

use compile::{Checkpoint, Error, Tensors};
use parakeet::model::{
    DECODER_HIDDEN, DECODER_LAYERS, ENCODER_SHAPE, GATES, MELS, SPEECH_HIDDEN, SPEECH_KERNEL,
    SUBSAMPLING_SHAPE,
};

use crate::Profile;
use crate::filterbank;
use crate::ternary::Export;

/// The feed-forwards' half step.
const HALF_STEP: f32 = 0.5;
/// The convolution module's BatchNorm epsilon (PyTorch's default).
const BATCH_NORM_EPS: f32 = 1e-5;
/// Taps of the subsampler's square kernels.
const TAPS: usize = SUBSAMPLING_SHAPE.kernel * SUBSAMPLING_SHAPE.kernel;
/// Frequencies of the subsampler's output.
const SUBSAMPLED_MELS: usize = MELS / SUBSAMPLING_SHAPE.factor();
/// Each conformer layer's feed-forwards: runtime name, norm and module in the checkpoint.
const FEED_FORWARDS: [(&str, &str, &str); 2] = [
    ("ff1", "norm_feed_forward1", "feed_forward1"),
    ("ff2", "norm_feed_forward2", "feed_forward2"),
];

/// The frontend's window and filterbank.
pub(crate) fn frontend(tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    tensors.vector("features.window", &filterbank::window())?;
    let mel = filterbank::mel();
    tensors.add("features.mel", vec![MELS, mel.len() / MELS], &mel)
}

/// The subsampler and every conformer layer.
pub(crate) fn encoder(
    checkpoint: &Checkpoint,
    export: &Export,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    subsampling(checkpoint, tensors)?;
    for layer in 0..ENCODER_SHAPE.layers {
        let layer = Layer {
            checkpoint,
            export,
            prefix: format!("encoder.layers.{layer}"),
        };
        layer.write(tensors)?;
    }
    Ok(())
}

/// One conformer layer, under the same prefix in the checkpoint and in the bundle.
struct Layer<'checkpoint> {
    checkpoint: &'checkpoint Checkpoint,
    export: &'checkpoint Export,
    prefix: String,
}

impl Layer<'_> {
    fn write(&self, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
        let checkpoint = self.checkpoint;
        let norm = |tensors: &mut Tensors<Profile>, name: &str, source: &str| {
            tensors.copy_norm(checkpoint, &self.path(name), &self.path(source))
        };
        for (name, norm_source, module) in FEED_FORWARDS {
            norm(tensors, &format!("{name}.norm"), norm_source)?;
            let up = [format!("{module}.linear1")];
            self.ternary(tensors, &format!("{name}.up"), &up, 1.0)?;
            let down = [format!("{module}.linear2")];
            self.ternary(tensors, &format!("{name}.down"), &down, HALF_STEP)?;
        }
        let attention = |part: &str| format!("self_attn.{part}");
        norm(tensors, "attention.norm", "norm_self_att")?;
        let qkv = ["q_proj", "k_proj", "v_proj"].map(attention);
        self.ternary(tensors, "attention.qkv", &qkv, 1.0)?;
        let position = [attention("relative_k_proj")];
        self.ternary(tensors, "attention.position", &position, 1.0)?;
        for bias in ["bias_u", "bias_v"] {
            let values = checkpoint.f32(&self.path(&attention(bias)))?;
            tensors.vector(&self.path(&format!("attention.{bias}")), &values)?;
        }
        self.ternary(tensors, "attention.out", &[attention("o_proj")], 1.0)?;
        norm(tensors, "conv.norm", "norm_conv")?;
        let pointwise1 = ["conv.pointwise_conv1".to_string()];
        self.ternary(tensors, "conv.pointwise1", &pointwise1, 1.0)?;
        self.depthwise(tensors)?;
        let pointwise2 = ["conv.pointwise_conv2".to_string()];
        self.ternary(tensors, "conv.pointwise2", &pointwise2, 1.0)?;
        norm(tensors, "norm", "norm_out")
    }

    fn path(&self, part: &str) -> String {
        format!("{}.{part}", self.prefix)
    }

    /// The ternary modules `modules`, their rows stacked and scaled by `scale`, as linear `name`.
    fn ternary(
        &self,
        tensors: &mut Tensors<Profile>,
        name: &str,
        modules: &[String],
        scale: f32,
    ) -> Result<(), Error> {
        let (mut weight, mut rows, mut cols) = (Vec::new(), 0, None);
        for module in modules {
            let matrix = self.export.weight(self.checkpoint, &self.path(module))?;
            if *cols.get_or_insert(matrix.cols) != matrix.cols {
                return Err(Error::Format(format!(
                    "{}: modules of different widths stacked",
                    self.path(name)
                )));
            }
            rows += matrix.values.len() / matrix.cols;
            weight.extend(matrix.values.iter().map(|value| value * scale));
        }
        tensors.linear(&self.path(name), rows, &weight, None)
    }

    /// The depthwise convolution with the BatchNorm after it folded in: its kernel tap-major,
    /// `[kernel, hidden]`, and a bias.
    fn depthwise(&self, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
        let read = |part: &str| self.checkpoint.f32(&self.path(&format!("conv.{part}")));
        let kernel = read("depthwise_conv.weight")?;
        let (gamma, beta) = (read("norm.weight")?, read("norm.bias")?);
        let (mean, variance) = (read("norm.running_mean")?, read("norm.running_var")?);
        let taps = ENCODER_SHAPE.kernel;
        let hidden = gamma.len();
        let scales: Vec<f32> = gamma
            .iter()
            .zip(&variance)
            .map(|(gamma, variance)| gamma / (variance + BATCH_NORM_EPS).sqrt())
            .collect();
        let mut folded = vec![0.0; taps * hidden];
        for (channel, scale) in scales.iter().enumerate() {
            for tap in 0..taps {
                folded[tap * hidden + channel] = kernel[channel * taps + tap] * scale;
            }
        }
        let bias: Vec<f32> = (0..hidden)
            .map(|channel| beta[channel] - mean[channel] * scales[channel])
            .collect();
        tensors.vector(&self.path("conv.depthwise"), &folded)?;
        tensors.vector(&self.path("conv.depthwise_bias"), &bias)
    }
}

/// The subsampler: its square kernels tap-major (`[taps, channels]`), its pointwise convolutions as
/// linears, and its last linear's inputs reordered from channel-major to frequency-major.
fn subsampling(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    const SOURCE: &str = "encoder.subsampling";
    for (name, layer) in [("conv", 0), ("depthwise1", 2), ("depthwise2", 5)] {
        let weight = checkpoint.f32(&format!("{SOURCE}.layers.{layer}.weight"))?;
        let channels = weight.len() / TAPS;
        let tap_major: Vec<f32> = (0..TAPS * channels)
            .map(|at| weight[(at % channels) * TAPS + at / channels])
            .collect();
        tensors.vector(&format!("{SOURCE}.{name}.weight"), &tap_major)?;
        let bias = checkpoint.f32(&format!("{SOURCE}.layers.{layer}.bias"))?;
        tensors.vector(&format!("{SOURCE}.{name}.bias"), &bias)?;
    }
    for (name, layer) in [("pointwise1", 3), ("pointwise2", 6)] {
        let source = format!("{SOURCE}.layers.{layer}");
        tensors.copy_linear(checkpoint, &format!("{SOURCE}.{name}"), &source)?;
    }
    let weight = checkpoint.f32(&format!("{SOURCE}.linear.weight"))?;
    let channels = SUBSAMPLING_SHAPE.channels;
    let cols = channels * SUBSAMPLED_MELS;
    let reordered: Vec<f32> = (0..weight.len())
        .map(|at| {
            let (row, input) = (at / cols, at % cols);
            let (frequency, channel) = (input / channels, input % channels);
            weight[row * cols + channel * SUBSAMPLED_MELS + frequency]
        })
        .collect();
    let bias = checkpoint.f32(&format!("{SOURCE}.linear.bias"))?;
    let rows = reordered.len() / cols;
    tensors.linear(&format!("{SOURCE}.out"), rows, &reordered, Some(&bias))
}

/// The speech head: its 1-wide convolutions as linears, its context convolution's inputs
/// frame-major (oldest first).
pub(crate) fn speech(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    tensors.copy_linear(checkpoint, "speech.projection", "vad_head.proj")?;
    let weight = checkpoint.f32("vad_head.ctx.weight")?;
    let cols = SPEECH_KERNEL * SPEECH_HIDDEN;
    let frame_major: Vec<f32> = (0..weight.len())
        .map(|at| {
            let (row, input) = (at / cols, at % cols);
            let (tap, channel) = (input / SPEECH_HIDDEN, input % SPEECH_HIDDEN);
            weight[(row * SPEECH_HIDDEN + channel) * SPEECH_KERNEL + tap]
        })
        .collect();
    let bias = checkpoint.f32("vad_head.ctx.bias")?;
    tensors.linear("speech.context", SPEECH_HIDDEN, &frame_major, Some(&bias))?;
    tensors.copy_linear(checkpoint, "speech.out", "vad_head.out")
}

/// The prediction network and the joint: each LSTM layer's input and recurrent matrices side by
/// side and their biases summed, rows unit-major.
pub(crate) fn decoder(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
) -> Result<(), Error> {
    let embedding = checkpoint.f32("decoder.embedding.weight")?;
    let shape = vec![embedding.len() / DECODER_HIDDEN, DECODER_HIDDEN];
    tensors.add("decoder.embedding", shape, &embedding)?;
    for layer in 0..DECODER_LAYERS {
        let read = |part: &str| checkpoint.f32(&format!("decoder.lstm.{part}_l{layer}"));
        let (input, recurrent) = (read("weight_ih")?, read("weight_hh")?);
        let (input_bias, recurrent_bias) = (read("bias_ih")?, read("bias_hh")?);
        let rows = GATES * DECODER_HIDDEN;
        let mut weight = Vec::with_capacity(rows * 2 * DECODER_HIDDEN);
        let mut bias = Vec::with_capacity(rows);
        for unit_major in 0..rows {
            let row = (unit_major % GATES) * DECODER_HIDDEN + unit_major / GATES;
            weight.extend_from_slice(&input[row * DECODER_HIDDEN..(row + 1) * DECODER_HIDDEN]);
            weight.extend_from_slice(&recurrent[row * DECODER_HIDDEN..(row + 1) * DECODER_HIDDEN]);
            bias.push(input_bias[row] + recurrent_bias[row]);
        }
        let name = format!("decoder.layers.{layer}");
        tensors.linear(&name, rows, &weight, Some(&bias))?;
    }
    tensors.copy_linear(checkpoint, "decoder.projector", "decoder.decoder_projector")?;
    tensors.copy_linear(checkpoint, "joint.encoder", "encoder_projector")?;
    tensors.copy_linear(checkpoint, "joint.head", "joint.head")
}
