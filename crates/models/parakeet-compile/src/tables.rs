//! The runtime tensors (`parakeet::model`'s names) computed from the checkpoint's, with the folds
//! the crate docs list.

use compile::{Checkpoint, Error};
use parakeet::model::{
    DECODER_HIDDEN, DECODER_LAYERS, ENCODER_SHAPE, MELS, SPEECH_HIDDEN, SPEECH_KERNEL,
};

use crate::filterbank;
use crate::ternary::Export;

/// The feed-forwards' half step.
const HALF_STEP: f32 = 0.5;
/// The convolution module's BatchNorm epsilon (PyTorch's default).
const BATCH_NORM_EPS: f32 = 1e-5;
/// Taps of the subsampler's square kernels.
const TAPS: usize = ENCODER_SHAPE.subsampling_kernel * ENCODER_SHAPE.subsampling_kernel;
/// Frequencies of the subsampler's output.
const SUBSAMPLED_MELS: usize = MELS / ENCODER_SHAPE.subsampling_factor();
/// The LSTM gates per unit, in PyTorch's row order: input, forget, cell, output.
const GATES: usize = 4;
/// Each conformer layer's feed-forwards: runtime name, norm and module in the checkpoint.
const FEED_FORWARDS: [(&str, &str, &str); 2] = [
    ("ff1", "norm_feed_forward1", "feed_forward1"),
    ("ff2", "norm_feed_forward2", "feed_forward2"),
];

/// How a tensor may be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// One of the encoder's ternary projections.
    Ternary,
    /// A large dense matrix.
    Matrix,
    /// Kept f32 in every profile: vectors, kernels, the speech head and the frontend.
    F32,
}

/// A runtime tensor before encoding.
pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
    pub(crate) kind: Kind,
}

/// The runtime tensors in bundle order.
#[derive(Default)]
pub(crate) struct Tensors(Vec<Entry>);

impl Tensors {
    pub(crate) fn into_entries(self) -> Vec<Entry> {
        self.0
    }

    fn add(&mut self, name: String, shape: Vec<usize>, values: Vec<f32>, kind: Kind) {
        self.0.push(Entry {
            name,
            shape,
            values,
            kind,
        });
    }

    fn vector(&mut self, name: String, values: Vec<f32>) {
        self.add(name, vec![values.len()], values, Kind::F32);
    }

    /// Linear `name`: a weight of `cols` inputs and an optional bias.
    fn linear(
        &mut self,
        (name, kind): (&str, Kind),
        cols: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) {
        let shape = vec![weight.len() / cols, cols];
        self.add(format!("{name}.weight"), shape, weight, kind);
        if let Some(bias) = bias {
            self.vector(format!("{name}.bias"), bias);
        }
    }

    /// The checkpoint's linear (or 1-wide convolution) `source`, its weight and bias, as `name`.
    fn copy_linear(
        &mut self,
        checkpoint: &Checkpoint,
        (name, kind): (&str, Kind),
        source: &str,
    ) -> Result<(), Error> {
        let weight_name = format!("{source}.weight");
        let shape = &checkpoint.entry(&weight_name)?.shape;
        let cols = shape[1..].iter().product();
        let weight = checkpoint.f32(&weight_name)?;
        let bias = checkpoint.f32(&format!("{source}.bias"))?;
        self.linear((name, kind), cols, weight, Some(bias));
        Ok(())
    }

    /// A LayerNorm's `weight` and `bias` as norm `name`.
    fn copy_norm(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
    ) -> Result<(), Error> {
        for part in ["weight", "bias"] {
            let values = checkpoint.f32(&format!("{source}.{part}"))?;
            self.vector(format!("{name}.{part}"), values);
        }
        Ok(())
    }
}

/// The frontend's window and filterbank.
pub(crate) fn frontend(tensors: &mut Tensors) {
    tensors.vector("features.window".into(), filterbank::window());
    let mel = filterbank::mel();
    let shape = vec![MELS, mel.len() / MELS];
    tensors.add("features.mel".into(), shape, mel, Kind::F32);
}

/// The subsampler and every conformer layer.
pub(crate) fn encoder(
    checkpoint: &Checkpoint,
    export: &Export,
    tensors: &mut Tensors,
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
    fn write(&self, tensors: &mut Tensors) -> Result<(), Error> {
        let checkpoint = self.checkpoint;
        for (name, norm, module) in FEED_FORWARDS {
            tensors.copy_norm(
                checkpoint,
                &self.path(&format!("{name}.norm")),
                &self.path(norm),
            )?;
            self.ternary(
                tensors,
                &format!("{name}.up"),
                &[&format!("{module}.linear1")],
                1.0,
            )?;
            let down = [&format!("{module}.linear2")];
            self.ternary(
                tensors,
                &format!("{name}.down"),
                &down.map(String::as_str),
                HALF_STEP,
            )?;
        }
        let attention = |part: &str| format!("self_attn.{part}");
        tensors.copy_norm(
            checkpoint,
            &self.path("attention.norm"),
            &self.path("norm_self_att"),
        )?;
        let qkv = ["q_proj", "k_proj", "v_proj"].map(attention);
        self.ternary(
            tensors,
            "attention.qkv",
            &qkv.each_ref().map(String::as_str),
            1.0,
        )?;
        self.ternary(
            tensors,
            "attention.position",
            &[&attention("relative_k_proj")],
            1.0,
        )?;
        for bias in ["bias_u", "bias_v"] {
            let values = checkpoint.f32(&self.path(&attention(bias)))?;
            tensors.vector(self.path(&format!("attention.{bias}")), values);
        }
        self.ternary(tensors, "attention.out", &[&attention("o_proj")], 1.0)?;
        tensors.copy_norm(checkpoint, &self.path("conv.norm"), &self.path("norm_conv"))?;
        self.ternary(tensors, "conv.pointwise1", &["conv.pointwise_conv1"], 1.0)?;
        self.depthwise(tensors)?;
        self.ternary(tensors, "conv.pointwise2", &["conv.pointwise_conv2"], 1.0)?;
        tensors.copy_norm(checkpoint, &self.path("norm"), &self.path("norm_out"))
    }

    fn path(&self, part: &str) -> String {
        format!("{}.{part}", self.prefix)
    }

    /// The ternary modules `modules`, their rows stacked and scaled by `scale`, as linear `name`.
    fn ternary(
        &self,
        tensors: &mut Tensors,
        name: &str,
        modules: &[&str],
        scale: f32,
    ) -> Result<(), Error> {
        let mut weight = Vec::new();
        let mut cols = 0;
        for module in modules {
            let matrix = self.export.weight(self.checkpoint, &self.path(module))?;
            cols = matrix.cols;
            weight.extend(matrix.values.iter().map(|value| value * scale));
        }
        tensors.linear((&self.path(name), Kind::Ternary), cols, weight, None);
        Ok(())
    }

    /// The depthwise convolution with the BatchNorm after it folded in: its kernel tap-major,
    /// `[kernel, hidden]`, and a bias.
    fn depthwise(&self, tensors: &mut Tensors) -> Result<(), Error> {
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
        let bias = (0..hidden).map(|channel| beta[channel] - mean[channel] * scales[channel]);
        tensors.vector(self.path("conv.depthwise"), folded);
        tensors.vector(self.path("conv.depthwise_bias"), bias.collect());
        Ok(())
    }
}

/// The subsampler: its square kernels tap-major (`[taps, channels]`), its pointwise convolutions as
/// linears, and its last linear's inputs reordered from channel-major to frequency-major.
fn subsampling(checkpoint: &Checkpoint, tensors: &mut Tensors) -> Result<(), Error> {
    const SOURCE: &str = "encoder.subsampling";
    for (name, layer) in [("conv", 0), ("depthwise1", 2), ("depthwise2", 5)] {
        let weight = checkpoint.f32(&format!("{SOURCE}.layers.{layer}.weight"))?;
        let channels = weight.len() / TAPS;
        let tap_major =
            (0..TAPS * channels).map(|at| weight[(at % channels) * TAPS + at / channels]);
        tensors.vector(format!("{SOURCE}.{name}.weight"), tap_major.collect());
        let bias = checkpoint.f32(&format!("{SOURCE}.layers.{layer}.bias"))?;
        tensors.vector(format!("{SOURCE}.{name}.bias"), bias);
    }
    for (name, layer) in [("pointwise1", 3), ("pointwise2", 6)] {
        let source = format!("{SOURCE}.layers.{layer}");
        tensors.copy_linear(
            checkpoint,
            (&format!("{SOURCE}.{name}"), Kind::F32),
            &source,
        )?;
    }
    let weight = checkpoint.f32(&format!("{SOURCE}.linear.weight"))?;
    let cols = ENCODER_SHAPE.channels * SUBSAMPLED_MELS;
    let reordered = (0..weight.len()).map(|at| {
        let (row, input) = (at / cols, at % cols);
        let (frequency, channel) = (
            input / ENCODER_SHAPE.channels,
            input % ENCODER_SHAPE.channels,
        );
        weight[row * cols + channel * SUBSAMPLED_MELS + frequency]
    });
    let bias = checkpoint.f32(&format!("{SOURCE}.linear.bias"))?;
    let name = format!("{SOURCE}.out");
    tensors.linear((&name, Kind::Matrix), cols, reordered.collect(), Some(bias));
    Ok(())
}

/// The speech head: its 1-wide convolutions as linears, its context convolution's inputs
/// frame-major (oldest first).
pub(crate) fn speech(checkpoint: &Checkpoint, tensors: &mut Tensors) -> Result<(), Error> {
    tensors.copy_linear(
        checkpoint,
        ("speech.projection", Kind::F32),
        "vad_head.proj",
    )?;
    let weight = checkpoint.f32("vad_head.ctx.weight")?;
    let cols = SPEECH_KERNEL * SPEECH_HIDDEN;
    let frame_major = (0..weight.len()).map(|at| {
        let (row, input) = (at / cols, at % cols);
        let (tap, channel) = (input / SPEECH_HIDDEN, input % SPEECH_HIDDEN);
        weight[(row * SPEECH_HIDDEN + channel) * SPEECH_KERNEL + tap]
    });
    let bias = checkpoint.f32("vad_head.ctx.bias")?;
    tensors.linear(
        ("speech.context", Kind::F32),
        cols,
        frame_major.collect(),
        Some(bias),
    );
    tensors.copy_linear(checkpoint, ("speech.out", Kind::F32), "vad_head.out")
}

/// The prediction network and the joint: each LSTM layer's input and recurrent matrices side by
/// side and their biases summed, rows unit-major.
pub(crate) fn decoder(checkpoint: &Checkpoint, tensors: &mut Tensors) -> Result<(), Error> {
    let embedding = checkpoint.f32("decoder.embedding.weight")?;
    let shape = vec![embedding.len() / DECODER_HIDDEN, DECODER_HIDDEN];
    tensors.add("decoder.embedding".into(), shape, embedding, Kind::Matrix);
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
        tensors.linear(
            (&name, Kind::Matrix),
            2 * DECODER_HIDDEN,
            weight,
            Some(bias),
        );
    }
    tensors.copy_linear(
        checkpoint,
        ("decoder.projector", Kind::Matrix),
        "decoder.decoder_projector",
    )?;
    tensors.copy_linear(
        checkpoint,
        ("joint.encoder", Kind::Matrix),
        "encoder_projector",
    )?;
    tensors.copy_linear(checkpoint, ("joint.head", Kind::Matrix), "joint.head")
}
