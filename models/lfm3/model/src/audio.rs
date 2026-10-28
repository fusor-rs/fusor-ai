//! The audio of LFM3-Audio-2B and LFM3-Omni-2B. `Hearing` hears a clip of 16 kHz speech: its
//! log-mel features as the checkpoint's feature extractor computes them (`media::mel`'s power,
//! its `log10` floored at −12, then `(x + 4) / 4`), then `blocks::audio`'s encoder (`conformer`'s
//! subsampler with SiLU, 8 frames to a token, LFM2's stack: `blocks::interleaved`, ungated
//! attention over a window of 32 tokens, and its final norm) and projector into the language
//! model's tokens. `Voice` speaks: the depth decoder (`blocks::depth`)
//! draws a frame's 12 codes from the language model's state, the frame's codebook embeddings,
//! summed, are the row the language model reads next, and the codec (`blocks::codec`) turns
//! frames into 24 kHz samples, 1,920 a frame.
//!
//! Tensors: `audio`'s (`blocks::audio`); `depth`'s (`blocks::depth`), `codebooks` (`[12 · 2017,
//! hidden]`, codebook-major), and `codec`'s (`blocks::codec`).

use blocks::audio::{self as encoder, Encoder};
use blocks::codec::{self, Decoder};
use blocks::conformer::subsampling::{self, Activation as SubsamplingActivation};
use blocks::depth::{self, Depth};
use blocks::hybrid::AttentionShape;
use blocks::hybrid::{Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Mixer, Norms, Window};
use blocks::rotary::Rotation;
use bundle::{Load, Store};
use kernels::Table;
use media::mel::{self, MelSpectrogram};
use team::{SharedBuf, Team};

use crate::Error;
use crate::model::{HIDDEN, NORM_EPS};
use crate::sampling::FrameSampler;

/// The rate of the speech heard.
pub const SAMPLE_RATE: u32 = 16_000;
/// The rate of the speech spoken.
pub const SPEECH_RATE: u32 = 24_000;
/// The longest clip heard, in seconds.
pub const MAX_CLIP_SECONDS: usize = 60;
const MAX_CLIP_SAMPLES: usize = MAX_CLIP_SECONDS * SAMPLE_RATE as usize;

pub const MEL: mel::Shape = mel::Shape {
    sample_rate: SAMPLE_RATE,
    fft: 400,
    hop: 160,
    mels: 128,
};
const LOG_FLOOR: f32 = -12.0;
const NORM_OFFSET: f32 = 4.0;
const NORM_DIVISOR: f32 = 4.0;

const ENCODER_HIDDEN: usize = 768;
const ENCODER_HEAD_DIM: usize = 96;
const AUDIO_ROPE_BASE: f32 = 1_000_000.0;

pub const SUBSAMPLING: subsampling::Shape = subsampling::Shape {
    mels: MEL.mels,
    channels: 256,
    kernel: 3,
    stride: 2,
    stages: 3,
    activation: SubsamplingActivation::Silu,
    padding_frames: 0,
    hidden: ENCODER_HIDDEN,
};
/// Output frames a thread of the subsampler computes at a time.
const SUBSAMPLING_BLOCK_FRAMES: usize = 8;

/// The encoder's `layer_types`, its full attention layers attending over its window.
const ENCODER_MIXERS: [Mixer; 17] = {
    use Mixer::{Conv, Window};
    [
        Conv, Conv, Window, Conv, Conv, Window, Conv, Conv, Window, Conv, Conv, Window, Conv, Conv,
        Window, Window, Window,
    ]
};

/// The encoder's window: 32 tokens, each rotated whole.
const ENCODER_WINDOW: Window = Window {
    rows: 32,
    rotary_dim: ENCODER_HEAD_DIM,
    rope_base: AUDIO_ROPE_BASE,
    rotation: Rotation::Sequence,
};

/// LFM2's stack over the subsampled frames: 17 layers, gated short convolutions of 3 taps
/// between ungated attention of 8 query heads over 4 key and value heads of 96, causal over a
/// window of 32 tokens.
pub const ENCODER_STACK: interleaved::Shape = interleaved::Shape {
    hidden: ENCODER_HIDDEN,
    mixers: &ENCODER_MIXERS,
    attention: AttentionShape {
        heads: 8,
        kv_heads: 4,
        head_dim: ENCODER_HEAD_DIM,
        rotary_dim: ENCODER_HEAD_DIM,
        rope_base: AUDIO_ROPE_BASE,
        rotation: Rotation::Sequence,
        head_norm: HeadNorm::Rms,
        gate: Gate::None,
    },
    window: ENCODER_WINDOW,
    conv_kernel: 3,
    feed_forward: 1792,
    activation: Activation::Silu,
    norms: Norms::Pre,
    norm_eps: NORM_EPS,
};

/// The encoder: the subsampler, the stack and the projector into the language model's tokens.
pub const ENCODER: encoder::Shape = encoder::Shape {
    subsampling: SUBSAMPLING,
    stack: ENCODER_STACK,
    hidden: HIDDEN,
    projector_eps: NORM_EPS,
};

pub const CODEBOOKS: usize = 12;
/// Codes of a codebook: the codec's 2,016, then the end of speech.
pub const CODES: usize = 2017;
/// The code that ends a reply's speech.
pub const END_CODE: u32 = 2016;
/// A spoken frame: one code per codebook.
pub type Frame = [u32; CODEBOOKS];
/// Codebooks revealed per pass of the depth decoder.
const REVEALED_PER_PASS: usize = 4;

const DEPTH_HIDDEN: usize = 1024;
const DEPTH_HEAD_DIM: usize = 32;
const DEPTH_MIXERS: [Mixer; 3] = [Mixer::Full; 3];

/// The depth decoder: 3 full attention layers among a frame's 12 slots, 32 query heads over 8
/// key and value heads of 32, each head RMS-normed and gated by one sigmoid.
pub const DEPTH: depth::Shape = depth::Shape {
    codebooks: CODEBOOKS,
    vocabulary: CODES,
    input: HIDDEN,
    stack: interleaved::Shape {
        hidden: DEPTH_HIDDEN,
        mixers: &DEPTH_MIXERS,
        attention: AttentionShape {
            heads: 32,
            kv_heads: 8,
            head_dim: DEPTH_HEAD_DIM,
            rotary_dim: DEPTH_HEAD_DIM,
            rope_base: AUDIO_ROPE_BASE,
            rotation: Rotation::Sequence,
            head_norm: HeadNorm::Rms,
            gate: Gate::Headwise,
        },
        window: ENCODER_WINDOW,
        conv_kernel: 0,
        feed_forward: 2816,
        activation: Activation::Silu,
        norms: Norms::Pre,
        norm_eps: NORM_EPS,
    },
};

const CODEC_HIDDEN: usize = 512;
const CODEC_HEAD_DIM: usize = 64;
const CODEC_MIXERS: [Mixer; 8] = {
    use Mixer::{Conv, Window};
    [Conv, Conv, Window, Conv, Window, Conv, Window, Conv]
};

/// The codec's decoder: finite scalar quantization of levels 8, 7, 6 and 6 per codebook, each
/// frame 6 rows of an LFM3 stack of 8 layers (gated short convolutions of 7 taps, gated attention
/// over a window of 30 rows), then an inverse STFT of 1,280 points every 320 samples.
pub const CODEC: codec::Shape = codec::Shape {
    codebooks: CODEBOOKS,
    levels: &[8, 7, 6, 6],
    latent: 256,
    stack: interleaved::Shape {
        hidden: CODEC_HIDDEN,
        mixers: &CODEC_MIXERS,
        attention: AttentionShape {
            heads: 8,
            kv_heads: 2,
            head_dim: CODEC_HEAD_DIM,
            rotary_dim: CODEC_HEAD_DIM,
            rope_base: AUDIO_ROPE_BASE,
            rotation: Rotation::Sequence,
            head_norm: HeadNorm::Rms,
            gate: Gate::Headwise,
        },
        window: Window {
            rows: 30,
            rotary_dim: CODEC_HEAD_DIM,
            rope_base: AUDIO_ROPE_BASE,
            rotation: Rotation::Sequence,
        },
        conv_kernel: 7,
        feed_forward: 2304,
        activation: Activation::Silu,
        norms: Norms::Sandwich,
        norm_eps: NORM_EPS,
    },
    upsample: 6,
    n_fft: 1280,
    hop: 320,
};

/// The rows a clip of `samples` samples takes in the prompt.
pub fn tokens(samples: usize) -> usize {
    ENCODER.tokens(MEL.frames(padded(samples)))
}

/// A clip's samples once the feature extractor pads it: to a whole window, then to whole hops.
fn padded(samples: usize) -> usize {
    match samples.checked_sub(MEL.fft) {
        Some(past) => samples + (MEL.hop - past % MEL.hop) % MEL.hop,
        None => MEL.fft,
    }
}

/// Whether the clip `samples` can be heard.
///
/// # Errors
/// When it is longer than `MAX_CLIP_SECONDS`.
pub fn check(samples: &[f32]) -> Result<(), Error> {
    if samples.len() > MAX_CLIP_SAMPLES {
        return Err(Error::Request(format!(
            "a clip of {:.1} s: the longest heard is {MAX_CLIP_SECONDS} s",
            samples.len() as f64 / f64::from(SAMPLE_RATE)
        )));
    }
    Ok(())
}

/// Whether a frame ends the reply's speech.
pub fn ends(frame: &Frame) -> bool {
    frame.contains(&END_CODE)
}

pub struct Hearing {
    spectrogram: MelSpectrogram,
    encoder: Encoder,
    buffers: encoder::Buffers,
}

impl Hearing {
    pub(crate) fn load(store: &mut Store, threads: usize) -> Result<Self, Error> {
        let mel_frames = MEL.frames(MAX_CLIP_SAMPLES);
        Ok(Self {
            spectrogram: MelSpectrogram::new(MEL),
            encoder: Encoder::load(store, "audio", ENCODER)?,
            buffers: ENCODER.buffers(mel_frames, SUBSAMPLING_BLOCK_FRAMES, threads),
        })
    }

    /// A clip's log-mel features, `frames × mels`.
    ///
    /// # Errors
    /// When the clip is longer than `MAX_CLIP_SECONDS`.
    pub fn features(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        check(samples)?;
        let mut clip = samples.to_vec();
        clip.resize(padded(samples.len()), 0.0);
        let mut features = Vec::new();
        self.spectrogram.power(&clip, &mut features);
        for value in &mut features {
            *value = (value.log10().max(LOG_FLOOR) + NORM_OFFSET) / NORM_DIVISOR;
        }
        Ok(features)
    }

    /// The encoder's tokens of `features` (`features`' output), final-normed, `tokens × 768`.
    pub fn encode(&mut self, team: &Team, features: &[f32]) -> Vec<f32> {
        let len = ENCODER.tokens(features.len() / MEL.mels) * ENCODER_HIDDEN;
        self.buffers.clear();
        let (encoder, buffers) = (&self.encoder, &self.buffers);
        team.run(|ctx| {
            encoder.encode(ctx, buffers, features);
        });
        buffers.encoded.read()[..len].to_vec()
    }

    /// The language model's tokens of `encoded` (`encode`'s output), `tokens × hidden`.
    pub fn project(&mut self, team: &Team, encoded: &[f32]) -> Vec<f32> {
        let frames = encoded.len() / ENCODER_HIDDEN;
        self.buffers.encoded.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (encoder, buffers) = (&self.encoder, &self.buffers);
        team.run(|ctx| encoder.project(ctx, buffers, frames));
        buffers.tokens.read()[..frames * HIDDEN].to_vec()
    }

    /// A clip's tokens in the language model, through every stage in one team run.
    ///
    /// # Errors
    /// When the clip is longer than `MAX_CLIP_SECONDS`.
    pub fn hear(&mut self, team: &Team, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let features = self.features(samples)?;
        let len = ENCODER.tokens(features.len() / MEL.mels) * HIDDEN;
        self.buffers.clear();
        let (encoder, buffers) = (&self.encoder, &self.buffers);
        team.run(|ctx| encoder.forward(ctx, buffers, &features));
        Ok(buffers.tokens.read()[..len].to_vec())
    }
}

pub struct Voice {
    depth: Depth,
    buffers: depth::Buffers,
    /// Every codebook's code embeddings, codebook-major.
    codebooks: Table,
    pub(crate) codec: Decoder,
}

impl Voice {
    pub(crate) fn load(store: &mut Store, threads: usize) -> Result<Self, Error> {
        Ok(Self {
            depth: Depth::load(store, "depth", DEPTH)?,
            buffers: depth::Buffers::new(&DEPTH, threads),
            codebooks: Table::load(store, "codebooks", [CODEBOOKS * CODES, HIDDEN])?,
            codec: Decoder::load(store, "codec", CODEC)?,
        })
    }

    /// The next frame, drawn from the language model's final-normed state `hidden` (one row).
    pub fn frame(&mut self, team: &Team, hidden: &SharedBuf, sampler: &mut FrameSampler) -> Frame {
        let mut codes = [None; CODEBOOKS];
        let order = sampler.order();
        let (depth, buffers) = (&self.depth, &self.buffers);
        team.run(|ctx| depth.slots(ctx, buffers, hidden));
        for pass in order.chunks(REVEALED_PER_PASS) {
            let (depth, buffers) = (&self.depth, &self.buffers);
            team.run(|ctx| depth.score(ctx, buffers, &codes, pass));
            for &codebook in pass {
                let logits = self.buffers.predictions[codebook].logits.get_mut();
                codes[codebook] = Some(sampler.draw(codebook, logits));
            }
        }
        codes.map(|code| code.expect("every pass reveals its codebooks"))
    }

    /// Every codebook's logits of the first pass over a frame, all masked, from `hidden`,
    /// `codebooks × codes`.
    pub fn logits(&self, team: &Team, hidden: &SharedBuf) -> Vec<f32> {
        let every: Vec<usize> = (0..CODEBOOKS).collect();
        let (depth, buffers) = (&self.depth, &self.buffers);
        team.run(|ctx| {
            depth.slots(ctx, buffers, hidden);
            depth.score(ctx, buffers, &[None; CODEBOOKS], &every);
        });
        let predictions = buffers.predictions.iter();
        predictions
            .flat_map(|prediction| prediction.logits.read().iter().copied())
            .collect()
    }

    /// The row the language model reads for `frame`: its codes' embeddings, summed.
    pub fn embed(&self, frame: &Frame, out: &mut [f32]) {
        let matrix = self.codebooks.matrix();
        let mut code = vec![0.0; out.len()];
        out.fill(0.0);
        for (codebook, &value) in frame.iter().enumerate() {
            matrix.dequantize_row(codebook * CODES + value as usize, &mut code);
            for (sum, value) in out.iter_mut().zip(&code) {
                *sum += value;
            }
        }
    }
}
