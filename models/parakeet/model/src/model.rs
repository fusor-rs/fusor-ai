//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `features`, `encoder`, `speech`, `decoder`, `joint` — so this is the table of
//! contents; the compiler (`parakeet-compile`) writes exactly these names.

use blocks::conformer::{self, Encoder, Subsampling, subsampling};
use bundle::{Load, Scope, Store, load_each};
use kernels::{Linear, Table, Vector};

use crate::Error;
use crate::features::Frontend;

/// The bundle part holding the vocabulary (`vocabulary::Vocabulary`).
pub const VOCABULARY_PART: &str = "vocabulary.json";

pub const SAMPLE_RATE: u32 = 16_000;
/// Samples between two mel frames (10 ms).
pub const HOP: usize = 160;
/// Samples of a mel frame's Hann window (25 ms), centred in the FFT.
pub const WINDOW: usize = 400;
pub const FFT: usize = 512;
pub const MELS: usize = 128;
/// Frequency bins of the FFT.
pub const BINS: usize = FFT / 2 + 1;

pub const ENCODER_SHAPE: conformer::Shape = conformer::Shape {
    hidden: 1024,
    layers: 24,
    heads: 8,
    feed_forward: 4096,
    kernel: 9,
    norm_eps: 1e-5,
};
pub const SUBSAMPLING_SHAPE: subsampling::Shape = subsampling::Shape {
    mels: MELS,
    channels: 256,
    kernel: 3,
    stride: 2,
    stages: 3,
    hidden: ENCODER_SHAPE.hidden,
};
/// Seconds of audio per encoder frame.
pub const FRAME_SECONDS: f64 = (HOP * SUBSAMPLING_SHAPE.factor()) as f64 / SAMPLE_RATE as f64;

pub const DECODER_HIDDEN: usize = 640;
pub const DECODER_LAYERS: usize = 2;
/// LSTM gates per unit, in PyTorch's order: input, forget, cell, output.
pub const GATES: usize = 4;
/// The vocabulary's 8192 pieces and the blank.
pub const TOKENS: usize = 8193;
pub const BLANK: usize = 8192;
/// How many frames each duration the joint can pick advances by.
pub const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

/// Width of the speech head's convolutions.
pub const SPEECH_HIDDEN: usize = 128;
/// Encoder frames the speech head's context convolution sees, centred.
pub const SPEECH_KERNEL: usize = 5;

/// An f32 linear layer of the speech head, whose small products run as dot products.
pub(crate) struct Dense {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vector,
}

/// The speech head: per frame, a projection of the subsampled frame, a convolution over
/// `SPEECH_KERNEL` neighbouring frames (its inputs frame-major, oldest first), each followed by
/// SiLU, and a speech logit.
pub(crate) struct SpeechHead {
    pub(crate) projection: Dense,
    pub(crate) context: Dense,
    pub(crate) out: Dense,
}

/// The prediction network: token embeddings through LSTM layers and a projection. Each layer
/// is one linear over `[input; hidden]` with the two biases summed, its rows unit-major (the
/// input, forget, cell and output gates of unit 0, then of unit 1…).
pub(crate) struct Decoder {
    pub(crate) embedding: Table,
    pub(crate) layers: Vec<Linear>,
    pub(crate) projector: Linear,
}

/// The joint network: `head(relu(encoder(frame) + decoder))`, the token logits then the
/// duration logits.
pub(crate) struct Joint {
    pub(crate) encoder: Linear,
    pub(crate) head: Linear,
}

pub(crate) struct Model {
    pub(crate) frontend: Frontend,
    pub(crate) subsampling: Subsampling,
    pub(crate) encoder: Encoder,
    pub(crate) speech: SpeechHead,
    pub(crate) decoder: Decoder,
    pub(crate) joint: Joint,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let hidden = ENCODER_SHAPE.hidden;
        let speech = SpeechHead {
            projection: Dense::load(store, "speech.projection", [SPEECH_HIDDEN, hidden])?,
            context: Dense::load(
                store,
                "speech.context",
                [SPEECH_HIDDEN, SPEECH_KERNEL * SPEECH_HIDDEN],
            )?,
            out: Dense::load(store, "speech.out", [1, SPEECH_HIDDEN])?,
        };
        let lstm = [GATES * DECODER_HIDDEN, 2 * DECODER_HIDDEN];
        let decoder = Decoder {
            embedding: Table::load(store, "decoder.embedding", [TOKENS, DECODER_HIDDEN])?,
            layers: load_each(store, "decoder.layers", DECODER_LAYERS, |store, name| {
                Linear::load(store, name, lstm)
            })?,
            projector: Linear::load(store, "decoder.projector", [DECODER_HIDDEN; 2])?,
        };
        let logits = TOKENS + DURATIONS.len();
        let joint = Joint {
            encoder: Linear::load(store, "joint.encoder", [DECODER_HIDDEN, hidden])?,
            head: Linear::load(store, "joint.head", [logits, DECODER_HIDDEN])?,
        };
        Ok(Self {
            frontend: Frontend::load(store, "features", ())?,
            subsampling: Subsampling::load(store, "encoder.subsampling", SUBSAMPLING_SHAPE)?,
            encoder: Encoder::load(store, "encoder", ENCODER_SHAPE)?,
            speech,
            decoder,
            joint,
        })
    }
}

/// `p.weight`, `[outputs, inputs]` in f32, and `p.bias`.
impl Load for Dense {
    type Shape = [usize; 2];

    fn load(store: &mut Store, prefix: &str, shape: [usize; 2]) -> Result<Self, Error> {
        let mut scope = Scope::new(store, prefix);
        let weight: Table = scope.load("weight", shape)?;
        Ok(Self {
            weight: weight.dequantize(),
            bias: scope.load("bias", shape[0])?,
        })
    }
}
