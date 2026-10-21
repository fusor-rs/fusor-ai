//! The model assembled from a loaded bundle. Every component loads itself (`bundle::Load`) under
//! its prefix — `features`, `encoder`, `speech`, `decoder`, `joint` — so this is the table of
//! contents; the compiler (`parakeet-compile`) writes exactly these names.

use blocks::conformer::{self, Encoder, Subsampling};
use bundle::{Load, Store};
use kernels::{Linear, MatrixStorage, Tensor};

use crate::Error;

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
    mels: MELS,
    channels: 256,
    subsampling_kernel: 3,
    subsampling_stride: 2,
    subsampling_stages: 3,
    hidden: 1024,
    layers: 24,
    heads: 8,
    feed_forward: 4096,
    kernel: 9,
    norm_eps: 1e-5,
};
/// Seconds of audio per encoder frame.
pub const FRAME_SECONDS: f64 =
    (HOP * ENCODER_SHAPE.subsampling_factor()) as f64 / SAMPLE_RATE as f64;

pub const DECODER_HIDDEN: usize = 640;
pub const DECODER_LAYERS: usize = 2;
/// The vocabulary's 8192 pieces and the blank.
pub const TOKENS: usize = 8193;
pub const BLANK: usize = 8192;
/// How many frames each duration the joint can pick advances by.
pub const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

/// Width of the speech head's convolutions.
pub const SPEECH_HIDDEN: usize = 128;
/// Encoder frames the speech head's context convolution sees, centred.
pub const SPEECH_KERNEL: usize = 5;

/// The speech head, in f32: per frame, a projection of the subsampled frame, a convolution over
/// `SPEECH_KERNEL` neighbouring frames (its inputs frame-major, oldest first), each followed by
/// SiLU, and a speech logit.
pub(crate) struct SpeechHead {
    pub(crate) projection: Linear,
    pub(crate) context: Linear,
    pub(crate) out: Linear,
}

/// The prediction network: token embeddings through LSTM layers and a projection. Each layer
/// is one linear over `[input; hidden]` with the two biases summed, its rows unit-major (the
/// input, forget, cell and output gates of unit 0, then of unit 1…).
pub(crate) struct Decoder {
    pub(crate) embedding: Tensor,
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
    /// The Hann window centred in `FFT` samples.
    pub(crate) window: Tensor,
    /// The Slaney mel filterbank, `[MELS, BINS]`.
    pub(crate) mel: Tensor,
    pub(crate) subsampling: Subsampling,
    pub(crate) encoder: Encoder,
    pub(crate) speech: SpeechHead,
    pub(crate) decoder: Decoder,
    pub(crate) joint: Joint,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let hidden = ENCODER_SHAPE.hidden;
        let context = [SPEECH_HIDDEN, SPEECH_KERNEL * SPEECH_HIDDEN];
        let speech = SpeechHead {
            projection: f32_linear(store, "speech.projection", [SPEECH_HIDDEN, hidden])?,
            context: f32_linear(store, "speech.context", context)?,
            out: f32_linear(store, "speech.out", [1, SPEECH_HIDDEN])?,
        };
        let embedding = Tensor::load(store, "decoder.embedding", ())?;
        check(&embedding, "decoder.embedding", [TOKENS, DECODER_HIDDEN])?;
        let layers = (0..DECODER_LAYERS)
            .map(|layer| {
                let shape = [4 * DECODER_HIDDEN, 2 * DECODER_HIDDEN];
                linear(store, &format!("decoder.layers.{layer}"), shape)
            })
            .collect::<Result<_, _>>()?;
        let decoder = Decoder {
            embedding,
            layers,
            projector: linear(store, "decoder.projector", [DECODER_HIDDEN; 2])?,
        };
        let logits = TOKENS + DURATIONS.len();
        let joint = Joint {
            encoder: linear(store, "joint.encoder", [DECODER_HIDDEN, hidden])?,
            head: linear(store, "joint.head", [logits, DECODER_HIDDEN])?,
        };
        let window = Tensor::load(store, "features.window", ())?;
        check(&window, "features.window", [1, FFT])?;
        let mel = Tensor::load(store, "features.mel", ())?;
        check(&mel, "features.mel", [MELS, BINS])?;
        Ok(Self {
            window,
            mel,
            subsampling: Subsampling::load(store, "encoder.subsampling", ENCODER_SHAPE)?,
            encoder: Encoder::load(store, "encoder", ENCODER_SHAPE)?,
            speech,
            decoder,
            joint,
        })
    }
}

/// Linear `name`, whose weight must be `[outputs, inputs]`.
fn linear(store: &mut Store, name: &str, shape: [usize; 2]) -> Result<Linear, Error> {
    let linear = Linear::load(store, name, ())?;
    let matrix = linear.matrix();
    if [matrix.rows, matrix.cols] != shape {
        let found = [matrix.rows, matrix.cols];
        return Err(Error::Format(format!("{name}: {found:?}, not {shape:?}")));
    }
    Ok(linear)
}

/// Linear `name` as `linear` loads it, with f32 weights.
fn f32_linear(store: &mut Store, name: &str, shape: [usize; 2]) -> Result<Linear, Error> {
    let linear = linear(store, name, shape)?;
    match linear.matrix().storage {
        MatrixStorage::F32(_) => Ok(linear),
        MatrixStorage::Quant { dtype, .. } => {
            Err(Error::Format(format!("{name}: {dtype:?}, not f32")))
        }
    }
}

/// Checks that tensor `name` is `[rows, cols]` (a vector when `rows` is 1).
fn check(tensor: &Tensor, name: &str, [rows, cols]: [usize; 2]) -> Result<(), Error> {
    let fits = match tensor.shape[..] {
        [len] => rows == 1 && len == cols && tensor.values().is_ok(),
        [found_rows, found_cols] => [found_rows, found_cols] == [rows, cols],
        _ => false,
    };
    if fits {
        return Ok(());
    }
    Err(Error::Format(format!(
        "{name}: shape {:?}, not [{rows}, {cols}]",
        tensor.shape
    )))
}
