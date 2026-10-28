//! The talker assembled from a loaded bundle, and the constants every part of the model shares.
//! Every component loads itself (`bundle::Load`) under its prefix — `talker`, `predictor`,
//! `speaker`, `codec.encoder`, `codec.decoder`, `codec.vocoder` — so this is the table of
//! contents; the compiler (`qwen3-tts-compile`) writes exactly these names.

use blocks::lm::{self, Embedding, Lm};
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture};
use bundle::{Load, Store, Tokenizer};
use kernels::Table;
use serde::Deserialize;

use crate::Error;

pub const SAMPLE_RATE: u32 = 24_000;
/// Samples per 80 ms frame.
pub const FRAME_SAMPLES: usize = 1920;
pub const HIDDEN: usize = 1024;
pub const TEXT_VOCABULARY: usize = 151_936;
pub const CODEC_VOCABULARY: usize = 3072;
/// Codes of each codebook of a frame.
pub const CODES: usize = 2048;
/// Codebooks of a frame: the talker draws the first, the code predictor the other 15.
pub const CODE_GROUPS: usize = 16;
/// Seconds of a reference clip, at the least and the most.
pub const MIN_REFERENCE_SECONDS: usize = 3;
pub const MAX_REFERENCE_SECONDS: usize = 20;

/// One frame of speech: a code for each codebook.
pub type Frame = [u32; CODE_GROUPS];

/// Ids of the text side, embedded through the text projection.
pub const TTS_PAD: u32 = 151_671;
pub const TTS_BOS: u32 = 151_672;
pub const TTS_EOS: u32 = 151_673;

/// Ids of the codec side, embedded through the talker's codec embedding.
pub const CODEC_PAD: u32 = 2148;
pub const CODEC_BOS: u32 = 2149;
/// The first code of the frame that ends speech.
pub const CODEC_EOS: u32 = 2150;
pub const CODEC_THINK: u32 = 2154;
pub const CODEC_THINK_BOS: u32 = 2156;
pub const CODEC_THINK_EOS: u32 = 2157;

/// The languages a request may speak, as the checkpoint's `codec_language_id` names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Chinese,
    English,
    German,
    Italian,
    Portuguese,
    Spanish,
    Japanese,
    Korean,
    French,
    Russian,
}

impl Language {
    pub const ALL: [Language; 10] = [
        Language::Chinese,
        Language::English,
        Language::German,
        Language::Italian,
        Language::Portuguese,
        Language::Spanish,
        Language::Japanese,
        Language::Korean,
        Language::French,
        Language::Russian,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Language::Chinese => "chinese",
            Language::English => "english",
            Language::German => "german",
            Language::Italian => "italian",
            Language::Portuguese => "portuguese",
            Language::Spanish => "spanish",
            Language::Japanese => "japanese",
            Language::Korean => "korean",
            Language::French => "french",
            Language::Russian => "russian",
        }
    }

    pub(crate) fn codec_id(self) -> u32 {
        match self {
            Language::Chinese => 2055,
            Language::English => 2050,
            Language::German => 2053,
            Language::Italian => 2070,
            Language::Portuguese => 2071,
            Language::Spanish => 2054,
            Language::Japanese => 2058,
            Language::Korean => 2064,
            Language::French => 2061,
            Language::Russian => 2069,
        }
    }
}

const NORM_EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 1_000_000.0;

/// The talker: Qwen3, 28 layers of 16 query heads over 8 key and value heads of 128, RMS-normed
/// heads, rotary embeddings of the sequence position (the checkpoint's multimodal sections turn
/// alike on a text row), and a head over the codec vocabulary.
pub const TALKER: lm::Shape = lm::Shape {
    stack: transformer::Shape {
        architecture: Architecture::Qwen3,
        hidden: HIDDEN,
        layers: 28,
        heads: 16,
        kv_heads: 8,
        head_dim: 128,
        feed_forward: 3072,
        rope_base: ROPE_BASE,
        rotation: Rotation::Sequence,
        norm_eps: NORM_EPS,
    },
    vocabulary: CODEC_VOCABULARY,
    embedding: Embedding::Own,
};

/// The code predictor's stack: the talker's layers, five of them.
pub const PREDICTOR: transformer::Shape = transformer::Shape {
    layers: 5,
    ..TALKER.stack
};

/// The talker with the text side's embeddings: `text` holds each text token's embedding already
/// through the text projection.
pub(crate) struct Talker {
    pub(crate) lm: Lm,
    text: Table,
    pub(crate) tokenizer: Tokenizer,
}

impl Talker {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            lm: Lm::load(store, "talker", TALKER)?,
            text: Table::load(store, "talker.text", [TEXT_VOCABULARY, HIDDEN])?,
            tokenizer: Tokenizer::take(store)?,
        })
    }

    /// The text embeddings of `ids`, `ids.len() × HIDDEN`, into `out`.
    pub(crate) fn text(&self, ids: &[u32], out: &mut [f32]) {
        let embeddings = self.text.matrix();
        for (&id, row) in ids.iter().zip(out.as_chunks_mut::<HIDDEN>().0.iter_mut()) {
            embeddings.dequantize_row(id as usize, row);
        }
    }
}
