//! The checkpoint's files and the bundle's parts, and every network loaded from a bundle with
//! the settings `meta.json` holds: the detector's long side, the base recognizer's window, the
//! vocabularies, the script groups and the language models' weights.

use std::collections::HashMap;

use bundle::{Load, Store};
use serde::Deserialize;

use crate::Error;
use crate::classifier::Classifier;
use crate::detector::{self, Detector};
use crate::lm::{Ngram, Weights};
use crate::math::MathTokens;
use crate::math::decoder::Decoder;
use crate::math::encoder::Encoder;
use crate::order::{self, Order};
use crate::orient::Orientation;
use crate::recognizer::{LINE_HEIGHT, Recognizer, RecognizerShape};
use crate::vocab::Vocab;

/// Each network's ONNX file and the prefix its tensors take in the bundle.
pub const NETWORKS: [(&str, &str); 11] = [
    ("det.onnx", detector::PREFIX),
    ("orient.onnx", "orient"),
    ("script.onnx", "script"),
    ("rec.onnx", "rec"),
    ("rec_cjk.onnx", "rec_cjk"),
    ("rec_rtl.onnx", "rec_rtl"),
    ("rec_world.onnx", "rec_world"),
    ("order.onnx", order::PREFIX),
    ("mathcls.onnx", "mathcls"),
    ("math_enc.onnx", "math_enc"),
    ("math_dec_step.onnx", "math_dec"),
];
pub const META: &str = "meta.json";
pub const MATH_META: &str = "math_meta.json";
/// Each language model's checkpoint file and its part in the bundle; the base recognizer's
/// first, then the groups' (`lm_<group>`).
pub const LANGUAGE_MODELS: [(&str, &str); 3] = [
    ("lm.npz", "lm.bin"),
    ("lm_cjk.npz", "lm_cjk.bin"),
    ("lm_world.npz", "lm_world.bin"),
];
/// The order network's Fourier frequencies, in the bundle.
pub const ORDER_FREQUENCIES: &str = "order.frequencies";
/// The script group the base recognizer reads.
pub const BASE: &str = "base";
const VOCAB_PREFIX: &str = "vocab_";
/// The base recognizer's language-model weights and gate, unless `meta.json` sets them.
const LM_ALPHA: f64 = 0.5;
const LM_BETA: f64 = 1.5;
const LM_GATE: f64 = 0.9;

#[derive(Deserialize)]
struct Meta {
    det_long_side: usize,
    rec_height: usize,
    #[serde(default = "one")]
    rec_window: usize,
    vocab: String,
    script_groups: Vec<String>,
    #[serde(default)]
    lm_params: HashMap<String, LmParams>,
    #[serde(flatten)]
    rest: HashMap<String, serde_json::Value>,
}

fn one() -> usize {
    1
}

#[derive(Deserialize, Default, Clone, Copy)]
struct LmParams {
    alpha: Option<f64>,
    beta: Option<f64>,
    gate: Option<f64>,
}

/// A recognizer, its vocabulary and its language model.
pub struct Group {
    pub name: String,
    pub recognizer: Recognizer,
    pub vocab: Vocab,
    pub lm: Option<LanguageModel>,
}

/// A language model, its weights, and the confidence under which it rereads a line.
pub struct LanguageModel {
    pub ngram: Ngram,
    pub weights: Weights,
    pub gate: f64,
}

/// The math reader's networks and tokens.
pub struct MathModels {
    pub classifier: Classifier,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub tokens: MathTokens,
}

pub struct Model {
    pub detector: Detector,
    pub orientation: Orientation,
    pub script: Classifier,
    /// The base recognizer first, then the groups in `meta.json`'s order.
    pub groups: Vec<Group>,
    /// The script classifier's classes, each a group's name.
    pub script_groups: Vec<String>,
    pub order: Order,
    pub math: MathModels,
    pub long_side: usize,
    /// The base recognizer's window, in bands.
    pub window: usize,
}

impl Model {
    pub fn load(store: &mut Store) -> Result<Self, Error> {
        let meta: Meta = serde_json::from_slice(&store.take_blob(META)?)
            .map_err(|error| Error::Format(format!("{META}: {error}")))?;
        if meta.rec_height != LINE_HEIGHT {
            return Err(Error::Format(format!(
                "{META}: lines {} high, not {LINE_HEIGHT}",
                meta.rec_height
            )));
        }
        let mut groups = vec![group(
            store,
            (BASE, &meta.vocab),
            meta.rec_window,
            &meta.lm_params,
        )?];
        for (key, vocab) in &meta.rest {
            let Some(name) = key.strip_prefix(VOCAB_PREFIX) else {
                continue;
            };
            let vocab = vocab
                .as_str()
                .ok_or_else(|| Error::Format(format!("{META}: {key} is not a string")))?;
            groups.push(group(store, (name, vocab), 1, &meta.lm_params)?);
        }
        groups[1..].sort_by_key(|group| {
            let class = meta
                .script_groups
                .iter()
                .position(|name| *name == group.name);
            class.unwrap_or(meta.script_groups.len())
        });
        let math_meta = store.take_blob(MATH_META)?;
        Ok(Self {
            detector: Detector::load(store, detector::PREFIX, ())?,
            orientation: Orientation::load(store, "orient.m", ())?,
            script: Classifier::load(store, "script.m", meta.script_groups.len())?,
            groups,
            script_groups: meta.script_groups,
            order: Order::load(store, order::PREFIX, ())?,
            math: MathModels {
                classifier: Classifier::load(store, "mathcls.m", 1)?,
                encoder: Encoder::load(store, "math_enc.enc", ())?,
                decoder: Decoder::load(store, "math_dec.c.dec", ())?,
                tokens: MathTokens::parse(&math_meta)?,
            },
            long_side: meta.det_long_side,
            window: meta.rec_window,
        })
    }
}

/// The recognizer of group `name` (`rec` for the base one, `rec_<name>` for a group's), with
/// its vocabulary and language model.
fn group(
    store: &mut Store,
    (name, vocab): (&str, &str),
    window: usize,
    params: &HashMap<String, LmParams>,
) -> Result<Group, Error> {
    let vocab = Vocab::parse(vocab)?;
    let network = if name == BASE {
        "rec".to_string()
    } else {
        format!("rec_{name}")
    };
    let shape = RecognizerShape {
        classes: vocab.classes(),
        window,
    };
    let recognizer = Recognizer::load(store, &format!("{network}.rec"), shape)?;
    let part = if name == BASE {
        "lm.bin".to_string()
    } else {
        format!("lm_{name}.bin")
    };
    let lm = match LANGUAGE_MODELS.iter().find(|(_, file)| *file == part) {
        Some(_) => {
            let ngram = Ngram::from_bytes(&store.take_blob(&part)?)?;
            // A group without settings of its own takes the base recognizer's.
            let set = params.get(name).copied().unwrap_or_default();
            Some(LanguageModel {
                ngram,
                weights: Weights {
                    alpha: set.alpha.unwrap_or(LM_ALPHA),
                    beta: set.beta.unwrap_or(LM_BETA),
                },
                gate: set.gate.unwrap_or(LM_GATE),
            })
        }
        None => None,
    };
    Ok(Group {
        name: name.to_string(),
        recognizer,
        vocab,
        lm,
    })
}
