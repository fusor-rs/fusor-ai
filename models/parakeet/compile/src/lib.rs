//! `parakeet compile`: the checkpoint folder (`model.safetensors`, `ternary.json`,
//! `tokenizer.json`) → a browser bundle.
//!
//! The encoder's ternary projections are unpacked from the export's base-3 codes and kept
//! ternary (`T2b128`) or widened (the `f32` profile). Every path that does not depend on the
//! audio is folded here: each layer's queries, keys and values into one projection; the
//! feed-forwards' half steps into their `down` projections; the convolution module's BatchNorm
//! into its depthwise kernel and a bias; each LSTM layer's two matrices and two biases into one,
//! its rows unit-major; the subsampler's last linear's inputs reordered frequency-major. The
//! frontend's Hann window and Slaney filterbank are computed here, and the tokenizer becomes the
//! vocabulary's pieces.

mod filterbank;
mod tables;
mod ternary;

use std::collections::HashMap;
use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors, read_json};
use kernels::DType;
use parakeet::model::{TOKENS, VOCABULARY_PART};
use serde::Deserialize;

const REPO: &str = "moondream/parakeet-redux";
const REVISION: &str = "2bf128600aac4b16946f7ed8372e56117fe5e23b";
const WEIGHTS: &str = "model.safetensors";
const TOKENIZER: &str = "tokenizer.json";
/// The conformer layers, whose matrices are the export's ternary projections.
const TERNARY_PREFIX: &str = "encoder.layers.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The encoder's ternary projections `T2b128`, the other large matrices 7-bit.
    Ternary,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Ternary];
    /// The frontend, the subsampler's pointwise convolutions and the speech head.
    const ALWAYS_F32: &'static [&'static str] =
        &["features.", "encoder.subsampling.pointwise", "speech."];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Ternary => "ternary",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Ternary if name.starts_with(TERNARY_PREFIX) => DType::T2b128,
            Self::Ternary => DType::Q7b32,
        }
    }
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let ternary = ternary::Export::read(checkpoint_dir)?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    tables::frontend(&mut tensors)?;
    tables::encoder(&checkpoint, &ternary, &mut tensors)?;
    tables::speech(&checkpoint, &mut tensors)?;
    tables::decoder(&checkpoint, &mut tensors)?;
    let vocabulary = vocabulary(&checkpoint_dir.join(TOKENIZER))?;
    writer.add_blob(VOCABULARY_PART, &vocabulary)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}

#[derive(Deserialize)]
struct Tokenizer {
    model: TokenizerModel,
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, usize>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    special: bool,
}

/// The vocabulary blob (`parakeet::vocabulary::Vocabulary`): the piece of every token id, `null`
/// for the special tokens, which decoding skips.
fn vocabulary(path: &Path) -> Result<Vec<u8>, Error> {
    let tokenizer: Tokenizer = read_json(path)?;
    let mut pieces: Vec<Option<String>> = vec![None; TOKENS];
    let mut place = |id: usize, piece: Option<String>| match pieces.get_mut(id) {
        Some(slot) => {
            *slot = piece;
            Ok(())
        }
        None => Err(Error::Format(format!(
            "token {id} past the {TOKENS} tokens"
        ))),
    };
    for (piece, id) in tokenizer.model.vocab {
        place(id, Some(piece))?;
    }
    for token in tokenizer.added_tokens {
        place(token.id, (!token.special).then_some(token.content))?;
    }
    serde_json::to_vec(&pieces).map_err(|error| Error::Format(format!("vocabulary: {error}")))
}
