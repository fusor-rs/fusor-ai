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

use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error};
use kernels::{DType, Tensor};
use parakeet::model::{TOKENS, VOCABULARY_PART};
use serde::Deserialize;

use crate::tables::{Kind, Tensors};

pub const REPO: &str = "moondream/parakeet-redux";
pub const REVISION: &str = "2bf128600aac4b16946f7ed8372e56117fe5e23b";
const WEIGHTS: &str = "model.safetensors";
const TOKENIZER: &str = "tokenizer.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The encoder's ternary projections `T2b128`, the other large matrices 7-bit.
    Ternary,
}

impl Profile {
    pub const ALL: [Self; 2] = [Self::F32, Self::Ternary];

    /// The name in `manifest.json` and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Ternary => "ternary",
        }
    }

    fn dtype(self, kind: Kind) -> DType {
        match (self, kind) {
            (Self::Ternary, Kind::Ternary) => DType::T2b128,
            (Self::Ternary, Kind::Matrix) => DType::Q7b32,
            (Self::F32, _) | (_, Kind::F32) => DType::F32,
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.name() == name)
            .ok_or_else(|| Error::Format(format!("no profile {name:?} (f32 or ternary)")))
    }
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let ternary = ternary::Export::read(checkpoint_dir)?;
    let mut tensors = Tensors::default();
    tables::frontend(&mut tensors);
    tables::encoder(&checkpoint, &ternary, &mut tensors)?;
    tables::speech(&checkpoint, &mut tensors)?;
    tables::decoder(&checkpoint, &mut tensors)?;
    let mut writer = BundleWriter::create(out)?;
    for entry in tensors.into_entries() {
        if entry.values.len() != entry.shape.iter().product::<usize>() {
            let (name, shape) = (entry.name, entry.shape);
            return Err(Error::Format(format!("{name}: shape {shape:?}")));
        }
        let tensor = Tensor::encode(profile.dtype(entry.kind), entry.shape, &entry.values);
        writer.add_tensor(&entry.name, &tensor)?;
    }
    writer.add_blob(
        VOCABULARY_PART,
        &vocabulary(&checkpoint_dir.join(TOKENIZER))?,
    )?;
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
    vocab: std::collections::HashMap<String, usize>,
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
    let json = std::fs::read(path).map_err(|source| Error::Io {
        path: path.into(),
        source,
    })?;
    let tokenizer: Tokenizer = serde_json::from_slice(&json)
        .map_err(|error| Error::Format(format!("{}: {error}", path.display())))?;
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
