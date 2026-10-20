//! `phonon compile`: the checkpoint folder (`model.safetensors`, `tokenizer.json`,
//! `default-voice.safetensors`, and the catalog voices under `voices/`) → a browser bundle.
//!
//! Every path that does not depend on the request is folded here: the voices through the speaker
//! projection; the conditions (one speaker, unchanged rate) into the input projection's bias;
//! the one-step flow head's time embedding into its condition bias; the flow head's LayerNorm
//! affines, `1 + scale` and shifts into one modulation matrix (`blocks::adaln_mlp`); Mimi's latent
//! denormalization into its quantizer projection and its layer scales into the transformer's
//! output rows; SEANet's convs and transposed convs into causal-conv matrices
//! (`blocks::seanet`). Weights are then quantized per profile and packed by `BundleWriter`.

mod tables;

use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error};
use kernels::{BLOCK, DType, Tensor};
use phonon::model::{SEANET_SHAPE, TOKENIZER_PART};

use crate::tables::Tensors;

pub const REPO: &str = "gradium/phonon-31m-en";
pub const REVISION: &str = "52433c46975bfa13144c56204c9f4ee1c5b148aa";
/// The `gradium/phonon-voices` revision whose voices the bundle ships.
pub const VOICES_REVISION: &str = "0ee17205f254b5abc43b266c8a0f215565c352f7";
const WEIGHTS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
}

impl Profile {
    pub const ALL: [Self; 2] = [Self::F32, Self::Q7];

    /// The name in `manifest.json` and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.name() == name)
            .ok_or_else(|| Error::Format(format!("no profile {name:?} (f32 or q7)")))
    }
}

/// Small or precision-critical matrices kept in f32 in every profile.
const ALWAYS_F32: [&str; 7] = [
    "lm.input.",
    "lm.eos.",
    "flow.input.",
    "flow.final.",
    "mimi.latent.",
    "mimi.upsample",
    "mimi.decoder.final.",
];

/// The 1-wide conv that ends SEANet's last residual unit: its int8 activations (one block of
/// channels per sample) set the noise floor of quiet passages, about −70 dBFS against −82 in f32.
fn last_residual(name: &str) -> bool {
    let last_stage = SEANET_SHAPE.ratios.len() - 1;
    name.starts_with(&format!("mimi.decoder.stages.{last_stage}.residual2."))
}

fn dtype(profile: Profile, name: &str, shape: &[usize]) -> DType {
    let matrix = shape.len() == 2 && shape[1].is_multiple_of(BLOCK);
    let kept = ALWAYS_F32.iter().any(|prefix| name.starts_with(prefix)) || last_residual(name);
    if !matrix || profile == Profile::F32 || kept {
        DType::F32
    } else {
        DType::Q7b32
    }
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut tensors = Tensors::default();
    tables::flow_lm(&checkpoint, &mut tensors)?;
    tables::flow_head(&checkpoint, &mut tensors)?;
    tables::mimi(&checkpoint, &mut tensors)?;
    tables::voices(checkpoint_dir, &checkpoint, &mut tensors)?;
    let mut writer = BundleWriter::create(out)?;
    for entry in tensors.into_entries() {
        if entry.values.len() != entry.shape.iter().product::<usize>() {
            let (name, shape) = (entry.name, entry.shape);
            return Err(Error::Format(format!("{name}: shape {shape:?}")));
        }
        let dtype = dtype(profile, &entry.name, &entry.shape);
        let tensor = Tensor::encode(dtype, entry.shape, &entry.values);
        writer.add_tensor(&entry.name, &tensor)?;
    }
    let path = checkpoint_dir.join(TOKENIZER_PART);
    let tokenizer = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
    writer.add_blob(TOKENIZER_PART, &tokenizer)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
