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

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;
use phonon::model::SEANET_SHAPE;

const REPO: &str = "gradium/phonon-31m-en";
const REVISION: &str = "52433c46975bfa13144c56204c9f4ee1c5b148aa";
/// The `gradium/phonon-voices` revision whose voices the bundle ships.
const VOICES_REVISION: &str = "0ee17205f254b5abc43b266c8a0f215565c352f7";
const WEIGHTS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// 7-bit weights for every large matrix.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    const ALWAYS_F32: &'static [&'static str] = &[
        "lm.input.",
        "lm.eos.",
        "flow.input.",
        "flow.final.",
        "mimi.latent.",
        "mimi.upsample",
        "mimi.decoder.final.",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        match self {
            Self::Q7 if !last_residual(name) => DType::Q7b32,
            Self::F32 | Self::Q7 => DType::F32,
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        compile::profile(name)
    }
}

/// The 1-wide conv that ends SEANet's last residual unit: its int8 activations (one block of
/// channels per sample) set the noise floor of quiet passages, about −70 dBFS against −82 in f32.
fn last_residual(name: &str) -> bool {
    let last_stage = SEANET_SHAPE.ratios.len() - 1;
    name.starts_with(&format!("mimi.decoder.stages.{last_stage}.residual2."))
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    tables::flow_lm(&checkpoint, &mut tensors)?;
    tables::flow_head(&checkpoint, &mut tensors)?;
    tables::mimi(&checkpoint, &mut tensors)?;
    tables::voices(checkpoint_dir, &checkpoint, &mut tensors)?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
