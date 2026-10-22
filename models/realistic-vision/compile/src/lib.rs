//! `realistic-vision compile`: the checkpoint folder (Realistic Vision's repository layout: its
//! single-file checkpoint and `tokenizer/`, with the SD VAE ft-MSE's checkpoint beside them, as
//! the app's `fetch-checkpoint.sh` downloads them) → a browser bundle.
//!
//! The text encoder and the UNet come from Realistic Vision's single-file (LDM) checkpoint, the
//! decoder from the ft-MSE autoencoder's; the checkpoint's own autoencoder is not read. What
//! does not depend on the prompt is folded here: each attention's queries, keys and values into
//! one projection (the cross-attentions' keys and values of the context into one), the GEGLU's
//! projection split into its hidden and gate halves, every convolution's weights laid out as
//! its matrix (`kernels::conv`), pointwise convolutions as linears, and the latents' scaling into
//! the decoder's `post_quant`. CLIP's `vocab.json` and `merges.txt` become the bundle's
//! `tokenizer.json`.

mod tables;
mod tokenizer;

use std::path::Path;
use std::str::FromStr;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;

const REPOS: &str = "SG161222/Realistic_Vision_V6.0_B1_noVAE, stabilityai/sd-vae-ft-mse-original";
const REVISIONS: &str =
    "9a857a696b9aabbf509073e0aa55ec8200b6ef7d, 629b3ad3030ce36e15e70c5db7d91df0d60c627f";
const CHECKPOINT: &str = "Realistic_Vision_V6.0_NV_B1_fp16.safetensors";
const AUTOENCODER: &str = "vae-ft-mse-840000-ema-pruned.safetensors";
const VOCABULARY: &str = "tokenizer/vocab.json";
const MERGES: &str = "tokenizer/merges.txt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// Every large matrix 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    /// The output convolutions (a few channels each, read straight into pixels), the decoder's
    /// `post_quant` and the position embeddings.
    const ALWAYS_F32: &'static [&'static str] = &[
        "unet.conv_out",
        "vae.conv_out",
        "vae.post_quant",
        "text.positions",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
        }
    }

    fn matrix_dtype(self, _name: &str) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Q7 => DType::Q7b32,
        }
    }
}

impl FromStr for Profile {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        compile::profile(name)
    }
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(CHECKPOINT))?;
    let autoencoder = Checkpoint::read(&checkpoint_dir.join(AUTOENCODER))?;
    let tokenizer = tokenizer::json(
        &checkpoint_dir.join(VOCABULARY),
        &checkpoint_dir.join(MERGES),
    )?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    tables::text(&checkpoint, &mut tensors)?;
    tables::unet(&checkpoint, &mut tensors)?;
    tables::vae(&autoencoder, &mut tensors)?;
    writer.add_blob(TOKENIZER_PART, &tokenizer)?;
    writer.finish(Source {
        repo: REPOS.into(),
        revision: REVISIONS.into(),
        profile: profile.name().into(),
    })
}
