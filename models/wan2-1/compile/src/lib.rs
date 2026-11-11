//! `wan2-1 compile`: the checkpoint folder (Wan2.1-T2V-1.3B's repository layout: the diffusion
//! transformer's `diffusion_pytorch_model.safetensors`, umT5-XXL's and the autoencoder's PyTorch
//! archives, and the umT5 tokenizer under `google/umt5-xxl/`) → two browser bundles, `text` and
//! `video`, which a page runs on separate memories.
//!
//! What does not depend on the prompt is folded here: each attention's queries, keys and values
//! into one projection (the cross-attentions' keys and values of the context into one),
//! convolutions laid out as their matrices (`kernels::conv`, tap-major over the kernel's
//! dimensions), pointwise convolutions as linears, the latents' denormalization into the
//! decoder's `post_quant`, and umT5's per-layer relative position biases as vectors. The
//! autoencoder's encoder is not read.

mod dit;
mod text;
mod vae;

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;
use wan2_1::model::TEXT;

const REPO: &str = "Wan-AI/Wan2.1-T2V-1.3B";
const REVISION: &str = "37ec512624d61f7aa208f7ea8140a131f93afc9a";
const TEXT_WEIGHTS: &str = "models_t5_umt5-xxl-enc-bf16.pth";
const TOKENIZER: &str = "google/umt5-xxl/tokenizer.json";
const DIT_WEIGHTS: &str = "diffusion_pytorch_model.safetensors";
const VAE_WEIGHTS: &str = "Wan2.1_VAE.pth";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 but the text encoder's matrices, 7-bit: an f32 umT5-XXL (22.7 GB) does
    /// not fit beside the rest in the parity machine's memory (native parity only).
    F32,
    /// The text encoder's matrices 4-bit (its bundle must fit a wasm32 memory), every other large
    /// matrix 7-bit.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Mix];
    /// The transformer's patch, time and head layers and the decoder's `post_quant` and output
    /// convolution (a few channels each, read straight into pixels).
    const ALWAYS_F32: &'static [&'static str] = &[
        "video.dit.patch",
        "video.dit.time",
        "video.dit.modulation",
        "video.dit.head",
        "video.vae.post_quant",
        "video.vae.head",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Mix => "mix",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        match (self, name.starts_with(TEXT)) {
            (Self::F32, true) | (Self::Mix, false) => DType::Q7b32,
            (Self::Mix, true) => DType::Q4b32,
            (Self::F32, false) => DType::F32,
        }
    }
}

/// The text encoder's bundle (`out`): umT5-XXL and its tokenizer.
pub fn text(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read_pth(&checkpoint_dir.join(TEXT_WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    text::write(&checkpoint, &mut Tensors::new(&mut writer, profile))?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER))?;
    finish(writer, profile)
}

/// The video bundle (`out`): the diffusion transformer and the autoencoder's decoder.
pub fn video(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let transformer = Checkpoint::read(&checkpoint_dir.join(DIT_WEIGHTS))?;
    let autoencoder = Checkpoint::read_pth(&checkpoint_dir.join(VAE_WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    dit::write(&transformer, &mut tensors)?;
    vae::write(&autoencoder, &mut tensors)?;
    finish(writer, profile)
}

fn finish(writer: BundleWriter, profile: Profile) -> Result<Manifest, Error> {
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
