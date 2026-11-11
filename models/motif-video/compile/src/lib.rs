//! `motif-video compile`: the checkpoint folder (Motif-Video-2B's repository layout:
//! `text_encoder/` and `transformer/` and `vae/` safetensors, and the tokenizer) → two browser
//! bundles, `text` and `video`, which a page runs on separate memories.
//!
//! What does not depend on the prompt is folded here: each attention's queries, keys and values
//! into one projection (the cross-attentions' keys and values of the context into one, each dual
//! block's two modulations into one), Gemma's zero-centred norms as `1 + w`, SigLIP's MLP padded
//! to whole quantization blocks, the patch convolutions as matrices, the head's rows into the
//! patch layout, the latents' denormalization into the decoder's `post_quant`, and the
//! autoencoder's first-frame encoder cut to what one frame reads: each causal kernel to its last
//! frame tap, each stride-2 resample written into a 5 × 5 window.

mod dit;
mod text;
mod vae;
mod vision;

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors, read_json};
use kernels::DType;
use motif_video::model::TEXT;
use serde_json::Value;

const REPO: &str = "Motif-Technologies/Motif-Video-2B";
const REVISION: &str = "6748a1f5861a859aca13b30a0c65dd6f9942ca35";
const TEXT_WEIGHTS: &str = "text_encoder/model.safetensors";
const TOKENIZER: &str = "tokenizer/tokenizer.json";
const DIT_WEIGHTS: &str = "transformer/diffusion_pytorch_model.safetensors";
const VAE_WEIGHTS: &str = "vae/diffusion_pytorch_model.safetensors";
/// The tokenizer file's settings that pad every encoding to 512 and cut it: the model never pads
/// (it encodes only the valid ids) and `prompt::tokenize` cuts.
const PADDING_SETTINGS: [&str; 2] = ["truncation", "padding"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 but T5Gemma2's text matrices, 7-bit: an f32 text model (15.5 GB) does not
    /// fit beside the rest in the parity machine's memory (native parity only).
    F32,
    /// T5Gemma2's text matrices 4-bit (its bundle must fit a wasm32 memory with SigLIP), every
    /// other large matrix 7-bit.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Mix];
    /// The end-of-image embedding, SigLIP's patch and position tables, the transformer's patch,
    /// time and head layers, and the autoencoder's layers of a few channels.
    const ALWAYS_F32: &'static [&'static str] = &[
        "text.end_of_image",
        "vision.patch.",
        "vision.positions",
        "video.dit.patch.",
        "video.dit.time.",
        "video.dit.head.linear.",
        "video.vae.post_quant.",
        "video.vae.head.conv.",
        "video.vae.encoder.conv_in.",
        "video.vae.encoder.quant.",
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

/// The text bundle (`out`): T5Gemma2's text encoder, SigLIP and the tokenizer.
pub fn text(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(TEXT_WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    text::write(&checkpoint, &mut tensors)?;
    vision::write(&checkpoint, &mut tensors)?;
    let mut tokenizer: Value = read_json(&checkpoint_dir.join(TOKENIZER))?;
    for setting in PADDING_SETTINGS {
        tokenizer[setting] = Value::Null;
    }
    let bytes = serde_json::to_vec(&tokenizer)
        .map_err(|error| Error::Format(format!("{TOKENIZER_PART}: {error}")))?;
    writer.add_blob(TOKENIZER_PART, &bytes)?;
    finish(writer, profile)
}

/// The video bundle (`out`): the transformer and the autoencoder's encoder and decoder.
pub fn video(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let transformer = Checkpoint::read(&checkpoint_dir.join(DIT_WEIGHTS))?;
    let autoencoder = Checkpoint::read(&checkpoint_dir.join(VAE_WEIGHTS))?;
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
