//! `pjit-21k compile`: Qwen3.5-4B's and P-JiT 21K's checkpoints → two browser bundles, `text` and
//! `image`.
//!
//! The text bundle: Qwen3.5-4B's token embeddings and decoder layers 0 to 27 (its later layers,
//! final norm and the multi-token prediction layer are left out: the states are read before
//! them), renamed to `blocks::hybrid`'s vocabulary, the zero-centred RMSNorms' `1 + w` folded,
//! each attention layer's interleaved query and gate heads split apart, each delta layer's input
//! projections fused; the tokenizer beside them. The image bundle: P-JiT 21K's DiT, renamed to
//! `blocks::pyramid_jit`'s vocabulary, each block's queries, keys, values and gate logits fused,
//! the patch convolution laid out tap-major. Both quantized per profile; fusor-ai's
//! `BundleWriter` packs and hashes the parts and writes `manifest.json`.

mod image;
mod text;

use std::path::Path;

use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;
use pjit_21k::model::TEXT;

/// Where the fetch script puts each checkpoint under the checkpoint directory.
const TEXT_DIR: &str = "qwen3.5-4b";
const IMAGE_DIR: &str = "pjit-21k";
const TEXT_REPO: &str = "Qwen/Qwen3.5-4B";
const TEXT_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
const IMAGE_REPO: &str = "Linum-AI/pjit-21k";
const IMAGE_REVISION: &str = "251725ad9a0618f0a4ba3cdf4a488d35312be352";
const IMAGE_WEIGHTS: &str = "model.safetensors";
/// The text tensors kept 4-bit in every profile (name suffixes): the embeddings and the
/// feed-forwards' gate and up projections.
const TEXT_FOUR_BIT: [&str; 3] = ["embedding", ".gate.weight", ".up.weight"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// The DiT's every weight f32 (native parity). The text encoder as `Mix` has it: its f32
    /// weights (15.6 GB) would not fit beside the rest, and its parity line then measures the
    /// encoder the page runs.
    F32,
    /// The text encoder's embeddings and feed-forward gate and up projections 4-bit and its other
    /// matrices 7-bit, so that its bundle (3.27 GB) and activations fit a wasm32 memory with
    /// room; the DiT's large matrices 7-bit, its patch embedding, timestep MLP and roots, text
    /// projection and head f32.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Mix];
    const ALWAYS_F32: &'static [&'static str] = &[
        "image.patch.",
        "image.time.",
        "image.roots.",
        "image.text.",
        "image.head.",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Mix => "mix",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        if name.starts_with(TEXT) {
            let four_bit = TEXT_FOUR_BIT.iter().any(|suffix| name.ends_with(suffix));
            return if four_bit { DType::Q4b32 } else { DType::Q7b32 };
        }
        match self {
            Self::F32 => DType::F32,
            Self::Mix => DType::Q7b32,
        }
    }
}

/// The text bundle (`out`): Qwen3.5-4B's embeddings and first layers, and its tokenizer.
pub fn text(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let directory = checkpoint_dir.join(TEXT_DIR);
    let mut writer = BundleWriter::create(out)?;
    text::write(&directory, &mut Tensors::new(&mut writer, profile))?;
    writer.add_blob_file(TOKENIZER_PART, &directory.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: TEXT_REPO.into(),
        revision: TEXT_REVISION.into(),
        profile: profile.name().into(),
    })
}

/// The image bundle (`out`): P-JiT 21K's DiT.
pub fn image(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let weights = checkpoint_dir.join(IMAGE_DIR).join(IMAGE_WEIGHTS);
    let checkpoint = Checkpoint::read(&weights)?;
    let mut writer = BundleWriter::create(out)?;
    image::write(&checkpoint, &mut Tensors::new(&mut writer, profile))?;
    writer.finish(Source {
        repo: IMAGE_REPO.into(),
        revision: IMAGE_REVISION.into(),
        profile: profile.name().into(),
    })
}
