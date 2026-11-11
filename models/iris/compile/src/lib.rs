//! `iris compile`: Iris-3B's checkpoint (the base model in the repository's root, and `turbo/`,
//! `depth/` and `upscaler/`, each a `model.safetensors`) and Qwen3-VL-4B's language model →
//! browser bundles: `text`, and one per transformer, which a page runs on separate memories.
//!
//! What does not depend on the prompt is folded here: each attention's queries, keys and values
//! into one projection, the SwiGLUs padded to whole quantization blocks, the patch embedding's
//! columns in the order the runtime gathers a patch (pixel-major), the depth model's zero input
//! channel dropped and its 1×1 reducer folded into its last layer, and the empty prompt's real
//! rows (depth and the restorer). Qwen3-VL's last layer and final norm feed nothing and are left
//! out.

mod dit;
mod text;

use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Error, Profile as _, Tensors};
use iris::model::{TEXT, VARIANT_PART, Variant};
use kernels::DType;

const REPO: &str = "speridlabs/iris-3b";
const REVISION: &str = "adc05b909fb05d4276d26aae5a52b36406b2ef26";
const TEXT_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const TEXT_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// The transformer's weights f32 and Qwen3-VL's matrices 7-bit: an f32 Qwen3-VL (14.5 GB)
    /// does not fit beside the parity run (native parity only).
    F32,
    /// The feed-forward matrices of Qwen3-VL and of the transformer's blocks 4-bit, every other
    /// large matrix 7-bit: Qwen3-VL at 7-bit is 4.4 GB, more than a wasm32 memory holds.
    Mix,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Mix];
    /// The patch, time and pixel layers, the learned positions, and the pixel head's MLPs.
    const ALWAYS_F32: &'static [&'static str] = &[
        "dit.patch.",
        "dit.time.",
        "dit.positions",
        "dit.pixel.embed.",
        "dit.pixel.final",
        "dit.pixel.blocks.0.ff",
        "dit.pixel.blocks.1.ff",
        "dit.pixel.blocks.2.ff",
        "dit.pixel.blocks.3.ff",
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Mix => "mix",
        }
    }

    fn matrix_dtype(self, name: &str) -> DType {
        let feed_forward = if name.starts_with(TEXT) {
            [".gate.weight", ".up.weight", ".down.weight"]
                .iter()
                .any(|part| name.ends_with(part))
        } else {
            name.starts_with("dit.blocks.") && name.contains(".ff_")
        };
        match (self, feed_forward, name.starts_with(TEXT)) {
            (Self::Mix, true, _) => DType::Q4b32,
            (Self::F32, _, false) => DType::F32,
            _ => DType::Q7b32,
        }
    }
}

/// The text bundle (`out`): Qwen3-VL's first layers, its embeddings and tokenizer, from
/// `checkpoint_dir` (its repository's files).
pub fn text(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let mut writer = BundleWriter::create(out)?;
    text::write(checkpoint_dir, &mut writer, profile)?;
    let source = Source {
        repo: TEXT_REPO.into(),
        revision: TEXT_REVISION.into(),
        profile: profile.name().into(),
    };
    writer.finish(source)
}

/// The bundle of one transformer (`out`), from `checkpoint_dir`: its folder of the repository.
pub fn transformer(
    checkpoint_dir: &Path,
    out: &Path,
    (variant, profile): (Variant, Profile),
) -> Result<Manifest, Error> {
    let mut writer = BundleWriter::create(out)?;
    dit::write(
        checkpoint_dir,
        &mut Tensors::new(&mut writer, profile),
        variant,
    )?;
    let blob = serde_json::to_vec(&variant)
        .map_err(|error| Error::Format(format!("{VARIANT_PART}: {error}")))?;
    writer.add_blob(VARIANT_PART, &blob)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}

/// The folder of `variant`'s files in Iris-3B's repository: the root, `turbo`, `depth` or
/// `upscaler`.
pub fn folder(variant: Variant) -> &'static str {
    match variant {
        Variant::Base => "",
        other => other.name(),
    }
}
