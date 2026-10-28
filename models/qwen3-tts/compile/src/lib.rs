//! `qwen3-tts compile`: the checkpoint folder (`model.safetensors`, `speech_tokenizer/
//! model.safetensors`, `tokenizer.json`, `tokenizer_config.json`) → a browser bundle.
//!
//! Renames to the runtime's vocabulary (`qwen3_tts::model`), fuses each attention layer's query,
//! key and value projections, and folds what does not depend on the request: the text embedding
//! through the text projection (`text`), the speech tokenizer's codebooks (centroids from their
//! running sums; the decoder's through the quantizer's output projections), layer scales into
//! the output rows of both residual branches, the ConvNeXt blocks' `gamma`, and every conv into
//! a causal-conv matrix (`blocks::seanet`, `blocks::ecapa`, `blocks::code2wav`). The encoder's
//! rotary pairs are permuted from halves to neighbours. Weights are then quantized per profile
//! and packed by `BundleWriter`.

mod codec;
mod speaker;
mod talker;
mod text;
mod tokenizer;

use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Tensors};
use kernels::DType;

const REPO: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
const REVISION: &str = "5d83992436eae1d760afd27aff78a71d676296fc";
const WEIGHTS: &str = "model.safetensors";
const CODEC_WEIGHTS: &str = "speech_tokenizer/model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The large matrices 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    /// The encoder's nearest-code search reads exact centroids.
    const ALWAYS_F32: &'static [&'static str] = &["codec.encoder.codebooks"];

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

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    talker::write(&checkpoint, &mut tensors)?;
    text::write(&checkpoint, &mut tensors)?;
    speaker::write(&checkpoint, &mut tensors)?;
    drop(checkpoint);
    let codec = Checkpoint::read(&checkpoint_dir.join(CODEC_WEIGHTS))?;
    codec::write(&codec, &mut tensors)?;
    tokenizer::write(checkpoint_dir, &mut writer)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
