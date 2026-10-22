//! `__name__ compile`: the checkpoint (`model.safetensors`) → a browser bundle: the tensors the
//! model reads, renamed to its vocabulary (`compile::Table`) and quantized per profile;
//! fusor-ai's `BundleWriter` packs and hashes the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error, Profile as _, Table, Tensors};
use kernels::DType;

const REPO: &str = "__repo__";
const REVISION: &str = "__revision__";
const WEIGHTS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The large matrices 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    const ALWAYS_F32: &'static [&'static str] = &[];

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

/// The model's tensors, as the runtime names them, from the checkpoint's.
fn rename_table() -> Table {
    Table::default()
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    rename_table().write(&checkpoint, &mut tensors)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
