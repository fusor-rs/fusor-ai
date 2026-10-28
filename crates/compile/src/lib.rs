//! Native model compilation: read a checkpoint (safetensors, ONNX, NumPy `.npz`), encode each
//! tensor in the format the runtime reads, and write the bundle's parts and manifest.
//!
//! What a model keeps, how its tensors are renamed or fused, and which format each gets (its
//! `Profile`) is the model's own compiler; this crate is the part every model shares.

mod checkpoint;
mod conv;
mod npz;
mod onnx;
mod profile;
mod table;
mod tensors;
mod writer;

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

pub use checkpoint::Checkpoint;
pub use npz::{Array, Values, read_npz};
pub use onnx::{Node, Onnx, OnnxTensor};
pub use profile::{Profile, profile};
pub use table::{Layout, Table};
pub use tensors::Tensors;
pub use writer::BundleWriter;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{0}")]
    Format(String),
}

impl Error {
    fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| Error::Io { path, source }
    }
}

/// The JSON file at `path` (a tokenizer, an export's manifest).
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let json = std::fs::read(path).map_err(Error::io(path))?;
    serde_json::from_slice(&json)
        .map_err(|error| Error::Format(format!("{}: {error}", path.display())))
}

/// Multiplies row `i` of a row-major matrix by `scales[i]`.
pub fn scale_rows(matrix: &mut [f32], scales: &[f32]) {
    let cols = matrix.len() / scales.len();
    for (row, scale) in matrix.chunks_exact_mut(cols).zip(scales) {
        row.iter_mut().for_each(|value| *value *= scale);
    }
}
