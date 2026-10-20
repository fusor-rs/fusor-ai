//! Native model compilation: read a safetensors checkpoint, encode each tensor in the format the
//! runtime reads, and write the bundle's parts and manifest.
//!
//! What a model keeps, how its tensors are renamed or fused, and which dtype each gets is the
//! model's own compiler; this crate is the part every model shares.

mod checkpoint;
mod writer;

use std::path::PathBuf;

pub use checkpoint::{Checkpoint, Entry};
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
