//! Model bundles: `manifest.json` plus parts, each with its size and BLAKE3.
//!
//! A part holds tensors (cut at tensor boundaries, each tensor 16-byte aligned) or is a blob
//! (any other file the runtime needs, such as a tokenizer). `Store` receives a part's bytes in
//! order — a browser loader pushes them as they arrive, a native host reads files — and copies
//! each byte range straight into the tensor it belongs to, so a weight is copied once and no
//! part is held whole. `end_part` checks the size and the hash.

mod manifest;
mod store;

pub use manifest::{FORMAT, Manifest, Part, Source, TensorEntry};
pub use store::Store;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The manifest or a part does not have the expected structure.
    #[error("{0}")]
    Format(String),
    /// A part's bytes do not hash to the manifest's BLAKE3.
    #[error("{0}")]
    Integrity(String),
}

impl From<kernels::Error> for Error {
    fn from(error: kernels::Error) -> Self {
        Error::Format(error.to_string())
    }
}
