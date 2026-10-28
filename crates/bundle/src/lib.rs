//! Model bundles: `manifest.json` plus parts, each with its size and BLAKE3.
//!
//! A part holds tensors (cut at tensor boundaries) or is a blob (any other file the runtime
//! needs, such as a tokenizer). `Store` receives a part's bytes in order — a browser loader
//! pushes them as they arrive, a native host reads files — and copies each byte range straight
//! into the tensor it belongs to, so a weight is copied once and no part is held whole.
//! `end_part` checks the size and the hash. Once every part is in, each part of a model loads
//! itself from the store (`Load`), checking the shape of every tensor it takes.

mod engine;
mod load;
mod manifest;
pub mod python;
mod stages;
mod store;
#[cfg(feature = "tokenizer")]
mod tokenizer;

pub use engine::Engine;
pub use load::{Load, Scope, load_each};
pub use manifest::{FORMAT, Manifest, Part, Source, TensorEntry};
pub use stages::Stages;
pub use store::Store;
#[cfg(feature = "tokenizer")]
pub use tokenizer::{Encoded, TOKENIZER_PART, Tokenizer};

/// What a bundle or a model runtime reports: its `kind()` names the variant for hosts.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The manifest, a part or a tensor does not have the expected structure.
    #[error("format: {0}")]
    Format(String),
    /// A part's bytes do not hash to the manifest's BLAKE3.
    #[error("integrity: {0}")]
    Integrity(String),
    /// A request the model cannot serve.
    #[error("request: {0}")]
    Request(String),
}

impl Error {
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Format(_) => "format",
            Error::Integrity(_) => "integrity",
            Error::Request(_) => "request",
        }
    }
}

impl From<kernels::Error> for Error {
    fn from(error: kernels::Error) -> Self {
        Error::Format(error.to_string())
    }
}
