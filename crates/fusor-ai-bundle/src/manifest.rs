use std::collections::BTreeMap;

use fusor_ai_kernels::DType;
use serde::{Deserialize, Serialize};

use crate::Error;

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub model: Source,
    pub parts: Vec<Part>,
    pub tensors: BTreeMap<String, TensorEntry>,
}

/// Where the weights came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub repo: String,
    pub revision: String,
    /// The quantization profile the compiler applied.
    pub profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    /// Path relative to the manifest.
    pub file: String,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Index into `parts`.
    pub part: usize,
    /// Byte offset inside the part.
    pub offset: u64,
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, Error> {
        let manifest: Manifest = serde_json::from_str(json)
            .map_err(|error| Error::Format(format!("manifest: {error}")))?;
        if manifest.format != FORMAT {
            return Err(Error::Format(format!(
                "bundle format {} (this runtime reads {FORMAT})",
                manifest.format
            )));
        }
        Ok(manifest)
    }

    pub fn part_index(&self, file: &str) -> Result<usize, Error> {
        self.parts
            .iter()
            .position(|part| part.file == file)
            .ok_or_else(|| Error::Format(format!("no part {file}")))
    }

    /// Whether no tensor lives in part `part` (it is a blob).
    pub fn is_blob(&self, part: usize) -> bool {
        self.tensors.values().all(|entry| entry.part != part)
    }

    /// Bytes of the parts that hold tensors.
    pub fn tensor_bytes(&self) -> u64 {
        let parts = self.parts.iter().enumerate();
        parts
            .filter(|(index, _)| !self.is_blob(*index))
            .map(|(_, part)| part.bytes)
            .sum()
    }
}
