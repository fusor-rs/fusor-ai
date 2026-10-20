use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// A safetensors file read into memory.
pub struct Checkpoint {
    bytes: Vec<u8>,
    data_start: usize,
    entries: HashMap<String, Entry>,
}

#[derive(Deserialize)]
pub struct Entry {
    pub dtype: String,
    pub shape: Vec<usize>,
    data_offsets: [usize; 2],
}

const HEADER_LEN_BYTES: usize = 8;

impl Checkpoint {
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(Error::io(path))?;
        let truncated = || Error::Format(format!("{}: not a safetensors file", path.display()));
        let (length, rest) = bytes
            .split_first_chunk::<HEADER_LEN_BYTES>()
            .ok_or_else(truncated)?;
        let header_len = u64::from_le_bytes(*length) as usize;
        let header = rest.get(..header_len).ok_or_else(truncated)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header)
            .map_err(|error| Error::Format(format!("{}: {error}", path.display())))?;
        let entries = header
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, value)| {
                let entry = serde_json::from_value(value)
                    .map_err(|error| Error::Format(format!("{name}: {error}")))?;
                Ok((name, entry))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self {
            bytes,
            data_start: HEADER_LEN_BYTES + header_len,
            entries,
        })
    }

    pub fn entry(&self, name: &str) -> Result<&Entry, Error> {
        self.entries
            .get(name)
            .ok_or_else(|| Error::Format(format!("no tensor {name}")))
    }

    fn data(&self, entry: &Entry) -> &[u8] {
        let [start, end] = entry.data_offsets;
        &self.bytes[self.data_start + start..self.data_start + end]
    }

    /// The tensor as f32 (from bf16 or f32).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let entry = self.entry(name)?;
        let data = self.data(entry);
        match entry.dtype.as_str() {
            "BF16" => Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
                .collect()),
            "F32" => Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()),
            other => Err(Error::Format(format!("{name}: unsupported dtype {other}"))),
        }
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>, Error> {
        let entry = self.entry(name)?;
        if entry.dtype != "I32" {
            return Err(Error::Format(format!(
                "{name}: expected I32, got {}",
                entry.dtype
            )));
        }
        let words = self.data(entry).as_chunks::<4>().0;
        Ok(words.iter().map(|word| i32::from_le_bytes(*word)).collect())
    }
}
