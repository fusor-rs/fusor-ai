use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// A safetensors file read into memory, every tensor's data range checked against its dtype and
/// shape.
pub struct Checkpoint {
    bytes: Vec<u8>,
    data_start: usize,
    entries: HashMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// The element types a safetensors header may name. Tensors of another type can be present but
/// not read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum Dtype {
    Bf16,
    F16,
    F32,
    F64,
    U8,
    I32,
    I64,
    #[serde(other)]
    Unsupported,
}

impl Dtype {
    fn size(self) -> Option<usize> {
        match self {
            Dtype::U8 => Some(1),
            Dtype::Bf16 | Dtype::F16 => Some(2),
            Dtype::F32 | Dtype::I32 => Some(4),
            Dtype::F64 | Dtype::I64 => Some(8),
            Dtype::Unsupported => None,
        }
    }
}

const HEADER_LEN_BYTES: usize = 8;
const METADATA: &str = "__metadata__";

impl Checkpoint {
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(Error::io(path))?;
        let in_file = |message: String| Error::Format(format!("{}: {message}", path.display()));
        let truncated = || in_file("not a safetensors file".into());
        let (length, rest) = bytes
            .split_first_chunk::<HEADER_LEN_BYTES>()
            .ok_or_else(truncated)?;
        let header_len = u64::from_le_bytes(*length) as usize;
        let header = rest.get(..header_len).ok_or_else(truncated)?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|error| in_file(error.to_string()))?;
        let data_len = rest.len() - header_len;
        let entries = header
            .into_iter()
            .filter(|(name, _)| name != METADATA)
            .map(|(name, value)| {
                let entry: Entry = serde_json::from_value(value)
                    .map_err(|error| in_file(format!("{name}: {error}")))?;
                entry
                    .check(data_len)
                    .map_err(|message| in_file(format!("{name}: {message}")))?;
                Ok((name, entry))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self {
            bytes,
            data_start: HEADER_LEN_BYTES + header_len,
            entries,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(&self.entry(name)?.shape)
    }

    fn entry(&self, name: &str) -> Result<&Entry, Error> {
        self.entries
            .get(name)
            .ok_or_else(|| Error::Format(format!("no tensor {name}")))
    }

    fn data(&self, entry: &Entry) -> &[u8] {
        let [start, end] = entry.data_offsets;
        &self.bytes[self.data_start + start..self.data_start + end]
    }

    /// The tensor as f32 (from bf16, f16 or f32).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let entry = self.entry(name)?;
        let data = self.data(entry);
        let halves = data.as_chunks::<2>().0.iter();
        match entry.dtype {
            Dtype::Bf16 => Ok(halves
                .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
                .collect()),
            Dtype::F16 => Ok(halves
                .map(|half| f16_to_f32(u16::from_le_bytes(*half)))
                .collect()),
            Dtype::F32 => Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()),
            other => Err(Error::Format(format!(
                "{name}: {other:?} is not a float type"
            ))),
        }
    }

    pub fn u8(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.scalars(name, Dtype::U8, u8::from_le_bytes)
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>, Error> {
        self.scalars(name, Dtype::I32, i32::from_le_bytes)
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>, Error> {
        self.scalars(name, Dtype::I64, i64::from_le_bytes)
    }

    pub fn f64(&self, name: &str) -> Result<Vec<f64>, Error> {
        self.scalars(name, Dtype::F64, f64::from_le_bytes)
    }

    fn scalars<const BYTES: usize, T>(
        &self,
        name: &str,
        dtype: Dtype,
        decode: fn([u8; BYTES]) -> T,
    ) -> Result<Vec<T>, Error> {
        let entry = self.entry(name)?;
        if entry.dtype != dtype {
            return Err(Error::Format(format!(
                "{name}: expected {dtype:?}, got {:?}",
                entry.dtype
            )));
        }
        let words = self.data(entry).as_chunks::<BYTES>().0;
        Ok(words.iter().map(|word| decode(*word)).collect())
    }
}

impl Entry {
    /// Its data range lies within the file's `data_len` data bytes and holds exactly its shape.
    fn check(&self, data_len: usize) -> Result<(), String> {
        let [start, end] = self.data_offsets;
        if start > end || end > data_len {
            return Err(format!(
                "data offsets [{start}, {end}] outside the {data_len} data bytes"
            ));
        }
        let Some(size) = self.dtype.size() else {
            return Ok(());
        };
        let expected = self.shape.iter().product::<usize>() * size;
        if end - start != expected {
            return Err(format!(
                "{} bytes for a {:?} tensor of shape {:?}",
                end - start,
                self.dtype,
                self.shape
            ));
        }
        Ok(())
    }
}

const F16_EXPONENT_MASK: u16 = 0x1f;
const F16_MANTISSA_BITS: u32 = 10;
const F32_MANTISSA_BITS: u32 = 23;
const MANTISSA_SHIFT: u32 = F32_MANTISSA_BITS - F16_MANTISSA_BITS;
/// The difference of the f32 and f16 exponent biases (127 − 15).
const EXPONENT_REBIAS: u32 = 112;
/// f16's smallest subnormal, 2⁻²⁴.
const F16_SUBNORMAL_STEP: f32 = 1.0 / 16_777_216.0;

/// IEEE half precision to single, exactly (every f16 value is an f32 value).
fn f16_to_f32(bits: u16) -> f32 {
    let negative = bits >> 15 == 1;
    let exponent = (bits >> F16_MANTISSA_BITS) & F16_EXPONENT_MASK;
    let mantissa = u32::from(bits) & ((1 << F16_MANTISSA_BITS) - 1);
    let magnitude = match exponent {
        0 => mantissa as f32 * F16_SUBNORMAL_STEP,
        F16_EXPONENT_MASK => f32::from_bits(f32::INFINITY.to_bits() | mantissa << MANTISSA_SHIFT),
        _ => {
            let biased = u32::from(exponent) + EXPONENT_REBIAS;
            f32::from_bits(biased << F32_MANTISSA_BITS | mantissa << MANTISSA_SHIFT)
        }
    };
    if negative { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::Checkpoint;

    /// A safetensors file of one tensor `x` with this header entry and data.
    fn file(name: &str, entry: &str, data: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "fusor-ai-checkpoint-{name}-{}.safetensors",
            std::process::id()
        ));
        let header = format!("{{\"x\": {entry}}}");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn read(path: &Path) -> Result<Checkpoint, crate::Error> {
        let checkpoint = Checkpoint::read(path);
        std::fs::remove_file(path).unwrap();
        checkpoint
    }

    #[test]
    fn half_precision_widens_exactly() {
        let cases: [(u16, f32); 7] = [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x8000, -0.0),
            (0xfc00, f32::NEG_INFINITY),
        ];
        let data: Vec<u8> = cases
            .iter()
            .map(|(bits, _)| bits)
            .chain([&0x7e00])
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let entry = r#"{"dtype": "F16", "shape": [8], "data_offsets": [0, 16]}"#;
        let values = read(&file("f16", entry, &data)).unwrap().f32("x").unwrap();
        for ((bits, want), got) in cases.iter().zip(&values) {
            assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x}");
        }
        assert!(values[7].is_nan());
    }

    #[test]
    fn rejects_data_that_does_not_hold_the_shape() {
        let cases = [
            (
                "short",
                r#"{"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}"#,
                "8 bytes for a F32 tensor of shape [3]",
            ),
            (
                "past",
                r#"{"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}"#,
                "data offsets [0, 16] outside the 8 data bytes",
            ),
        ];
        for (name, entry, message) in cases {
            let path = file(name, entry, &[0; 8]);
            let Err(error) = read(&path) else {
                panic!("{name}: read");
            };
            assert_eq!(
                error.to_string(),
                format!("{}: x: {message}", path.display())
            );
        }
    }
}
