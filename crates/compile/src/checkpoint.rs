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

    /// The tensor as f32 (from bf16, f16 or f32).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let entry = self.entry(name)?;
        let data = self.data(entry);
        let halves = data.as_chunks::<2>().0.iter();
        match entry.dtype.as_str() {
            "BF16" => Ok(halves
                .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
                .collect()),
            "F16" => Ok(halves
                .map(|half| f16_to_f32(u16::from_le_bytes(*half)))
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

    pub fn u8(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.integers(name, "U8", u8::from_le_bytes)
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>, Error> {
        self.integers(name, "I32", i32::from_le_bytes)
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>, Error> {
        self.integers(name, "I64", i64::from_le_bytes)
    }

    fn integers<const BYTES: usize, T>(
        &self,
        name: &str,
        dtype: &str,
        decode: fn([u8; BYTES]) -> T,
    ) -> Result<Vec<T>, Error> {
        let entry = self.entry(name)?;
        if entry.dtype != dtype {
            return Err(Error::Format(format!(
                "{name}: expected {dtype}, got {}",
                entry.dtype
            )));
        }
        let words = self.data(entry).as_chunks::<BYTES>().0;
        Ok(words.iter().map(|word| decode(*word)).collect())
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
    use super::f16_to_f32;

    #[test]
    fn half_precision_widens_exactly() {
        let cases = [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x8000, -0.0),
            (0xfc00, f32::NEG_INFINITY),
        ];
        for (bits, want) in cases {
            assert_eq!(
                f16_to_f32(bits).to_bits(),
                f32::to_bits(want),
                "{bits:#06x}"
            );
        }
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
