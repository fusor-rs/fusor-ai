//! Weight tensors and their storage formats.
//!
//! A quantized matrix `[rows, cols]` is stored as one byte buffer: the quants of every row
//! (row-major, `cols` bytes per row for `Q7b32`, `cols / 2` for `Q4b32`) followed by one f32
//! scale per block of 32 along a row (`rows · cols / 32` scales, row-major). Both planes start
//! 16-byte aligned because `cols` is a multiple of 32.
//!
//! - `Q7b32`: `u = q + 64` with `q ∈ [−63, 63]`, so every stored byte is a valid unsigned 7-bit
//!   operand of `i32x4.relaxed_dot_i8x16_i7x16_add_s`; value = `d · (u − 64)`.
//! - `Q4b32`: ggml Q4_0 nibbles, `u = q + 8` with `q ∈ [−8, 7]`; byte `j` of a block holds element
//!   `j` in its low nibble and element `j + 16` in its high nibble; value = `d · (u − 8)`.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::blob::Blob;

pub const BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    Q7b32,
    Q4b32,
}

impl DType {
    /// Bytes of a tensor of this type with the given shape (`[rows, cols]` for quantized
    /// matrices).
    pub fn byte_len(self, shape: &[usize]) -> usize {
        let count: usize = shape.iter().product();
        match self {
            DType::F32 => count * 4,
            DType::Q7b32 | DType::Q4b32 => self.quant_bytes(count) + count / BLOCK * 4,
        }
    }

    /// Bytes of the quants of `count` values (0 for f32).
    pub(crate) fn quant_bytes(self, count: usize) -> usize {
        match self {
            DType::F32 => 0,
            DType::Q7b32 => count,
            DType::Q4b32 => count / 2,
        }
    }

    /// What a quantized weight byte must have subtracted to give the signed value.
    pub(crate) fn zero_point(self) -> i32 {
        match self {
            DType::F32 => 0,
            DType::Q7b32 => 64,
            DType::Q4b32 => 8,
        }
    }
}

pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    blob: Blob,
}

/// A borrowed `[rows, cols]` weight matrix (`rows` outputs, `cols` inputs).
#[derive(Clone, Copy)]
pub struct Matrix<'tensor> {
    pub rows: usize,
    pub cols: usize,
    pub storage: MatrixStorage<'tensor>,
}

#[derive(Clone, Copy)]
pub enum MatrixStorage<'tensor> {
    F32(&'tensor [f32]),
    Quant {
        dtype: DType,
        quants: &'tensor [u8],
        scales: &'tensor [f32],
    },
}

impl Tensor {
    /// An all-zero tensor to be filled through `bytes_mut` (a bundle loader).
    pub fn zeroed(dtype: DType, shape: Vec<usize>) -> Self {
        let blob = Blob::zeroed(dtype.byte_len(&shape));
        Self { dtype, shape, blob }
    }

    /// Encodes row-major `values` of `shape` as `dtype`. Quantized types need a matrix whose
    /// rows are a multiple of 32 long; quantization rounds ties to even.
    pub fn encode(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Self {
        let mut tensor = Self::zeroed(dtype, shape);
        let quant_len = dtype.quant_bytes(values.len());
        let (quants, scales) = tensor.blob.bytes_mut().split_at_mut(quant_len);
        let encoder = match dtype {
            DType::F32 => {
                let words = scales.as_chunks_mut::<4>().0;
                for (word, value) in words.iter_mut().zip(values) {
                    word.copy_from_slice(&value.to_le_bytes());
                }
                return tensor;
            }
            DType::Q7b32 => quantize_q7,
            DType::Q4b32 => quantize_q4,
        };
        let blocks = values.as_chunks::<BLOCK>().0;
        let outputs = quants.chunks_exact_mut(dtype.quant_bytes(BLOCK));
        let scale_slots = scales.as_chunks_mut::<4>().0;
        for ((block, output), slot) in blocks.iter().zip(outputs).zip(scale_slots) {
            slot.copy_from_slice(&encoder(block, output).to_le_bytes());
        }
        tensor
    }

    pub fn bytes(&self) -> &[u8] {
        self.blob.bytes()
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.blob.bytes_mut()
    }

    pub fn values(&self) -> Result<&[f32], Error> {
        match self.dtype {
            DType::F32 => Ok(self.blob.f32s(0)),
            dtype => Err(Error::Format(format!(
                "expected an f32 tensor, got {dtype:?}"
            ))),
        }
    }

    pub fn matrix(&self) -> Result<Matrix<'_>, Error> {
        let [rows, cols] = self.shape[..] else {
            return Err(Error::Format(format!(
                "expected a matrix, got shape {:?}",
                self.shape
            )));
        };
        let storage = match self.dtype {
            DType::F32 => MatrixStorage::F32(self.blob.f32s(0)),
            dtype => {
                let quant_len = dtype.quant_bytes(rows * cols);
                MatrixStorage::Quant {
                    dtype,
                    quants: &self.blob.bytes()[..quant_len],
                    scales: self.blob.f32s(quant_len),
                }
            }
        };
        Ok(Matrix {
            rows,
            cols,
            storage,
        })
    }
}

impl Matrix<'_> {
    /// Row `row` as f32 (an embedding gather); `out.len() == cols`.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) {
        let cols = self.cols;
        match self.storage {
            MatrixStorage::F32(weights) => {
                out.copy_from_slice(&weights[row * cols..(row + 1) * cols])
            }
            MatrixStorage::Quant {
                dtype,
                quants,
                scales,
            } => {
                let blocks = cols / BLOCK;
                let block_bytes = dtype.quant_bytes(BLOCK);
                let first = row * blocks;
                let row_quants = quants[first * block_bytes..].chunks_exact(block_bytes);
                let outputs = out.as_chunks_mut::<BLOCK>().0;
                for ((stored, output), &scale) in row_quants.zip(outputs).zip(&scales[first..]) {
                    dequantize_block(dtype, stored, scale, output);
                }
            }
        }
    }
}

fn dequantize_block(dtype: DType, stored: &[u8], scale: f32, out: &mut [f32]) {
    let level = |stored: u8| scale * (i32::from(stored) - dtype.zero_point()) as f32;
    match dtype {
        DType::Q7b32 => {
            for (output, &byte) in out.iter_mut().zip(stored) {
                *output = level(byte);
            }
        }
        DType::Q4b32 => {
            for (j, &byte) in stored.iter().enumerate() {
                out[j] = level(byte & 0x0f);
                out[j + BLOCK / 2] = level(byte >> 4);
            }
        }
        DType::F32 => unreachable!("F32 has no blocks"),
    }
}

const Q7_MAX: f32 = 63.0;

/// Symmetric 7-bit: `d = absmax / 63`, `q = round_ties_even(x / d)`. Returns `d`.
fn quantize_q7(block: &[f32], out: &mut [u8]) -> f32 {
    let absmax = block.iter().fold(0.0f32, |max, x| max.max(x.abs()));
    let scale = if absmax == 0.0 { 1.0 } else { absmax / Q7_MAX };
    for (stored, x) in out.iter_mut().zip(block) {
        let level = (x / scale).round_ties_even().clamp(-Q7_MAX, Q7_MAX) as i32;
        *stored = (level + 64) as u8;
    }
    scale
}

/// ggml Q4_0: `d = signed extreme / −8`, `u = clamp(round(x / d) + 8, 0, 15)`. Returns `d`.
fn quantize_q4(block: &[f32], out: &mut [u8]) -> f32 {
    let extreme = block.iter().fold(
        0.0f32,
        |extreme, &x| {
            if x.abs() > extreme.abs() { x } else { extreme }
        },
    );
    let scale = if extreme == 0.0 { 1.0 } else { extreme / -8.0 };
    let nibble = |x: f32| ((x / scale).round_ties_even() + 8.0).clamp(0.0, 15.0) as u8;
    for j in 0..BLOCK / 2 {
        out[j] = nibble(block[j]) | (nibble(block[j + BLOCK / 2]) << 4);
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::{BLOCK, DType, Tensor};

    fn ramp(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect()
    }

    /// Q4_0 maps the signed extreme to −8, so the opposite extreme clamps at +7: up to one full
    /// step of error there.
    fn assert_within_one_step(dtype: DType, block: &[f32], decoded: &[f32]) {
        let absmax = block.iter().fold(0.0f32, |max, x| max.max(x.abs()));
        let tolerance = match dtype {
            DType::Q7b32 => absmax / 126.0,
            _ => absmax / 8.0,
        };
        for (x, got) in block.iter().zip(decoded) {
            assert!(
                (got - x).abs() <= tolerance + 1e-5,
                "{dtype:?} {x} -> {got}"
            );
        }
    }

    #[test]
    fn quantized_rows_dequantize_within_one_step() {
        let values = ramp(4 * 64);
        for dtype in [DType::Q7b32, DType::Q4b32] {
            let tensor = Tensor::encode(dtype, vec![4, 64], &values);
            let matrix = tensor.matrix().unwrap();
            let mut row_values = vec![0.0; 64];
            for row in 0..4 {
                matrix.dequantize_row(row, &mut row_values);
                let originals = &values[row * 64..(row + 1) * 64];
                for (block, decoded) in originals.chunks(BLOCK).zip(row_values.chunks(BLOCK)) {
                    assert_within_one_step(dtype, block, decoded);
                }
            }
        }
    }
}
