//! Weight tensors, and the views of them a model reads once their type is checked.
//!
//! A quantized matrix `[rows, cols]` is stored as one byte buffer: the quants of its rows, four
//! at a time in `quant`'s quad layout (the rows padded to whole quads with zero codes; per quad
//! and block of 32 inputs 128 bytes for `Q7b32`, 64 for `Q4b32`, 32 for `T2b128`), followed by
//! one f32 scale per row and group along a row (32 values for `Q7b32` and `Q4b32`, 128 for
//! `T2b128`): quad-major, then group, then the quad's four rows. `cols` is a multiple of the
//! group, so the scales start on a 4-byte boundary and are read as f32 in place. The formats
//! are `quant`'s.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::blob::Blob;
use crate::quant::{BLOCK, QUAD, QuantFormat, pack_lane, unpack_lane};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    Q7b32,
    Q4b32,
    T2b128,
}

impl DType {
    /// The block-quantized format, `None` for f32.
    pub fn quant_format(self) -> Option<QuantFormat> {
        match self {
            DType::F32 => None,
            DType::Q7b32 => Some(QuantFormat::Q7b32),
            DType::Q4b32 => Some(QuantFormat::Q4b32),
            DType::T2b128 => Some(QuantFormat::T2b128),
        }
    }

    /// Bytes of a tensor of this type with the given shape (`[rows, cols]` for quantized
    /// matrices).
    fn byte_len(self, shape: &[usize]) -> usize {
        let count: usize = shape.iter().product();
        match self.quant_format() {
            None => count * size_of::<f32>(),
            Some(format) => {
                let padded = match shape {
                    [rows, cols] => padded_values(*rows, *cols),
                    _ => count,
                };
                format.quant_bytes(padded) + padded / format.group() * size_of::<f32>()
            }
        }
    }
}

/// Values of a `[rows, cols]` quantized matrix once its rows are padded to whole quads.
fn padded_values(rows: usize, cols: usize) -> usize {
    rows.div_ceil(QUAD) * QUAD * cols
}

pub struct Tensor {
    dtype: DType,
    shape: Vec<usize>,
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
        format: QuantFormat,
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
    /// rows are whole groups; quantization rounds ties to even. `T2b128` keeps ternary weights
    /// (`d · {−1, 0, 1}` per group) exactly; other values lose all but their sign and the
    /// largest magnitude of their group.
    ///
    /// # Errors
    /// When `values` does not fill `shape`, or a quantized `shape` is not such a matrix.
    pub fn encode(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Result<Self, Error> {
        let count: usize = shape.iter().product();
        if values.len() != count {
            return Err(Error::Format(format!(
                "{} values for a tensor of shape {shape:?}",
                values.len()
            )));
        }
        let format = dtype.quant_format();
        if let Some(format) = format {
            check_quantized(format, &shape)?;
        }
        let mut tensor = Self::zeroed(dtype, shape);
        match format {
            None => {
                let words = tensor.blob.bytes_mut().as_chunks_mut::<4>().0;
                for (word, value) in words.iter_mut().zip(values) {
                    word.copy_from_slice(&value.to_le_bytes());
                }
            }
            Some(format) => tensor.encode_quantized(format, values),
        }
        Ok(tensor)
    }

    /// Row by row into the quad layout: each row's codes into its lane of its quad's blocks,
    /// its scales into its slots.
    fn encode_quantized(&mut self, format: QuantFormat, values: &[f32]) {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let quant_len = format.quant_bytes(padded_values(rows, cols));
        let (quants, scales) = self.blob.bytes_mut().split_at_mut(quant_len);
        let (group, blocks) = (format.group(), cols / BLOCK);
        let quad_bytes = format.quant_bytes(QUAD * BLOCK);
        let mut codes = vec![0; cols];
        for (row, row_values) in values.chunks_exact(cols).enumerate() {
            let (quad, lane) = (row / QUAD, row % QUAD);
            for (index, group_values) in row_values.chunks_exact(group).enumerate() {
                let scale = format.quantize(group_values, &mut codes[index * group..][..group]);
                let slot = ((quad * (cols / group) + index) * QUAD + lane) * size_of::<f32>();
                scales[slot..slot + size_of::<f32>()].copy_from_slice(&scale.to_le_bytes());
            }
            for (block, block_codes) in codes.as_chunks::<BLOCK>().0.iter().enumerate() {
                let at = (quad * blocks + block) * quad_bytes;
                pack_lane(format, lane, block_codes, &mut quants[at..at + quad_bytes]);
            }
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn bytes(&self) -> &[u8] {
        self.blob.bytes()
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.blob.bytes_mut()
    }
}

/// A quantized tensor must be a `[rows, cols]` matrix of whole groups.
fn check_quantized(format: QuantFormat, shape: &[usize]) -> Result<(), Error> {
    match shape {
        [_, cols] if cols.is_multiple_of(format.group()) => Ok(()),
        _ => Err(Error::Format(format!(
            "a {format:?} tensor of shape {shape:?}: a matrix of whole groups of {} expected",
            format.group()
        ))),
    }
}

/// An f32 vector: a norm's weights, a bias, a window.
pub struct Vector(Tensor);

impl Vector {
    /// # Errors
    /// When `tensor` is not a vector of f32.
    pub fn new(tensor: Tensor) -> Result<Self, Error> {
        if tensor.dtype != DType::F32 || tensor.shape.len() != 1 {
            return Err(Error::Format(format!(
                "a {:?} tensor of shape {:?}: an f32 vector expected",
                tensor.dtype, tensor.shape
            )));
        }
        Ok(Self(tensor))
    }

    pub fn values(&self) -> &[f32] {
        self.0.blob.f32s(0)
    }
}

/// A `[rows, cols]` matrix in any format: a linear layer's weights, an embedding table.
pub struct Table(Tensor);

impl Table {
    /// # Errors
    /// When `tensor` is not a matrix, or a quantized one's rows are not whole groups.
    pub fn new(tensor: Tensor) -> Result<Self, Error> {
        if let Some(format) = tensor.dtype.quant_format() {
            check_quantized(format, &tensor.shape)?;
        }
        if tensor.shape.len() != 2 {
            return Err(Error::Format(format!(
                "a tensor of shape {:?}: a matrix expected",
                tensor.shape
            )));
        }
        Ok(Self(tensor))
    }

    pub fn matrix(&self) -> Matrix<'_> {
        let Tensor { dtype, shape, blob } = &self.0;
        let (rows, cols) = (shape[0], shape[1]);
        let storage = match dtype.quant_format() {
            None => MatrixStorage::F32(blob.f32s(0)),
            Some(format) => {
                let quant_len = format.quant_bytes(padded_values(rows, cols));
                MatrixStorage::Quant {
                    format,
                    quants: &blob.bytes()[..quant_len],
                    scales: blob.f32s(quant_len),
                }
            }
        };
        Matrix {
            rows,
            cols,
            storage,
        }
    }

    /// Every row as f32, row-major.
    pub fn dequantize(&self) -> Vec<f32> {
        let matrix = self.matrix();
        let mut values = vec![0.0; matrix.rows * matrix.cols];
        for (row, out) in values.chunks_exact_mut(matrix.cols).enumerate() {
            matrix.dequantize_row(row, out);
        }
        values
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
                format,
                quants,
                scales,
            } => {
                let (quad, lane) = (row / QUAD, row % QUAD);
                let (blocks, groups) = (cols / BLOCK, cols / format.group());
                let quad_bytes = format.quant_bytes(QUAD * BLOCK);
                let blocks_per_group = format.group() / BLOCK;
                let outputs = out.as_chunks_mut::<BLOCK>().0;
                for (block, output) in outputs.iter_mut().enumerate() {
                    let at = (quad * blocks + block) * quad_bytes;
                    let codes = unpack_lane(format, &quants[at..at + quad_bytes], lane);
                    let scale = scales[(quad * groups + block / blocks_per_group) * QUAD + lane];
                    format.dequantize_block(&codes, scale, output);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, Table, Tensor};
    use crate::quant::{BLOCK, TERNARY_GROUP};

    fn ramp(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect()
    }

    /// Q4_0 maps the signed extreme to −8, so the opposite extreme clamps at +7: up to one full
    /// step of error there.
    fn assert_within_one_step(dtype: DType, block: &[f32], decoded: &[f32]) {
        let largest = block.iter().fold(0.0f32, |max, x| max.max(x.abs()));
        let tolerance = match dtype {
            DType::Q7b32 => largest / 126.0,
            _ => largest / 8.0,
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
            let tensor = Tensor::encode(dtype, vec![4, 64], &values).unwrap();
            let table = Table::new(tensor).unwrap();
            let matrix = table.matrix();
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

    #[test]
    fn ternary_rows_keep_their_codes_and_group_scales() {
        let cols = 2 * TERNARY_GROUP;
        let scales = [0.5, 0.25, 2.0, 0.125];
        let values: Vec<f32> = (0..2 * cols)
            .map(|i| scales[i / TERNARY_GROUP] * ((i * 7 % 3) as f32 - 1.0))
            .collect();
        let tensor = Tensor::encode(DType::T2b128, vec![2, cols], &values).unwrap();
        // Elements 0, 8, 16 and 24 have codes 1 − 1, 1 + 1, 1 + 0 and 1 − 1: 0b00_01_10_00.
        assert_eq!(tensor.bytes()[0], 0b0001_1000);
        assert_eq!(Table::new(tensor).unwrap().dequantize(), values);
    }

    #[test]
    fn encoding_rejects_values_that_do_not_fill_the_shape() {
        let cases = [
            (DType::F32, vec![3], 4, "4 values for a tensor of shape [3]"),
            (
                DType::Q7b32,
                vec![2, 32],
                63,
                "63 values for a tensor of shape [2, 32]",
            ),
            (
                DType::Q7b32,
                vec![64],
                64,
                "a Q7b32 tensor of shape [64]: a matrix of whole groups of 32 expected",
            ),
            (
                DType::T2b128,
                vec![1, 64],
                64,
                "a T2b128 tensor of shape [1, 64]: a matrix of whole groups of 128 expected",
            ),
        ];
        for (dtype, shape, len, message) in cases {
            let Err(error) = Tensor::encode(dtype, shape, &vec![1.0; len]) else {
                panic!("{message}: encoded");
            };
            assert_eq!(error.to_string(), message);
        }
    }
}
