//! Matrix multiply against quantized or f32 weights.
//!
//! Quantized weights are multiplied with activations quantized to int8 per group (`d = absmax /
//! 127`, ties-even rounding), a group being the values that share a weight scale: 32 for `Q7b32`
//! and `Q4b32`, 128 for `T2b128`. A group's integer dot product is `Σ u·a − zero_point · Σ a`,
//! where `u` are the stored (offset) weight bytes; the correction is folded into the
//! accumulator's first lane before the dot instructions over the group's blocks, then the group
//! result is scaled by `d_w · d_a` into f32 lanes, as ggml's Q8_0 dot does.

#![expect(
    unsafe_code,
    reason = "unchecked block loads after one bounds assertion per call"
)]

use std::ops::Range;

use crate::arch::{Active, Backend, DOTS, activation_scale};
use crate::quant::{BLOCK, QuantFormat, TERNARY_GROUP};
use crate::tensor::{Matrix, MatrixStorage};

/// Rows of activations quantized for one weight format.
#[derive(Default)]
pub struct QuantizedRows {
    quants: Vec<i8>,
    /// One scale and one correction per group.
    scales: Vec<f32>,
    corrections: Vec<i32>,
}

/// A matrix quantized to `Q7b32` at run time, row by row (an attention's keys and values): its
/// `matrix` multiplies like a model's weights.
#[derive(Default)]
pub struct QuantizedMatrix {
    quants: Vec<u8>,
    scales: Vec<f32>,
    cols: usize,
}

impl QuantizedMatrix {
    /// Empties it for rows of `cols` values, a multiple of `BLOCK`.
    pub fn reset(&mut self, cols: usize) {
        assert!(cols.is_multiple_of(BLOCK), "rows of whole blocks");
        self.quants.clear();
        self.scales.clear();
        self.cols = cols;
    }

    /// Appends `row`, `cols` values.
    pub fn push(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.cols, "a row of the matrix's width");
        let start = self.quants.len();
        self.quants.resize(start + row.len(), 0);
        let blocks = row.as_chunks::<BLOCK>().0;
        let outputs = self.quants[start..].as_chunks_mut::<BLOCK>().0;
        for (block, out) in blocks.iter().zip(outputs) {
            self.scales.push(QuantFormat::Q7b32.quantize(block, out));
        }
    }

    pub fn matrix(&self) -> Matrix<'_> {
        Matrix {
            rows: self.quants.len() / self.cols.max(1),
            cols: self.cols,
            storage: MatrixStorage::Quant {
                format: QuantFormat::Q7b32,
                quants: &self.quants,
                scales: &self.scales,
            },
        }
    }
}

/// The activations a matmul reads: the f32 rows themselves for f32 weights, else their int8 form
/// for one quantized format.
pub enum Input<'rows> {
    F32(&'rows [f32]),
    Quant {
        rows: &'rows QuantizedRows,
        format: QuantFormat,
    },
}

impl Input<'_> {
    /// How many rows of `width` values the input holds.
    fn rows(&self, width: usize) -> usize {
        match self {
            Input::F32(values) => values.len() / width,
            Input::Quant { rows, .. } => rows.quants.len() / width,
        }
    }
}

impl QuantizedRows {
    /// Prepares `x`, whole rows of `weights.cols` values, as the input of `weights`.
    pub fn input<'rows>(&'rows mut self, weights: &Matrix, x: &'rows [f32]) -> Input<'rows> {
        assert!(
            x.len().is_multiple_of(weights.cols),
            "the input is whole rows of the weights' width"
        );
        match weights.storage {
            MatrixStorage::F32(_) => Input::F32(x),
            MatrixStorage::Quant { format, .. } => {
                self.quantize(x, format);
                Input::Quant { rows: self, format }
            }
        }
    }

    fn quantize(&mut self, x: &[f32], format: QuantFormat) {
        let group = format.group();
        let groups = x.len() / group;
        self.quants.resize(x.len(), 0);
        self.scales.resize(groups, 0.0);
        self.corrections.resize(groups, 0);
        let outputs = self.quants.chunks_exact_mut(group);
        let slots = outputs.zip(&mut self.scales).zip(&mut self.corrections);
        for (values, ((output, scale), correction)) in x.chunks_exact(group).zip(slots) {
            let blocks = values.as_chunks::<BLOCK>().0;
            let largest = blocks
                .iter()
                .map(|block| Active::absmax(block))
                .fold(0.0, f32::max);
            *scale = activation_scale(largest);
            let outputs = output.as_chunks_mut::<BLOCK>().0;
            let sums = blocks
                .iter()
                .zip(outputs)
                .map(|(block, out)| Active::quantize_block(block, *scale, out));
            *correction = -format.zero_point() * sums.sum::<i32>();
        }
    }
}

/// `store(row, col, Σ_k weights[col][k] · x[row][k])` for every input row and every output
/// column in `cols`.
///
/// # Panics
/// When `x` was prepared for another format than the weights'.
pub fn matmul(
    weights: &Matrix,
    x: &Input,
    cols: Range<usize>,
    store: impl FnMut(usize, usize, f32),
) {
    matmul_with::<Active>(weights, x, cols, store);
}

fn matmul_with<B: Backend>(
    weights: &Matrix,
    x: &Input,
    cols: Range<usize>,
    store: impl FnMut(usize, usize, f32),
) {
    let rows = x.rows(weights.cols);
    match (weights.storage, x) {
        (MatrixStorage::F32(values), Input::F32(x)) => {
            let dense = F32Weights {
                values,
                cols: weights.cols,
            };
            f32_matmul::<B>(&dense, x, rows, cols, store);
        }
        (
            MatrixStorage::Quant {
                format,
                quants,
                scales,
            },
            Input::Quant {
                rows: x,
                format: prepared,
            },
        ) => {
            assert_eq!(
                *prepared, format,
                "the input is prepared for the weights' format"
            );
            let quantized = QuantWeights {
                format,
                quants,
                scales,
                cols: weights.cols,
            };
            quant_matmul::<B>(&quantized, x, rows, cols, store);
        }
        _ => {
            unreachable!("Input::F32 is built for f32 weights and Input::Quant for quantized ones")
        }
    }
}

struct F32Weights<'tensor> {
    values: &'tensor [f32],
    cols: usize,
}

fn f32_matmul<B: Backend>(
    weights: &F32Weights,
    x: &[f32],
    rows: usize,
    cols: Range<usize>,
    mut store: impl FnMut(usize, usize, f32),
) {
    let width = weights.cols;
    let activation = |row: usize| &x[row * width..(row + 1) * width];
    let grouped = rows - rows % DOTS;
    for col in cols {
        let weight_row = &weights.values[col * width..(col + 1) * width];
        for first in (0..grouped).step_by(DOTS) {
            let group = std::array::from_fn(|offset| activation(first + offset));
            for (offset, value) in B::dot_f32_many(weight_row, group).into_iter().enumerate() {
                store(first + offset, col, value);
            }
        }
        for row in grouped..rows {
            store(row, col, B::dot_f32(weight_row, activation(row)));
        }
    }
}

struct QuantWeights<'tensor> {
    format: QuantFormat,
    quants: &'tensor [u8],
    scales: &'tensor [f32],
    cols: usize,
}

fn quant_matmul<B: Backend>(
    weights: &QuantWeights,
    x: &QuantizedRows,
    rows: usize,
    cols: Range<usize>,
    store: impl FnMut(usize, usize, f32),
) {
    match weights.format {
        QuantFormat::Q7b32 => quant_tiles::<Q7, B>(weights, x, rows, cols, store),
        QuantFormat::Q4b32 => quant_tiles::<Q4, B>(weights, x, rows, cols, store),
        QuantFormat::T2b128 => quant_tiles::<T2, B>(weights, x, rows, cols, store),
    }
}

/// How one quantized format stores its blocks of 32 weights.
trait Format {
    const BLOCK_BYTES: usize;
    /// Blocks along a row that share one weight scale (and one activation scale): a group.
    const BLOCKS_PER_GROUP: usize;

    /// # Safety
    /// `BLOCK_BYTES` readable bytes at `at`.
    unsafe fn load<B: Backend>(at: *const u8) -> B::Weights;
}

struct Q7;
struct Q4;
struct T2;

impl Format for Q7 {
    const BLOCK_BYTES: usize = BLOCK;
    const BLOCKS_PER_GROUP: usize = 1;

    #[inline(always)]
    unsafe fn load<B: Backend>(at: *const u8) -> B::Weights {
        // SAFETY: forwarded from the caller.
        unsafe { B::load_q7(at) }
    }
}

impl Format for Q4 {
    const BLOCK_BYTES: usize = BLOCK / 2;
    const BLOCKS_PER_GROUP: usize = 1;

    #[inline(always)]
    unsafe fn load<B: Backend>(at: *const u8) -> B::Weights {
        // SAFETY: forwarded from the caller.
        unsafe { B::load_q4(at) }
    }
}

impl Format for T2 {
    const BLOCK_BYTES: usize = BLOCK / 4;
    const BLOCKS_PER_GROUP: usize = TERNARY_GROUP / BLOCK;

    #[inline(always)]
    unsafe fn load<B: Backend>(at: *const u8) -> B::Weights {
        // SAFETY: forwarded from the caller.
        unsafe { B::load_t2(at) }
    }
}

fn quant_tiles<F: Format, B: Backend>(
    weights: &QuantWeights,
    x: &QuantizedRows,
    rows: usize,
    cols: Range<usize>,
    mut store: impl FnMut(usize, usize, f32),
) {
    let blocks = weights.cols / BLOCK;
    let groups = blocks / F::BLOCKS_PER_GROUP;
    assert!(
        weights.quants.len() >= cols.end * blocks * F::BLOCK_BYTES
            && weights.scales.len() >= cols.end * groups,
        "weights cover the columns"
    );
    assert!(
        x.quants.len() >= rows * weights.cols
            && x.scales.len() >= rows * groups
            && x.corrections.len() >= rows * groups,
        "activations cover the rows"
    );
    for start in (0..rows).step_by(B::TILE) {
        let tile = (rows - start).min(B::TILE);
        let mut store_tile = |row: usize, col: usize, value: f32| store(start + row, col, value);
        let span = TileSpan {
            first_row: start,
            cols: cols.clone(),
        };
        macro_rules! dispatch {
            ($($rows:literal)*) => {
                match tile {
                    $($rows => quant_tile::<$rows, F, B>(weights, x, &span, &mut store_tile),)*
                    _ => unreachable!("a backend's TILE is at most 6"),
                }
            };
        }
        dispatch!(1 2 3 4 5 6);
    }
}

/// The activation rows `first_row..first_row + ROWS` and weight rows `cols` of one tile.
struct TileSpan {
    first_row: usize,
    cols: Range<usize>,
}

/// One tile, two weight rows per pass so every activation block is loaded once for both; each
/// group's blocks add up as integers before one scaling.
#[inline(always)]
fn quant_tile<const ROWS: usize, F: Format, B: Backend>(
    weights: &QuantWeights,
    x: &QuantizedRows,
    span: &TileSpan,
    store: &mut impl FnMut(usize, usize, f32),
) {
    let blocks = weights.cols / BLOCK;
    let groups = blocks / F::BLOCKS_PER_GROUP;
    // `quant_tiles` asserted that the weights cover `cols.end · blocks` blocks and their group
    // scales, and the activations `rows · blocks` blocks and their group scales and corrections;
    // here `first_row + ROWS ≤ rows`, `col < cols.end`, `block < blocks` and `group < groups`.
    let load = |col: usize, block: usize| {
        let offset = (col * blocks + block) * F::BLOCK_BYTES;
        // SAFETY: the block lies within the asserted weight bytes (above).
        unsafe { F::load::<B>(weights.quants.as_ptr().add(offset)) }
    };
    let mut col = span.cols.start;
    while col < span.cols.end {
        let pair = col + 1 < span.cols.end;
        // The second weight row is `col` again when `col` is the last column (its sums are
        // dropped).
        let second = if pair { col + 1 } else { col };
        let mut first_sums = [B::zero(); ROWS];
        let mut second_sums = [B::zero(); ROWS];
        for group in 0..groups {
            let seeds: [B::Dots; ROWS] = std::array::from_fn(|row| {
                let index = (span.first_row + row) * groups + group;
                // SAFETY: the correction lies within the asserted activation corrections (above).
                B::seed(unsafe { *x.corrections.get_unchecked(index) })
            });
            let (mut first_dots, mut second_dots) = (seeds, seeds);
            for block in group * F::BLOCKS_PER_GROUP..(group + 1) * F::BLOCKS_PER_GROUP {
                let (first_weights, second_weights) = (load(col, block), load(second, block));
                let dots = first_dots.iter_mut().zip(second_dots.iter_mut());
                for (row, (first_dot, second_dot)) in dots.enumerate() {
                    let index = (span.first_row + row) * blocks + block;
                    // SAFETY: the block lies within the asserted activation bytes (above).
                    let activations =
                        unsafe { B::load_activations(x.quants.as_ptr().add(index * BLOCK)) };
                    *first_dot = B::dot(*first_dot, &activations, &first_weights);
                    *second_dot = B::dot(*second_dot, &activations, &second_weights);
                }
            }
            // SAFETY: both weight scales lie within the asserted weight scales (above).
            let (first_scale, second_scale) = unsafe {
                (
                    *weights.scales.get_unchecked(col * groups + group),
                    *weights.scales.get_unchecked(second * groups + group),
                )
            };
            for row in 0..ROWS {
                // SAFETY: the scale lies within the asserted activation scales (above).
                let scale = unsafe {
                    *x.scales
                        .get_unchecked((span.first_row + row) * groups + group)
                };
                first_sums[row] = B::scale(first_sums[row], first_dots[row], first_scale * scale);
                second_sums[row] =
                    B::scale(second_sums[row], second_dots[row], second_scale * scale);
            }
        }
        for (row, (first_sum, second_sum)) in first_sums.into_iter().zip(second_sums).enumerate() {
            store(row, col, B::sum(first_sum));
            if pair {
                store(row, col + 1, B::sum(second_sum));
            }
        }
        col += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::{QuantizedRows, matmul_with};
    use crate::arch::{Active, Backend, Scalar};
    use crate::tensor::{DType, Matrix, Table, Tensor};

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// The dequantized weights times the activations dequantized from int8 per `group` values,
    /// in f64.
    fn reference(weights: &Matrix, x: &[f32], (rows, group): (usize, usize)) -> Vec<f64> {
        let width = weights.cols;
        let mut weight_row = vec![0.0; width];
        let mut out = vec![0.0; rows * weights.rows];
        for col in 0..weights.rows {
            weights.dequantize_row(col, &mut weight_row);
            for row in 0..rows {
                let quantized = x[row * width..(row + 1) * width]
                    .chunks(group)
                    .flat_map(|block| {
                        let scale = block.iter().fold(0.0f32, |max, x| max.max(x.abs())) / 127.0;
                        block
                            .iter()
                            .map(move |x| f64::from((x / scale).round_ties_even() * scale))
                    });
                let products = weight_row
                    .iter()
                    .zip(quantized)
                    .map(|(&weight, x)| f64::from(weight) * x);
                out[row * weights.rows + col] = products.sum();
            }
        }
        out
    }

    fn assert_matches_dequantized_product<B: Backend>() {
        let (outputs, width) = (25, 128);
        let values = pseudo_random(outputs * width, 1);
        for dtype in [DType::Q7b32, DType::Q4b32, DType::T2b128] {
            let tensor = Tensor::encode(dtype, vec![outputs, width], &values).unwrap();
            let table = Table::new(tensor).unwrap();
            let weights = table.matrix();
            let group = dtype.quant_format().unwrap().group();
            for rows in [1, 3, 8, 11] {
                let x = pseudo_random(rows * width, 7 + rows as u32);
                let expected = reference(&weights, &x, (rows, group));
                let mut scratch = QuantizedRows::default();
                let input = scratch.input(&weights, &x);
                let mut got = vec![0.0f32; rows * outputs];
                matmul_with::<B>(&weights, &input, 0..outputs, |row, col, value| {
                    got[row * outputs + col] = value;
                });
                for (value, want) in got.iter().zip(&expected) {
                    let error = (f64::from(*value) - want).abs();
                    assert!(
                        error < 1e-4 * (1.0 + want.abs()),
                        "{dtype:?} rows={rows}: {value} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_the_dequantized_product() {
        assert_matches_dequantized_product::<Active>();
        assert_matches_dequantized_product::<Scalar>();
    }
}
