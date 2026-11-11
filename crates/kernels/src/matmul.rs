//! Matrix multiply against quantized or f32 weights.
//!
//! Quantized weights are multiplied with activations quantized to int8 per group (`d = absmax /
//! 127`, ties-even rounding), a group being the values that share a weight scale: 32 for `Q7b32`
//! and `Q4b32`, 128 for `T2b128`. A group's integer dot product is `Σ u·a − zero_point · Σ a`,
//! where `u` are the stored (offset) weight codes; the correction seeds the accumulator before
//! the dot instructions over the group's blocks, then the group result is scaled by `d_w · d_a`
//! into f32, as ggml's Q8_0 dot does. The weights' quad layout (`quant`) gives every dot
//! instruction four output columns, one per lane.

#![expect(
    unsafe_code,
    reason = "unchecked block loads after one bounds assertion per call"
)]

use std::hint::black_box;
use std::mem::MaybeUninit;
use std::ops::Range;

use crate::arch::{Active, Backend, CHUNKS, activation_scale, dot_f32_each_with};
use crate::quant::{BLOCK, CHUNK_INPUTS, QUAD, QuantFormat, TERNARY_GROUP, WORD, pack_lane};
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
    rows: usize,
    codes: [u8; BLOCK],
}

impl QuantizedMatrix {
    /// Empties it for rows of `cols` values, a multiple of `BLOCK`.
    pub fn reset(&mut self, cols: usize) {
        assert!(cols.is_multiple_of(BLOCK), "rows of whole blocks");
        self.quants.clear();
        self.scales.clear();
        self.cols = cols;
        self.rows = 0;
    }

    /// Appends `row`, `cols` values.
    pub fn push(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.cols, "a row of the matrix's width");
        let format = QuantFormat::Q7b32;
        let (quad, lane) = (self.rows / QUAD, self.rows % QUAD);
        let blocks = self.cols / BLOCK;
        let quad_bytes = format.quant_bytes(QUAD * BLOCK);
        if lane == 0 {
            self.quants.resize((quad + 1) * blocks * quad_bytes, 0);
            self.scales.resize((quad + 1) * blocks * QUAD, 0.0);
        }
        for (block, values) in row.as_chunks::<BLOCK>().0.iter().enumerate() {
            let index = quad * blocks + block;
            self.scales[index * QUAD + lane] = format.quantize(values, &mut self.codes);
            let at = index * quad_bytes;
            pack_lane(
                format,
                lane,
                &self.codes,
                &mut self.quants[at..at + quad_bytes],
            );
        }
        self.rows += 1;
    }

    pub fn matrix(&self) -> Matrix<'_> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            storage: MatrixStorage::Quant {
                format: QuantFormat::Q7b32,
                quants: &self.quants,
                scales: &self.scales,
            },
        }
    }
}

/// Rows quantized to int8 for one weight format, as `matmul` reads them: `quants` row-major,
/// one scale and one correction per group.
#[derive(Clone, Copy)]
pub struct Quantized<'rows> {
    pub format: QuantFormat,
    pub quants: &'rows [i8],
    pub scales: &'rows [f32],
    pub corrections: &'rows [i32],
}

/// The activations a matmul reads: the f32 rows themselves for f32 weights, else their int8 form
/// for one quantized format.
pub enum Input<'rows> {
    F32(&'rows [f32]),
    Quant(Quantized<'rows>),
}

impl Input<'_> {
    /// How many rows of `width` values the input holds.
    fn rows(&self, width: usize) -> usize {
        match self {
            Input::F32(values) => values.len() / width,
            Input::Quant(rows) => rows.quants.len() / width,
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
        let MatrixStorage::Quant { format, .. } = weights.storage else {
            return Input::F32(x);
        };
        let groups = x.len() / format.group();
        self.quants.resize(x.len(), 0);
        self.scales.resize(groups, 0.0);
        self.corrections.resize(groups, 0);
        quantize_rows(
            x,
            format,
            &mut self.quants,
            &mut self.scales,
            &mut self.corrections,
        );
        Input::Quant(Quantized {
            format,
            quants: &self.quants,
            scales: &self.scales,
            corrections: &self.corrections,
        })
    }
}

/// `x`, whole groups of `format`, to int8 (`d = absmax / 127` per group, ties-even rounding)
/// into `quants`, with each group's scale and correction (`−zero_point · Σq`).
pub fn quantize_rows(
    x: &[f32],
    format: QuantFormat,
    quants: &mut [i8],
    scales: &mut [f32],
    corrections: &mut [i32],
) {
    let group = format.group();
    let outputs = quants.chunks_exact_mut(group);
    let slots = outputs.zip(scales).zip(corrections);
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

/// `store(row, col, values)` for every input row and every quad of output columns in `cols`,
/// `values[i]` being `Σ_k weights[col + i][k] · x[row][k]` (0 for a column past the weights'
/// rows). `cols` starts and ends on quads.
///
/// # Panics
/// When `x` was prepared for another format than the weights', or `cols` is not whole quads.
pub fn matmul(
    weights: &Matrix,
    x: &Input,
    cols: Range<usize>,
    store: impl FnMut(usize, usize, [f32; QUAD]),
) {
    assert!(
        cols.start.is_multiple_of(QUAD) && cols.end.is_multiple_of(QUAD),
        "columns in whole quads"
    );
    matmul_with::<Active>(weights, x, cols, store);
}

fn matmul_with<B: Backend>(
    weights: &Matrix,
    x: &Input,
    cols: Range<usize>,
    store: impl FnMut(usize, usize, [f32; QUAD]),
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
            Input::Quant(x),
        ) => {
            assert_eq!(
                x.format, format,
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
    mut store: impl FnMut(usize, usize, [f32; QUAD]),
) {
    let width = weights.cols;
    let outputs = weights.values.len() / width;
    let activation = |row: usize| &x[row * width..(row + 1) * width];
    let mut strip = vec![[0.0; QUAD]; rows];
    for first in cols.step_by(QUAD) {
        for (lane, col) in (first..first + QUAD).enumerate() {
            if col >= outputs {
                break;
            }
            let weight_row = &weights.values[col * width..(col + 1) * width];
            dot_f32_each_with::<B>(weight_row, rows, activation, |row, value| {
                strip[row][lane] = value;
            });
        }
        for (row, values) in strip.iter().enumerate() {
            store(row, first, *values);
        }
    }
}

struct QuantWeights<'tensor> {
    format: QuantFormat,
    quants: &'tensor [u8],
    scales: &'tensor [f32],
    cols: usize,
}

/// Rows per call of `quant_tiles`: V8 optimizes a function once a call has spent its tiering
/// budget, so the first product of an image runs its first rows in baseline code and the
/// rest optimized, rather than all of its thousands of rows.
const ROWS_PER_CALL: usize = 64;

fn quant_matmul<B: Backend>(
    weights: &QuantWeights,
    x: &Quantized,
    rows: usize,
    cols: Range<usize>,
    mut store: impl FnMut(usize, usize, [f32; QUAD]),
) {
    for first in (0..rows).step_by(ROWS_PER_CALL) {
        let rows = first..(first + ROWS_PER_CALL).min(rows);
        match weights.format {
            QuantFormat::Q7b32 => quant_tiles::<Q7, B>(weights, x, rows, cols.clone(), &mut store),
            QuantFormat::Q4b32 => quant_tiles::<Q4, B>(weights, x, rows, cols.clone(), &mut store),
            QuantFormat::T2b128 => quant_tiles::<T2, B>(weights, x, rows, cols.clone(), &mut store),
        }
    }
}

/// How one quantized format stores a quad's block of 32 weights.
trait Format {
    /// Words of a quad block, each holding the codes of `CHUNKS / WORDS` chunks: chunk `c`'s
    /// are part `c / WORDS` of word `c % WORDS`.
    const WORDS: usize;
    const QUAD_BYTES: usize = Self::WORDS * WORD;
    /// Blocks along a row that share one weight scale (and one activation scale): a group.
    const BLOCKS_PER_GROUP: usize = 1;

    /// The codes of chunk `part` of `word`.
    fn chunk<B: Backend>(word: B::Chunk, part: usize) -> B::Chunk;
}

struct Q7;
struct Q4;
struct T2;

impl Format for Q7 {
    const WORDS: usize = CHUNKS;

    #[inline(always)]
    fn chunk<B: Backend>(word: B::Chunk, _part: usize) -> B::Chunk {
        word
    }
}

impl Format for Q4 {
    const WORDS: usize = CHUNKS / 2;

    #[inline(always)]
    fn chunk<B: Backend>(word: B::Chunk, part: usize) -> B::Chunk {
        B::nibbles(word, part)
    }
}

impl Format for T2 {
    const WORDS: usize = CHUNKS / 4;
    const BLOCKS_PER_GROUP: usize = TERNARY_GROUP / BLOCK;

    #[inline(always)]
    fn chunk<B: Backend>(word: B::Chunk, part: usize) -> B::Chunk {
        B::bit_pairs(word, part)
    }
}

/// How the dots read a tile's activations, a chunk at a time: four values of a row on every
/// lane.
trait Activations<B: Backend> {
    /// Bytes from one chunk of a row to the next.
    const CHUNK_BYTES: usize;

    /// # Safety
    /// A chunk readable at `at`.
    unsafe fn chunk(at: *const u8) -> B::Chunk;
}

/// The quantized rows as they lie: a chunk's four values are broadcast at every read.
struct Compact;

/// A tile's rows broadcast beforehand, a chunk per four values. V8 pays a vector operation for a
/// broadcast and none for a load, and a tile reads each chunk once per pass.
struct Broadcast;

impl<B: Backend> Activations<B> for Compact {
    const CHUNK_BYTES: usize = CHUNK_INPUTS;

    #[inline(always)]
    unsafe fn chunk(at: *const u8) -> B::Chunk {
        // SAFETY: forwarded from the caller.
        unsafe { B::broadcast(at.cast()) }
    }
}

impl<B: Backend> Activations<B> for Broadcast {
    const CHUNK_BYTES: usize = size_of::<B::Chunk>();

    #[inline(always)]
    unsafe fn chunk(at: *const u8) -> B::Chunk {
        // SAFETY: forwarded from the caller; `broadcast_tile` stored the chunk, aligned.
        unsafe { *at.cast() }
    }
}

/// Quads per pass of a tile: 16 output columns.
const QUADS: usize = 4;
/// The widest rows a full tile broadcasts beforehand: four of them, broadcast, are a quarter
/// of a 128 KB L1 beside a pass's weights; wider, the copy costs more than its broadcasts.
const BROADCAST_COLS: usize = 2048;
/// The rows of the tallest tile (`Backend::TILE`).
const MAX_TILE: usize = 6;

/// Never inlined: V8 optimizes a wasm function only on a later call, once its tiering budget is
/// spent, never a loop already running (no on-stack replacement), so the hot loops live in a
/// function called per `ROWS_PER_CALL` rows rather than in a caller entered once per image.
#[inline(never)]
fn quant_tiles<F: Format, B: Backend>(
    weights: &QuantWeights,
    x: &Quantized,
    rows: Range<usize>,
    cols: Range<usize>,
    store: &mut impl FnMut(usize, usize, [f32; QUAD]),
) {
    let blocks = weights.cols / BLOCK;
    let groups = blocks / F::BLOCKS_PER_GROUP;
    let quads = cols.end / QUAD;
    assert!(
        weights.quants.len() >= quads * blocks * F::QUAD_BYTES
            && weights.scales.len() >= quads * groups * QUAD,
        "weights cover the columns"
    );
    assert!(
        x.quants.len() >= rows.end * weights.cols
            && x.scales.len() >= rows.end * groups
            && x.corrections.len() >= rows.end * groups,
        "activations cover the rows"
    );
    let one = black_box(1);
    let mut broadcast = [const { MaybeUninit::<B::Chunk>::uninit() }; TILE_CHUNKS];
    for start in rows.clone().step_by(B::TILE) {
        let tile = (rows.end - start).min(B::TILE);
        let quants = &x.quants[start * weights.cols..(start + tile) * weights.cols];
        let mut store_tile =
            |row: usize, col: usize, values: [f32; QUAD]| store(start + row, col, values);
        macro_rules! dispatch {
            ($activations:ty, $first:expr, $stride:expr, $($tile:literal)*) => {{
                let rows = Strided {
                    first: $first,
                    stride: $stride,
                };
                let span = TileSpan {
                    first_row: start,
                    rows,
                    cols: cols.clone(),
                    one,
                };
                match tile {
                    $($tile => {
                        quant_tile::<$tile, $activations, F, B>(weights, x, &span, &mut store_tile);
                    })*
                    _ => unreachable!("a backend's TILE is at most {MAX_TILE}"),
                }
            }};
        }
        if tile == B::TILE && weights.cols <= BROADCAST_COLS {
            broadcast_tile::<B>(quants, &mut broadcast);
            let stride = weights.cols / CHUNK_INPUTS * size_of::<B::Chunk>();
            dispatch!(Broadcast, broadcast.as_ptr().cast(), stride, 1 2 3 4 5 6);
        } else {
            dispatch!(Compact, quants.as_ptr().cast(), weights.cols, 1 2 3 4 5 6);
        }
    }
}

/// Chunks of the largest tile that is broadcast beforehand.
const TILE_CHUNKS: usize = MAX_TILE * BROADCAST_COLS / CHUNK_INPUTS;

/// `quants`, the rows of a tile of at most `TILE_CHUNKS` chunks, into `chunks`: each four
/// values on every lane.
fn broadcast_tile<B: Backend>(quants: &[i8], chunks: &mut [MaybeUninit<B::Chunk>; TILE_CHUNKS]) {
    for (chunk, inputs) in chunks.iter_mut().zip(quants.as_chunks::<CHUNK_INPUTS>().0) {
        // SAFETY: four readable values.
        chunk.write(unsafe { B::broadcast(inputs.as_ptr()) });
    }
}

/// The activation rows `first_row..first_row + ROWS` and weight rows `cols` of one tile.
struct TileSpan {
    first_row: usize,
    /// The rows' first chunks, as the tile's `Activations` read them.
    rows: Strided,
    cols: Range<usize>,
    /// 1, as the compiler cannot see it: its multiples are run-time values to the tile's loops.
    one: usize,
}

/// One tile: `QUADS` quads per pass, then the quads left over one at a time. A single row
/// (a token being written) reads its weights once from memory, and one quad per pass keeps
/// that read one sequential stream: four interleaved streams run at half the bandwidth.
#[inline(always)]
fn quant_tile<const ROWS: usize, A: Activations<B>, F: Format, B: Backend>(
    weights: &QuantWeights,
    x: &Quantized,
    span: &TileSpan,
    store: &mut impl FnMut(usize, usize, [f32; QUAD]),
) {
    let (first, end) = (span.cols.start / QUAD, span.cols.end / QUAD);
    let full = if ROWS == 1 {
        first
    } else {
        first + (end - first) / QUADS * QUADS
    };
    for quad in (first..full).step_by(QUADS) {
        quad_pass::<ROWS, QUADS, A, F, B>(weights, x, span, quad, store);
    }
    for quad in full..end {
        quad_pass::<ROWS, 1, A, F, B>(weights, x, span, quad, store);
    }
}

/// `ROWS` activation rows against `PASS` quads from `first_quad`; each group's blocks add up as
/// integers before one scaling.
#[inline(always)]
fn quad_pass<const ROWS: usize, const PASS: usize, A: Activations<B>, F: Format, B: Backend>(
    weights: &QuantWeights,
    x: &Quantized,
    span: &TileSpan,
    first_quad: usize,
    store: &mut impl FnMut(usize, usize, [f32; QUAD]),
) {
    let blocks = weights.cols / BLOCK;
    let groups = blocks / F::BLOCKS_PER_GROUP;
    // `quant_tiles` asserted that the weights cover the quads of `cols.end` columns (their
    // `blocks` blocks and group scales) and the activations `rows · blocks` blocks and their
    // group scales and corrections, and `span.rows` holds the tile's; here `first_row + ROWS ≤
    // rows`, `first_quad + PASS ≤ cols.end / 4`, `block < blocks` and `group < groups`.
    let load_scales = |quad: usize, group: usize| {
        let index = (quad * groups + group) * QUAD;
        // SAFETY: the quad's scales lie within the asserted weight scales (above).
        unsafe { B::load_scales(weights.scales.as_ptr().add(index)) }
    };
    let activation = |row: usize, group: usize| (span.first_row + row) * groups + group;
    let mut sums = [[B::zero(); PASS]; ROWS];
    // The sums live in memory between groups, behind a row stride the compiler cannot see: two
    // rows' sums may then alias, so each is loaded where it is used and stored where it is
    // made. As locals they are `ROWS · PASS` more vectors than V8 has registers for beside the
    // accumulators, and it spills accumulators in the middle of their dots to make room.
    let stride = PASS * span.one;
    let sums_at: *mut B::Sums = sums.as_mut_ptr().cast();
    // A trip count the compiler cannot see keeps `accumulate`'s words a loop, in which every
    // accumulator takes one dot per word and part. Unrolled, the wasm backend nests each
    // accumulator's dots into one expression and V8 runs them as written: one dependent chain
    // after another, with a whole block's chunks and broadcasts live in registers it does not
    // have. A single row is one chain per quad either way, and unrolled it does without the
    // loop's overhead.
    let words = if ROWS == 1 {
        F::WORDS
    } else {
        F::WORDS * span.one
    };
    for group in 0..groups {
        let mut dots: [[B::Dots; PASS]; ROWS] = std::array::from_fn(|row| {
            // SAFETY: the correction lies within the asserted activation corrections (above).
            [B::seed(unsafe { *x.corrections.get_unchecked(activation(row, group)) }); PASS]
        });
        for block in group * F::BLOCKS_PER_GROUP..(group + 1) * F::BLOCKS_PER_GROUP {
            // SAFETY: the blocks lie within the asserted weight bytes and the tile's rows (above).
            let (quads, rows) = unsafe {
                let quad = weights
                    .quants
                    .as_ptr()
                    .add((first_quad * blocks + block) * F::QUAD_BYTES);
                let row = span.rows.at(0, block * CHUNKS * A::CHUNK_BYTES);
                let quads = Strided {
                    first: quad,
                    stride: blocks * F::QUAD_BYTES,
                };
                let rows = Strided {
                    first: row,
                    stride: span.rows.stride,
                };
                (quads, rows)
            };
            accumulate::<ROWS, PASS, A, F, B>(&mut dots, quads, rows, words);
        }
        let column_scales: [B::Sums; PASS] =
            std::array::from_fn(|quad| load_scales(first_quad + quad, group));
        for (row, row_dots) in dots.iter().enumerate() {
            // SAFETY: the scale lies within the asserted activation scales (above).
            let row_scale = unsafe { *x.scales.get_unchecked(activation(row, group)) };
            for (quad, (dot, scales)) in row_dots.iter().zip(&column_scales).enumerate() {
                // SAFETY: `row < ROWS`, `quad < PASS` and `stride` is `PASS`: within `sums`.
                unsafe {
                    let sum = sums_at.add(row * stride + quad);
                    *sum = B::scale(*sum, *dot, *scales, row_scale);
                }
            }
        }
    }
    for row in 0..ROWS {
        for quad in 0..PASS {
            // SAFETY: `row < ROWS`, `quad < PASS` and `stride` is `PASS`: within `sums`.
            let sum = unsafe { *sums_at.add(row * stride + quad) };
            store(row, (first_quad + quad) * QUAD, B::lanes(sum));
        }
    }
}

/// Operands `stride` bytes apart from `first`: a pass's quads of weights, or a tile's rows of
/// activations.
#[derive(Clone, Copy)]
struct Strided {
    first: *const u8,
    stride: usize,
}

impl Strided {
    /// Operand `index`, `offset` bytes in.
    ///
    /// # Safety
    /// The bytes read there lie within the operands' allocation.
    unsafe fn at(self, index: usize, offset: usize) -> *const u8 {
        // SAFETY: forwarded from the caller.
        unsafe { self.first.add(index * self.stride + offset) }
    }
}

/// `dots[row][quad] += Σ` over one block, a word of every quad at a time (`words`, the
/// format's `WORDS`): each of its chunks against every row's same four activations.
#[inline(always)]
fn accumulate<const ROWS: usize, const PASS: usize, A: Activations<B>, F: Format, B: Backend>(
    dots: &mut [[B::Dots; PASS]; ROWS],
    quads: Strided,
    rows: Strided,
    words: usize,
) {
    for word in 0..words {
        // SAFETY: the word lies within the weight bytes `quant_tiles` asserted.
        let codes: [B::Chunk; PASS] =
            std::array::from_fn(|quad| unsafe { B::load_word(quads.at(quad, word * WORD)) });
        for part in 0..CHUNKS / F::WORDS {
            let chunks = codes.map(|codes| F::chunk::<B>(codes, part));
            let chunk = part * F::WORDS + word;
            for (row, dots) in dots.iter_mut().enumerate() {
                // SAFETY: the chunk lies within the tile's rows.
                let activations = unsafe { A::chunk(rows.at(row, chunk * A::CHUNK_BYTES)) };
                for (dot, chunk) in dots.iter_mut().zip(chunks) {
                    *dot = B::dot(*dot, activations, chunk);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{QuantizedRows, matmul_with};
    use crate::arch::{Active, Backend, Scalar};
    use crate::quant::QUAD;
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
                let quads = outputs.next_multiple_of(QUAD);
                let mut got = vec![0.0f32; rows * quads];
                matmul_with::<B>(&weights, &input, 0..quads, |row, col, values| {
                    got[row * quads + col..][..QUAD].copy_from_slice(&values);
                });
                let got = got.chunks_exact(quads).flat_map(|row| &row[..outputs]);
                for (value, want) in got.zip(&expected) {
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
