//! The block-quantized formats: how a group of weights becomes codes and a scale, and back,
//! and the quad layout the kernels read.
//!
//! - `Q7b32`: `u = q + 64` with `q ∈ [−63, 63]`, so every stored code is a valid unsigned 7-bit
//!   operand of `i32x4.relaxed_dot_i8x16_i7x16_add_s`; value = `d · (u − 64)`.
//! - `Q4b32`: ggml Q4_0's levels, `u = q + 8` with `q ∈ [−8, 7]`; value = `d · (u − 8)`.
//! - `T2b128`: ternary weights, codes `u = q + 1` with `q ∈ {−1, 0, 1}` and one scale per 128
//!   values; value = `d · (u − 1)`.
//!
//! A matrix's rows are stored four at a time, a quad, so that one 128-bit dot-product
//! instruction computes four outputs: each block of 32 inputs of a quad is 8 chunks of 16
//! codes, chunk `c` holding inputs `4c..4c + 4` of each of the four rows in turn. A `Q7b32`
//! code takes a byte; a `Q4b32` byte holds a code of chunk `c < 4` in its low nibble and of
//! chunk `c + 4` in its high nibble; a `T2b128` byte holds the codes of chunks `c < 2`, `c + 2`,
//! `c + 4` and `c + 6` in its bit pairs, lowest first.

pub const BLOCK: usize = 32;
/// Values sharing one scale in `T2b128`.
pub const TERNARY_GROUP: usize = 128;
/// Rows stored together: the output columns a matmul stores at once.
pub const QUAD: usize = 4;
/// Consecutive inputs of one row in a chunk.
pub(crate) const CHUNK_INPUTS: usize = 16 / QUAD;
/// Bytes holding one chunk's codes (with those of the chunks sharing their bytes).
const WORD: usize = QUAD * CHUNK_INPUTS;

const Q7_MAX: f32 = 63.0;
const NIBBLE_MASK: u8 = 0x0f;

/// A block-quantized weight format: every `DType` but `F32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Q7b32,
    Q4b32,
    T2b128,
}

impl QuantFormat {
    /// Bytes of the quants of `count` values.
    pub(crate) fn quant_bytes(self, count: usize) -> usize {
        count / self.codes_per_byte()
    }

    /// Codes sharing one byte.
    fn codes_per_byte(self) -> usize {
        match self {
            Self::Q7b32 => 1,
            Self::Q4b32 => 2,
            Self::T2b128 => 4,
        }
    }

    /// Values sharing one scale.
    pub fn group(self) -> usize {
        match self {
            Self::Q7b32 | Self::Q4b32 => BLOCK,
            Self::T2b128 => TERNARY_GROUP,
        }
    }

    /// What a stored code must have subtracted to give the signed level.
    pub(crate) fn zero_point(self) -> i32 {
        match self {
            Self::Q7b32 => 64,
            Self::Q4b32 => 8,
            Self::T2b128 => 1,
        }
    }

    /// Encodes one group of `values` as one code per byte of `codes`; returns its scale.
    /// Quantization rounds ties to even.
    pub(crate) fn quantize(self, values: &[f32], codes: &mut [u8]) -> f32 {
        match self {
            Self::Q7b32 => quantize_q7(values, codes),
            Self::Q4b32 => quantize_q4(values, codes),
            Self::T2b128 => quantize_t2(values, codes),
        }
    }

    /// `out = d · (u − zero_point)` for one block of `codes` at scale `scale`.
    pub(crate) fn dequantize_block(self, codes: &[u8], scale: f32, out: &mut [f32]) {
        for (output, &code) in out.iter_mut().zip(codes) {
            *output = scale * (i32::from(code) - self.zero_point()) as f32;
        }
    }
}

/// The byte of a quad block that holds code `input` of row `lane`, and the code's shift in it.
fn code_position(format: QuantFormat, lane: usize, input: usize) -> (usize, u32) {
    let (chunk, offset) = (input / CHUNK_INPUTS, input % CHUNK_INPUTS);
    let words = BLOCK / CHUNK_INPUTS / format.codes_per_byte();
    let bits = 8 / format.codes_per_byte();
    let byte = chunk % words * WORD + lane * CHUNK_INPUTS + offset;
    (byte, (bits * (chunk / words)) as u32)
}

/// Sets row `lane` of a quad block of `format`, whose bits are still zero, to one block of
/// `codes`.
pub(crate) fn pack_lane(format: QuantFormat, lane: usize, codes: &[u8], quad: &mut [u8]) {
    for (input, &code) in codes[..BLOCK].iter().enumerate() {
        let (byte, shift) = code_position(format, lane, input);
        quad[byte] |= code << shift;
    }
}

/// The 32 codes of row `lane` of one stored quad block, in input order.
pub(crate) fn unpack_lane(format: QuantFormat, quad: &[u8], lane: usize) -> [u8; BLOCK] {
    let mask: u8 = 0xff >> (8 - 8 / format.codes_per_byte());
    std::array::from_fn(|input| {
        let (byte, shift) = code_position(format, lane, input);
        (quad[byte] >> shift) & mask
    })
}

/// The largest magnitude of `values`.
pub(crate) fn absmax(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |max, x| max.max(x.abs()))
}

/// `extreme / level`, the scale that maps a group's extreme value to `level` (1 for an all-zero
/// group).
pub(crate) fn group_scale(extreme: f32, level: f32) -> f32 {
    if extreme == 0.0 { 1.0 } else { extreme / level }
}

/// Symmetric 7-bit: `d = absmax / 63`, `q = round_ties_even(x / d)`.
fn quantize_q7(block: &[f32], codes: &mut [u8]) -> f32 {
    let scale = group_scale(absmax(block), Q7_MAX);
    let zero_point = QuantFormat::Q7b32.zero_point();
    for (code, x) in codes.iter_mut().zip(block) {
        let level = (x / scale).round_ties_even().clamp(-Q7_MAX, Q7_MAX) as i32;
        *code = (level + zero_point) as u8;
    }
    scale
}

/// ggml Q4_0: `d = signed extreme / −8`, `u = clamp(round(x / d) + 8, 0, 15)`.
fn quantize_q4(block: &[f32], codes: &mut [u8]) -> f32 {
    let extreme = block.iter().fold(
        0.0f32,
        |extreme, &x| {
            if x.abs() > extreme.abs() { x } else { extreme }
        },
    );
    let zero_point = QuantFormat::Q4b32.zero_point() as f32;
    let scale = group_scale(extreme, -zero_point);
    let largest = f32::from(NIBBLE_MASK);
    for (code, x) in codes.iter_mut().zip(block) {
        *code = ((x / scale).round_ties_even() + zero_point).clamp(0.0, largest) as u8;
    }
    scale
}

/// Ternary: `d = absmax`, `q = round_ties_even(x / d)` clamped to `[−1, 1]`.
fn quantize_t2(group: &[f32], codes: &mut [u8]) -> f32 {
    let zero_point = QuantFormat::T2b128.zero_point();
    let level = zero_point as f32;
    let scale = group_scale(absmax(group), level);
    for (code, x) in codes.iter_mut().zip(group) {
        *code = ((x / scale).round_ties_even().clamp(-level, level) as i32 + zero_point) as u8;
    }
    scale
}
