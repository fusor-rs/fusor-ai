//! The block-quantized formats: how a group of weights becomes codes and a scale, and back.
//!
//! - `Q7b32`: `u = q + 64` with `q ∈ [−63, 63]`, so every stored byte is a valid unsigned 7-bit
//!   operand of `i32x4.relaxed_dot_i8x16_i7x16_add_s`; value = `d · (u − 64)`.
//! - `Q4b32`: ggml Q4_0 nibbles, `u = q + 8` with `q ∈ [−8, 7]`; byte `j` of a block holds element
//!   `j` in its low nibble and element `j + 16` in its high nibble; value = `d · (u − 8)`.
//! - `T2b128`: ternary weights, two-bit codes `u = q + 1` with `q ∈ {−1, 0, 1}` and one scale per
//!   128 values; byte `j` of a block of 32 (8 bytes) holds elements `j`, `j + 8`, `j + 16` and
//!   `j + 24` in bits 0–1, 2–3, 4–5 and 6–7; value = `d · (u − 1)`.

pub const BLOCK: usize = 32;
/// Values sharing one scale in `T2b128`.
pub const TERNARY_GROUP: usize = 128;

const Q7_MAX: f32 = 63.0;
const NIBBLE_MASK: u8 = 0x0f;
const NIBBLE_BITS: u8 = 4;
const TERNARY_BITS: usize = 2;
const TERNARY_MASK: u8 = 0b11;
/// Bytes of one `T2b128` block of 32 values.
const TERNARY_BYTES: usize = BLOCK / 4;

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
        match self {
            Self::Q7b32 => count,
            Self::Q4b32 => count / 2,
            Self::T2b128 => count / 4,
        }
    }

    /// Values sharing one scale.
    pub(crate) fn group(self) -> usize {
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

    /// Encodes one group of `values` into `out` (`quant_bytes(group)` bytes); returns its scale.
    /// Quantization rounds ties to even.
    pub(crate) fn quantize(self, values: &[f32], out: &mut [u8]) -> f32 {
        match self {
            Self::Q7b32 => quantize_q7(values, out),
            Self::Q4b32 => quantize_q4(values, out),
            Self::T2b128 => quantize_t2(values, out),
        }
    }

    /// `out = d · (u − zero_point)` for one block of 32 stored values at scale `scale`.
    pub(crate) fn dequantize_block(self, stored: &[u8], scale: f32, out: &mut [f32]) {
        let codes = unpack_block(self, stored);
        for (output, code) in out.iter_mut().zip(codes) {
            *output = scale * (i32::from(code) - self.zero_point()) as f32;
        }
    }
}

/// The 32 unsigned codes of one stored block, in element order.
pub(crate) fn unpack_block(format: QuantFormat, stored: &[u8]) -> [u8; BLOCK] {
    std::array::from_fn(|j| match format {
        QuantFormat::Q7b32 => stored[j],
        QuantFormat::Q4b32 if j < BLOCK / 2 => stored[j] & NIBBLE_MASK,
        QuantFormat::Q4b32 => stored[j - BLOCK / 2] >> NIBBLE_BITS,
        QuantFormat::T2b128 => {
            let shift = TERNARY_BITS * (j / TERNARY_BYTES);
            (stored[j % TERNARY_BYTES] >> shift) & TERNARY_MASK
        }
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
fn quantize_q7(block: &[f32], out: &mut [u8]) -> f32 {
    let scale = group_scale(absmax(block), Q7_MAX);
    let zero_point = QuantFormat::Q7b32.zero_point();
    for (stored, x) in out.iter_mut().zip(block) {
        let level = (x / scale).round_ties_even().clamp(-Q7_MAX, Q7_MAX) as i32;
        *stored = (level + zero_point) as u8;
    }
    scale
}

/// ggml Q4_0: `d = signed extreme / −8`, `u = clamp(round(x / d) + 8, 0, 15)`.
fn quantize_q4(block: &[f32], out: &mut [u8]) -> f32 {
    let extreme = block.iter().fold(
        0.0f32,
        |extreme, &x| {
            if x.abs() > extreme.abs() { x } else { extreme }
        },
    );
    let zero_point = QuantFormat::Q4b32.zero_point() as f32;
    let scale = group_scale(extreme, -zero_point);
    let largest = f32::from(NIBBLE_MASK);
    let nibble = |x: f32| ((x / scale).round_ties_even() + zero_point).clamp(0.0, largest) as u8;
    for j in 0..BLOCK / 2 {
        out[j] = nibble(block[j]) | (nibble(block[j + BLOCK / 2]) << NIBBLE_BITS);
    }
    scale
}

/// Ternary: `d = absmax`, `q = round_ties_even(x / d)` clamped to `[−1, 1]`.
fn quantize_t2(group: &[f32], out: &mut [u8]) -> f32 {
    let zero_point = QuantFormat::T2b128.zero_point();
    let level = zero_point as f32;
    let scale = group_scale(absmax(group), level);
    out.fill(0);
    let blocks = group.as_chunks::<BLOCK>().0;
    for (block, stored) in blocks.iter().zip(out.as_chunks_mut::<TERNARY_BYTES>().0) {
        for (j, x) in block.iter().enumerate() {
            let code = (x / scale).round_ties_even().clamp(-level, level) as i32 + zero_point;
            stored[j % TERNARY_BYTES] |= (code as u8) << (TERNARY_BITS * (j / TERNARY_BYTES));
        }
    }
    scale
}
