#![expect(
    unsafe_code,
    reason = "block loads through raw pointers, as on the SIMD targets"
)]

use super::{ACTIVATION_MAX, CHUNKS, EXP_RUN, LANES};
use crate::ops::{exp, gelu_tanh};
use crate::quant::{BLOCK, CHUNK_INPUTS, QuantFormat, absmax, unpack_lane};

pub(crate) struct Scalar;

impl super::Backend for Scalar {
    type Chunk = [i32; 16];
    type Dots = [i32; LANES];
    type Sums = [f32; LANES];
    const TILE: usize = 4;

    unsafe fn load_q7(at: *const u8) -> [Self::Chunk; CHUNKS] {
        // SAFETY: the caller guarantees a quad block's bytes.
        unsafe { load_quad(QuantFormat::Q7b32, at) }
    }

    unsafe fn load_q4(at: *const u8) -> [Self::Chunk; CHUNKS] {
        // SAFETY: the caller guarantees a quad block's bytes.
        unsafe { load_quad(QuantFormat::Q4b32, at) }
    }

    unsafe fn load_t2(at: *const u8) -> [Self::Chunk; CHUNKS] {
        // SAFETY: the caller guarantees a quad block's bytes.
        unsafe { load_quad(QuantFormat::T2b128, at) }
    }

    unsafe fn broadcast(at: *const i8) -> Self::Chunk {
        // SAFETY: the caller guarantees 4 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, CHUNK_INPUTS) };
        std::array::from_fn(|i| i32::from(bytes[i % CHUNK_INPUTS]))
    }

    unsafe fn load_scales(at: *const f32) -> Self::Sums {
        // SAFETY: the caller guarantees 4 readable f32s.
        let scales = unsafe { std::slice::from_raw_parts(at, LANES) };
        std::array::from_fn(|lane| scales[lane])
    }

    fn zero() -> Self::Sums {
        [0.0; LANES]
    }

    fn absmax(x: &[f32]) -> f32 {
        absmax(&x[..BLOCK])
    }

    fn maximum(x: &[f32]) -> f32 {
        x.iter()
            .fold(f32::NEG_INFINITY, |max, &value| max.max(value))
    }

    fn quantize_block(x: &[f32], scale: f32, out: &mut [i8]) -> i32 {
        let mut sum = 0;
        for (output, value) in out[..BLOCK].iter_mut().zip(x) {
            let limit = ACTIVATION_MAX;
            let level = (value / scale).round_ties_even().clamp(-limit, limit) as i8;
            sum += i32::from(level);
            *output = level;
        }
        sum
    }

    fn seed(correction: i32) -> Self::Dots {
        [correction; LANES]
    }

    fn dot(dots: Self::Dots, activations: Self::Chunk, weights: Self::Chunk) -> Self::Dots {
        std::array::from_fn(|lane| {
            let products = (0..CHUNK_INPUTS).map(|input| {
                activations[lane * CHUNK_INPUTS + input] * weights[lane * CHUNK_INPUTS + input]
            });
            dots[lane] + products.sum::<i32>()
        })
    }

    fn scale(
        sums: Self::Sums,
        dots: Self::Dots,
        column_scales: Self::Sums,
        row_scale: f32,
    ) -> Self::Sums {
        std::array::from_fn(|lane| {
            sums[lane] + dots[lane] as f32 * (column_scales[lane] * row_scale)
        })
    }

    fn lanes(sums: Self::Sums) -> [f32; LANES] {
        sums
    }

    fn exp_sweep(x: &mut [f32], shift: f32) -> [f32; EXP_RUN] {
        let mut sums = [0.0; EXP_RUN];
        for run in x.as_chunks_mut::<EXP_RUN>().0 {
            for (value, sum) in run.iter_mut().zip(&mut sums) {
                *value = exp(*value - shift);
                *sum += *value;
            }
        }
        sums
    }

    fn gelu_tanh(x: [f32; LANES]) -> [f32; LANES] {
        x.map(gelu_tanh)
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}

/// The chunks of one stored quad block of `format`, widened: chunk `c` holds inputs
/// `4c..4c + 4` of each of the quad's four rows in turn.
///
/// # Safety
/// `format.quant_bytes(LANES · BLOCK)` readable bytes at `at`.
unsafe fn load_quad(format: QuantFormat, at: *const u8) -> [[i32; 16]; CHUNKS] {
    // SAFETY: forwarded from the caller.
    let stored = unsafe { std::slice::from_raw_parts(at, format.quant_bytes(LANES * BLOCK)) };
    let rows: [[u8; BLOCK]; LANES] = std::array::from_fn(|lane| unpack_lane(format, stored, lane));
    std::array::from_fn(|chunk| {
        std::array::from_fn(|i| {
            i32::from(rows[i / CHUNK_INPUTS][chunk * CHUNK_INPUTS + i % CHUNK_INPUTS])
        })
    })
}
