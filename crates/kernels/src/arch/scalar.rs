#![expect(
    unsafe_code,
    reason = "block loads through raw pointers, as on the SIMD targets"
)]

use super::{ACTIVATION_MAX, EXP_RUN, LANES};
use crate::ops::{exp, exp_fast, gelu_tanh, silu};
use crate::quant::{BLOCK, CHUNK_INPUTS, WORD, absmax};

pub(crate) struct Scalar;

impl super::Backend for Scalar {
    type Chunk = [i32; 16];
    type Dots = [i32; LANES];
    type Sums = [f32; LANES];
    const TILE: usize = 4;

    unsafe fn load_word(at: *const u8) -> Self::Chunk {
        // SAFETY: the caller guarantees 16 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, WORD) };
        std::array::from_fn(|i| i32::from(bytes[i]))
    }

    fn nibbles(word: Self::Chunk, part: usize) -> Self::Chunk {
        word.map(|byte| (byte >> (4 * part)) & 0x0f)
    }

    fn bit_pairs(word: Self::Chunk, part: usize) -> Self::Chunk {
        word.map(|byte| (byte >> (2 * part)) & 0b11)
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
        sweep(x, shift, exp)
    }

    fn exp_sweep_fast(x: &mut [f32], shift: f32) -> [f32; EXP_RUN] {
        sweep(x, shift, exp_fast)
    }

    fn gelu_tanh_sweep(x: &mut [f32]) {
        x.iter_mut().for_each(|value| *value = gelu_tanh(*value));
    }

    fn silu_sweep(x: &mut [f32]) {
        x.iter_mut().for_each(|value| *value = silu(*value));
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}

fn sweep(x: &mut [f32], shift: f32, exponential: impl Fn(f32) -> f32) -> [f32; EXP_RUN] {
    let mut sums = [0.0; EXP_RUN];
    for run in x.as_chunks_mut::<EXP_RUN>().0 {
        for (value, sum) in run.iter_mut().zip(&mut sums) {
            *value = exponential(*value - shift);
            *sum += *value;
        }
    }
    sums
}
