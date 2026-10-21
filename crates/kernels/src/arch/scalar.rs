#![expect(
    unsafe_code,
    reason = "block loads through raw pointers, as on the SIMD targets"
)]

use super::ACTIVATION_MAX;
use crate::quant::{BLOCK, QuantFormat, absmax, unpack_block};

pub(crate) struct Weights([i32; 32]);
pub(crate) struct Activations([i32; 32]);

pub(crate) struct Scalar;

impl super::Backend for Scalar {
    type Weights = Weights;
    type Activations = Activations;
    type Dots = i32;
    type Sums = f32;
    const TILE: usize = 4;

    unsafe fn load_q7(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees a block's bytes.
        unsafe { load_block(QuantFormat::Q7b32, at) }
    }

    unsafe fn load_q4(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees a block's bytes.
        unsafe { load_block(QuantFormat::Q4b32, at) }
    }

    unsafe fn load_t2(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees a block's bytes.
        unsafe { load_block(QuantFormat::T2b128, at) }
    }

    unsafe fn load_activations(at: *const i8) -> Self::Activations {
        // SAFETY: the caller guarantees 32 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, BLOCK) };
        Activations(std::array::from_fn(|j| i32::from(bytes[j])))
    }

    fn zero() -> f32 {
        0.0
    }

    fn absmax(x: &[f32]) -> f32 {
        absmax(&x[..BLOCK])
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

    fn seed(correction: i32) -> i32 {
        correction
    }

    fn dot(dots: i32, activations: &Self::Activations, weights: &Self::Weights) -> i32 {
        let products = activations.0.iter().zip(&weights.0);
        dots + products
            .map(|(activation, weight)| activation * weight)
            .sum::<i32>()
    }

    fn scale(sums: f32, dots: i32, scale: f32) -> f32 {
        sums + dots as f32 * scale
    }

    fn sum(sums: f32) -> f32 {
        sums
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}

/// The codes of one stored block of `format`, widened.
///
/// # Safety
/// `format.quant_bytes(BLOCK)` readable bytes at `at`.
unsafe fn load_block(format: QuantFormat, at: *const u8) -> Weights {
    // SAFETY: forwarded from the caller.
    let stored = unsafe { std::slice::from_raw_parts(at, format.quant_bytes(BLOCK)) };
    Weights(unpack_block(format, stored).map(i32::from))
}
