#![expect(unsafe_code, reason = "SIMD loads and stores through raw pointers")]

use core::arch::wasm32::{
    f32x4, f32x4_abs, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_max,
    f32x4_min, f32x4_mul, f32x4_nearest, f32x4_splat, f32x4_sub, i8x16_narrow_i16x8,
    i16x8_narrow_i32x4, i32x4_add, i32x4_extract_lane, i32x4_shl, i32x4_splat,
    i32x4_trunc_sat_f32x4, u8x16_shr, u8x16_splat, v128, v128_and, v128_load, v128_load32_splat,
    v128_store,
};
#[cfg(target_feature = "relaxed-simd")]
use core::arch::wasm32::{f32x4_relaxed_madd, i32x4_relaxed_dot_i8x16_i7x16_add};
#[cfg(not(target_feature = "relaxed-simd"))]
use core::arch::wasm32::{
    i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_extadd_pairwise_i16x8, i32x4_shuffle,
};

use super::{CHUNKS, DOTS, EXP_RUN, LANES};
use crate::ops::{
    EXP_BIAS, EXP_MAX, EXP_MIN, EXP_POLYNOMIAL, GELU_TANH_CUBIC, GELU_TANH_SCALE, LN2_HIGH, LN2_LOW,
};

pub(crate) struct Wasm;

impl super::Backend for Wasm {
    type Chunk = v128;
    type Dots = v128;
    type Sums = v128;
    const TILE: usize = 4;

    #[inline(always)]
    unsafe fn load_q7(at: *const u8) -> [v128; CHUNKS] {
        // SAFETY: the caller guarantees 128 readable bytes; v128.load is unaligned.
        std::array::from_fn(|chunk| unsafe { v128_load(at.add(16 * chunk).cast()) })
    }

    #[inline(always)]
    unsafe fn load_q4(at: *const u8) -> [v128; CHUNKS] {
        // SAFETY: the caller guarantees 64 readable bytes; v128.load is unaligned.
        let words: [v128; CHUNKS / 2] =
            std::array::from_fn(|word| unsafe { v128_load(at.add(16 * word).cast()) });
        std::array::from_fn(|chunk| {
            if chunk < CHUNKS / 2 {
                v128_and(words[chunk], u8x16_splat(0x0f))
            } else {
                u8x16_shr(words[chunk - CHUNKS / 2], 4)
            }
        })
    }

    #[inline(always)]
    unsafe fn load_t2(at: *const u8) -> [v128; CHUNKS] {
        // SAFETY: the caller guarantees 32 readable bytes; v128.load is unaligned.
        let words: [v128; CHUNKS / 4] =
            std::array::from_fn(|word| unsafe { v128_load(at.add(16 * word).cast()) });
        std::array::from_fn(|chunk| {
            let shift = 2 * (chunk / words.len()) as u32;
            v128_and(
                u8x16_shr(words[chunk % words.len()], shift),
                u8x16_splat(0b11),
            )
        })
    }

    #[inline(always)]
    unsafe fn broadcast(at: *const i8) -> v128 {
        // SAFETY: the caller guarantees 4 readable bytes; v128.load32_splat is unaligned.
        unsafe { v128_load32_splat(at.cast()) }
    }

    #[inline(always)]
    unsafe fn load_scales(at: *const f32) -> v128 {
        // SAFETY: the caller guarantees 4 readable f32s; v128.load is unaligned.
        unsafe { v128_load(at.cast()) }
    }

    #[inline(always)]
    fn zero() -> v128 {
        f32x4_splat(0.0)
    }

    fn absmax(x: &[f32]) -> f32 {
        assert!(x.len() >= 32, "one block");
        // SAFETY: 32 readable f32s (asserted); v128 loads are unaligned.
        let lanes: [v128; 8] =
            unsafe { std::array::from_fn(|i| v128_load(x.as_ptr().add(4 * i).cast())) };
        let max = lanes.iter().fold(f32x4_splat(0.0), |max, &lane| {
            f32x4_max(max, f32x4_abs(lane))
        });
        f32x4_extract_lane::<0>(max)
            .max(f32x4_extract_lane::<1>(max))
            .max(f32x4_extract_lane::<2>(max))
            .max(f32x4_extract_lane::<3>(max))
    }

    fn maximum(x: &[f32]) -> f32 {
        let (chunks, rest) = x.as_chunks::<4>();
        let mut max = f32x4_splat(f32::NEG_INFINITY);
        for chunk in chunks {
            // SAFETY: four readable f32s; v128.load is unaligned.
            max = f32x4_max(max, unsafe { v128_load(chunk.as_ptr().cast()) });
        }
        let lanes = Self::lanes(max);
        rest.iter()
            .chain(&lanes)
            .fold(f32::NEG_INFINITY, |max, &value| max.max(value))
    }

    fn quantize_block(x: &[f32], scale: f32, out: &mut [i8]) -> i32 {
        assert!(
            x.len() >= 32 && out.len() >= 32,
            "one block in, one block out"
        );
        // SAFETY: 32 readable f32s and 32 writable bytes (asserted); v128 loads and stores are
        // unaligned.
        unsafe {
            let levels: [v128; 8] = std::array::from_fn(|i| {
                let lane = v128_load(x.as_ptr().add(4 * i).cast());
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_div(lane, f32x4_splat(scale))))
            });
            let narrow = |quarter: &[v128]| {
                i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4(quarter[0], quarter[1]),
                    i16x8_narrow_i32x4(quarter[2], quarter[3]),
                )
            };
            v128_store(out.as_mut_ptr().cast(), narrow(&levels[..4]));
            v128_store(out.as_mut_ptr().add(16).cast(), narrow(&levels[4..]));
            let sums = levels
                .iter()
                .fold(i32x4_splat(0), |sums, &level| i32x4_add(sums, level));
            i32x4_extract_lane::<0>(sums)
                + i32x4_extract_lane::<1>(sums)
                + i32x4_extract_lane::<2>(sums)
                + i32x4_extract_lane::<3>(sums)
        }
    }

    #[inline(always)]
    fn seed(correction: i32) -> v128 {
        i32x4_splat(correction)
    }

    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)]
    fn dot(dots: v128, activations: v128, weights: v128) -> v128 {
        i32x4_relaxed_dot_i8x16_i7x16_add(activations, weights, dots)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    fn dot(dots: v128, activations: v128, weights: v128) -> v128 {
        // |a·u| ≤ 127 · 127, so two products fit an i16 lane before widening; the pairwise sums
        // then hold each column's four products as lanes 2c and 2c + 1 of (low, high).
        let low = i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(activations, weights));
        let high = i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(activations, weights));
        let columns = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(low, high),
            i32x4_shuffle::<1, 3, 5, 7>(low, high),
        );
        i32x4_add(dots, columns)
    }

    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)]
    fn scale(sums: v128, dots: v128, column_scales: v128, row_scale: f32) -> v128 {
        let scales = f32x4_mul(column_scales, f32x4_splat(row_scale));
        f32x4_relaxed_madd(f32x4_convert_i32x4(dots), scales, sums)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    fn scale(sums: v128, dots: v128, column_scales: v128, row_scale: f32) -> v128 {
        let scales = f32x4_mul(column_scales, f32x4_splat(row_scale));
        f32x4_add(sums, f32x4_mul(f32x4_convert_i32x4(dots), scales))
    }

    #[inline(always)]
    fn lanes(sums: v128) -> [f32; LANES] {
        [
            f32x4_extract_lane::<0>(sums),
            f32x4_extract_lane::<1>(sums),
            f32x4_extract_lane::<2>(sums),
            f32x4_extract_lane::<3>(sums),
        ]
    }

    fn exp_sweep(x: &mut [f32], shift: f32) -> [f32; EXP_RUN] {
        let shift = f32x4_splat(shift);
        let mut sums = [f32x4_splat(0.0); EXP_RUN / 4];
        for run in x.as_chunks_mut::<EXP_RUN>().0 {
            for (half, sum) in run.as_chunks_mut::<4>().0.iter_mut().zip(&mut sums) {
                // SAFETY: four f32s of the run, read then written; v128 loads and stores are
                // unaligned.
                let exps = unsafe {
                    let exps = exp(f32x4_sub(v128_load(half.as_ptr().cast()), shift));
                    v128_store(half.as_mut_ptr().cast(), exps);
                    exps
                };
                *sum = f32x4_add(*sum, exps);
            }
        }
        let [low, high] = sums.map(Self::lanes);
        [
            low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
        ]
    }

    fn gelu_tanh(x: [f32; LANES]) -> [f32; LANES] {
        let x = f32x4(x[0], x[1], x[2], x[3]);
        let cubic = f32x4_mul(f32x4_mul(f32x4_mul(f32x4_splat(GELU_TANH_CUBIC), x), x), x);
        let inner = f32x4_mul(f32x4_splat(GELU_TANH_SCALE), f32x4_add(x, cubic));
        let exps = exp(f32x4_mul(f32x4_splat(-2.0), inner));
        Self::lanes(f32x4_div(x, f32x4_add(f32x4_splat(1.0), exps)))
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len(), "dot product of equal lengths");
        let mut sums = f32x4_splat(0.0);
        let chunks = left.len() / 4;
        for i in 0..chunks {
            // SAFETY: i·4 + 4 ≤ len for both slices (asserted equal); v128.load is unaligned.
            let (x, y) = unsafe {
                (
                    v128_load(left.as_ptr().add(i * 4).cast()),
                    v128_load(right.as_ptr().add(i * 4).cast()),
                )
            };
            sums = f32x4_add(sums, f32x4_mul(x, y));
        }
        let tail = left[chunks * 4..].iter().zip(&right[chunks * 4..]);
        horizontal(sums) + tail.map(|(x, y)| x * y).sum::<f32>()
    }

    fn axpy_f32_many(out: &mut [f32], weights: [f32; DOTS], rights: [&[f32]; DOTS]) {
        assert!(
            rights.iter().all(|right| right.len() == out.len()),
            "rows of the output's length"
        );
        let splats = weights.map(|weight| f32x4_splat(weight));
        let chunks = out.len() / 4;
        for i in 0..chunks {
            // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); v128 loads and stores are
            // unaligned.
            unsafe {
                let at = out.as_mut_ptr().add(i * 4).cast();
                let mut sum = v128_load(at);
                for (weight, right) in splats.iter().zip(rights) {
                    let value = v128_load(right.as_ptr().add(i * 4).cast());
                    sum = f32x4_add(sum, f32x4_mul(*weight, value));
                }
                v128_store(at, sum);
            }
        }
        for i in chunks * 4..out.len() {
            for (weight, right) in weights.iter().zip(rights) {
                out[i] += weight * right[i];
            }
        }
    }

    fn dot_f32_many(left: &[f32], rights: [&[f32]; DOTS]) -> [f32; DOTS] {
        assert!(
            rights.iter().all(|right| right.len() == left.len()),
            "dot products of equal lengths"
        );
        let mut sums = [f32x4_splat(0.0); DOTS];
        let chunks = left.len() / 4;
        for i in 0..chunks {
            // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); v128.load is unaligned.
            let x = unsafe { v128_load(left.as_ptr().add(i * 4).cast()) };
            for (sum, right) in sums.iter_mut().zip(rights) {
                // SAFETY: as above.
                let y = unsafe { v128_load(right.as_ptr().add(i * 4).cast()) };
                *sum = f32x4_add(*sum, f32x4_mul(x, y));
            }
        }
        let mut out = [0.0; DOTS];
        for ((value, sum), right) in out.iter_mut().zip(sums).zip(rights) {
            let tail = left[chunks * 4..].iter().zip(&right[chunks * 4..]);
            *value = horizontal(sum) + tail.map(|(x, y)| x * y).sum::<f32>();
        }
        out
    }
}

fn horizontal(sums: v128) -> f32 {
    f32x4_extract_lane::<0>(sums)
        + f32x4_extract_lane::<1>(sums)
        + f32x4_extract_lane::<2>(sums)
        + f32x4_extract_lane::<3>(sums)
}

/// `ops::exp` on four lanes, operation for operation.
fn exp(x: v128) -> v128 {
    let x = f32x4_max(f32x4_min(x, f32x4_splat(EXP_MAX)), f32x4_splat(EXP_MIN));
    let exponent = f32x4_nearest(f32x4_mul(x, f32x4_splat(std::f32::consts::LOG2_E)));
    let remainder = f32x4_sub(x, f32x4_mul(exponent, f32x4_splat(LN2_HIGH)));
    let remainder = f32x4_sub(remainder, f32x4_mul(exponent, f32x4_splat(LN2_LOW)));
    let mut poly = f32x4_splat(EXP_POLYNOMIAL[0]);
    for &coefficient in &EXP_POLYNOMIAL[1..] {
        poly = f32x4_add(f32x4_mul(poly, remainder), f32x4_splat(coefficient));
    }
    let poly = f32x4_mul(f32x4_mul(poly, remainder), remainder);
    let poly = f32x4_add(f32x4_add(poly, remainder), f32x4_splat(1.0));
    let scale = i32x4_shl(
        i32x4_add(i32x4_trunc_sat_f32x4(exponent), i32x4_splat(EXP_BIAS)),
        23,
    );
    f32x4_mul(poly, scale)
}
