#![expect(unsafe_code, reason = "SIMD loads and stores through raw pointers")]

#[cfg(target_feature = "relaxed-simd")]
use core::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add;
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_max, f32x4_mul,
    f32x4_nearest, f32x4_splat, i8x16_narrow_i16x8, i16x8_narrow_i32x4, i32x4, i32x4_add,
    i32x4_extract_lane, i32x4_splat, i32x4_trunc_sat_f32x4, i64x2_shuffle, u8x16_shr, u8x16_splat,
    v128, v128_and, v128_load, v128_load64_splat, v128_store,
};
#[cfg(not(target_feature = "relaxed-simd"))]
use core::arch::wasm32::{
    i16x8_add, i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_extadd_pairwise_i16x8,
};

use super::DOTS;

pub(crate) struct Weights(v128, v128);
pub(crate) struct Activations(v128, v128);

pub(crate) struct Wasm;

impl super::Backend for Wasm {
    type Weights = Weights;
    type Activations = Activations;
    type Dots = v128;
    type Sums = v128;
    const TILE: usize = 4;

    #[inline(always)]
    unsafe fn load_q7(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 32 readable bytes; v128.load is unaligned.
        unsafe { Weights(v128_load(at.cast()), v128_load(at.add(16).cast())) }
    }

    #[inline(always)]
    unsafe fn load_q4(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 16 readable bytes; v128.load is unaligned.
        let packed = unsafe { v128_load(at.cast()) };
        Weights(v128_and(packed, u8x16_splat(0x0f)), u8x16_shr(packed, 4))
    }

    #[inline(always)]
    unsafe fn load_t2(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 8 readable bytes; v128.load64_splat is unaligned.
        let packed = unsafe { v128_load64_splat(at.cast()) };
        // Each half takes the low 8 bytes of one shifted copy: elements 0–7 and 8–15, then 16–23
        // and 24–31.
        let halves =
            |low: v128, high: v128| v128_and(i64x2_shuffle::<0, 2>(low, high), u8x16_splat(0b11));
        Weights(
            halves(packed, u8x16_shr(packed, 2)),
            halves(u8x16_shr(packed, 4), u8x16_shr(packed, 6)),
        )
    }

    #[inline(always)]
    unsafe fn load_activations(at: *const i8) -> Self::Activations {
        // SAFETY: the caller guarantees 32 readable bytes; v128.load is unaligned.
        unsafe { Activations(v128_load(at.cast()), v128_load(at.add(16).cast())) }
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
        i32x4(correction, 0, 0, 0)
    }

    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)]
    fn dot(dots: v128, activations: &Self::Activations, weights: &Self::Weights) -> v128 {
        let low = i32x4_relaxed_dot_i8x16_i7x16_add(activations.0, weights.0, dots);
        i32x4_relaxed_dot_i8x16_i7x16_add(activations.1, weights.1, low)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    fn dot(dots: v128, activations: &Self::Activations, weights: &Self::Weights) -> v128 {
        // |a·u| ≤ 127 · 127, so two products fit an i16 lane before widening.
        let half = |activation: v128, weight: v128| {
            let pairs = i16x8_add(
                i16x8_extmul_low_i8x16(activation, weight),
                i16x8_extmul_high_i8x16(activation, weight),
            );
            i32x4_extadd_pairwise_i16x8(pairs)
        };
        let sum = i32x4_add(
            half(activations.0, weights.0),
            half(activations.1, weights.1),
        );
        i32x4_add(dots, sum)
    }

    #[inline(always)]
    fn scale(sums: v128, dots: v128, scale: f32) -> v128 {
        f32x4_add(
            sums,
            f32x4_mul(f32x4_convert_i32x4(dots), f32x4_splat(scale)),
        )
    }

    #[inline(always)]
    fn sum(sums: v128) -> f32 {
        f32x4_extract_lane::<0>(sums)
            + f32x4_extract_lane::<1>(sums)
            + f32x4_extract_lane::<2>(sums)
            + f32x4_extract_lane::<3>(sums)
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
        Self::sum(sums) + tail.map(|(x, y)| x * y).sum::<f32>()
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
            *value = Self::sum(sum) + tail.map(|(x, y)| x * y).sum::<f32>();
        }
        out
    }
}
