#![expect(unsafe_code, reason = "SIMD loads and stores through raw pointers")]

#[cfg(target_feature = "relaxed-simd")]
use core::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add;
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_max, f32x4_mul,
    f32x4_nearest, f32x4_splat, i8x16_narrow_i16x8, i16x8_narrow_i32x4, i32x4, i32x4_add,
    i32x4_extract_lane, i32x4_splat, i32x4_trunc_sat_f32x4, u8x16_shr, u8x16_splat, v128, v128_and,
    v128_load, v128_store,
};
#[cfg(not(target_feature = "relaxed-simd"))]
use core::arch::wasm32::{
    i16x8_add, i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_extadd_pairwise_i16x8,
};

pub(crate) struct Weights(v128, v128);
pub(crate) struct Activations(v128, v128);

#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn block_dot(activations: &Activations, weights: &Weights, correction: i32) -> v128 {
    let seed = i32x4(correction, 0, 0, 0);
    let low = i32x4_relaxed_dot_i8x16_i7x16_add(activations.0, weights.0, seed);
    i32x4_relaxed_dot_i8x16_i7x16_add(activations.1, weights.1, low)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
fn block_dot(activations: &Activations, weights: &Weights, correction: i32) -> v128 {
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
    i32x4_add(sum, i32x4(correction, 0, 0, 0))
}

pub(crate) struct Wasm;

impl super::Backend for Wasm {
    type Weights = Weights;
    type Activations = Activations;
    type Sums = v128;

    /// # Safety
    /// 32 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_q7(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 32 readable bytes; v128.load is unaligned.
        unsafe { Weights(v128_load(at.cast()), v128_load(at.add(16).cast())) }
    }

    /// # Safety
    /// 16 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_q4(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 16 readable bytes; v128.load is unaligned.
        let packed = unsafe { v128_load(at.cast()) };
        Weights(v128_and(packed, u8x16_splat(0x0f)), u8x16_shr(packed, 4))
    }

    /// # Safety
    /// 32 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_activations(at: *const i8) -> Self::Activations {
        // SAFETY: the caller guarantees 32 readable bytes; v128.load is unaligned.
        unsafe { Activations(v128_load(at.cast()), v128_load(at.add(16).cast())) }
    }

    #[inline(always)]
    fn zero() -> v128 {
        f32x4_splat(0.0)
    }

    /// Quantizes 32 values to int8 (`round_ties_even(x / d)`); returns `(d, Σq)`.
    fn quantize_block(x: &[f32], out: &mut [i8]) -> (f32, i32) {
        assert!(
            x.len() >= 32 && out.len() >= 32,
            "one block in, one block out"
        );
        // SAFETY: 32 readable f32s and 32 writable bytes (asserted); v128 loads and stores are
        // unaligned.
        unsafe {
            let lanes: [v128; 8] = std::array::from_fn(|i| v128_load(x.as_ptr().add(4 * i).cast()));
            let max = lanes.iter().fold(f32x4_splat(0.0), |max, &lane| {
                f32x4_max(max, f32x4_abs(lane))
            });
            let absmax = f32x4_extract_lane::<0>(max)
                .max(f32x4_extract_lane::<1>(max))
                .max(f32x4_extract_lane::<2>(max))
                .max(f32x4_extract_lane::<3>(max));
            let scale = super::activation_scale(absmax);
            let levels: [v128; 8] = std::array::from_fn(|i| {
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_div(lanes[i], f32x4_splat(scale))))
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
            let sum = i32x4_extract_lane::<0>(sums)
                + i32x4_extract_lane::<1>(sums)
                + i32x4_extract_lane::<2>(sums)
                + i32x4_extract_lane::<3>(sums);
            (scale, sum)
        }
    }

    #[inline(always)]
    fn accumulate(
        sums: v128,
        activations: &Self::Activations,
        weights: &Self::Weights,
        correction: i32,
        scale: f32,
    ) -> v128 {
        let dot = f32x4_convert_i32x4(block_dot(activations, weights, correction));
        f32x4_add(sums, f32x4_mul(dot, f32x4_splat(scale)))
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
}
