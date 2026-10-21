#![expect(unsafe_code, reason = "NEON intrinsics and loads through raw pointers")]

use std::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint8x8_t, vabsq_f32, vaddq_s32, vaddvq_f32, vaddvq_s32,
    vand_u8, vandq_u8, vcombine_s8, vcombine_s16, vcombine_u8, vcvtq_f32_s32, vcvtq_s32_f32,
    vdivq_f32, vdotq_s32, vdup_n_u8, vdupq_n_f32, vdupq_n_s32, vdupq_n_u8, vfmaq_f32, vfmaq_n_f32,
    vld1_u8, vld1q_f32, vld1q_s8, vld1q_u8, vmaxq_f32, vmaxvq_f32, vqmovn_s16, vqmovn_s32,
    vreinterpretq_s8_u8, vrndnq_f32, vsetq_lane_s32, vshr_n_u8, vshrq_n_u8, vst1q_s8,
};

pub(crate) struct Weights(int8x16_t, int8x16_t);
pub(crate) struct Activations(int8x16_t, int8x16_t);

pub(crate) struct Neon;

impl super::Backend for Neon {
    type Weights = Weights;
    type Activations = Activations;
    type Dots = int32x4_t;
    type Sums = float32x4_t;
    const TILE: usize = 6;

    /// # Safety
    /// 32 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_q7(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 32 readable bytes; NEON loads are unaligned. Stored values
        // are ≤ 127, so reading them as i8 keeps them.
        unsafe { Weights(vld1q_s8(at.cast()), vld1q_s8(at.add(16).cast())) }
    }

    /// # Safety
    /// 16 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_q4(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 16 readable bytes; NEON loads are unaligned.
        unsafe {
            let packed = vld1q_u8(at);
            Weights(
                vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(packed)),
            )
        }
    }

    /// # Safety
    /// 8 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_t2(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 8 readable bytes; NEON loads are unaligned.
        unsafe {
            let packed = vld1_u8(at);
            let codes = |shifted: uint8x8_t| vand_u8(shifted, vdup_n_u8(0b11));
            let halves = |low: uint8x8_t, high: uint8x8_t| {
                vreinterpretq_s8_u8(vcombine_u8(codes(low), codes(high)))
            };
            Weights(
                halves(packed, vshr_n_u8::<2>(packed)),
                halves(vshr_n_u8::<4>(packed), vshr_n_u8::<6>(packed)),
            )
        }
    }

    /// # Safety
    /// 32 readable bytes at `at`.
    #[inline(always)]
    unsafe fn load_activations(at: *const i8) -> Self::Activations {
        // SAFETY: the caller guarantees 32 readable bytes; NEON loads are unaligned.
        unsafe { Activations(vld1q_s8(at), vld1q_s8(at.add(16))) }
    }

    #[inline(always)]
    fn zero() -> float32x4_t {
        // SAFETY: NEON is always available on aarch64.
        unsafe { vdupq_n_f32(0.0) }
    }

    fn absmax(x: &[f32]) -> f32 {
        assert!(x.len() >= 32, "one block");
        // SAFETY: 32 readable f32s (asserted); NEON loads are unaligned.
        unsafe {
            let lanes: [float32x4_t; 8] = std::array::from_fn(|i| vld1q_f32(x.as_ptr().add(4 * i)));
            let max = lanes.iter().fold(vdupq_n_f32(0.0), |max, &lane| {
                vmaxq_f32(max, vabsq_f32(lane))
            });
            vmaxvq_f32(max)
        }
    }

    /// Quantizes 32 values to int8 at scale `d` (`round_ties_even(x / d)`); returns `Σq`.
    fn quantize_block(x: &[f32], scale: f32, out: &mut [i8]) -> i32 {
        assert!(
            x.len() >= 32 && out.len() >= 32,
            "one block in, one block out"
        );
        // SAFETY: 32 readable f32s and 32 writable bytes (asserted); NEON loads and stores are
        // unaligned.
        unsafe {
            let levels: [int32x4_t; 8] = std::array::from_fn(|i| {
                let lane = vld1q_f32(x.as_ptr().add(4 * i));
                vcvtq_s32_f32(vrndnq_f32(vdivq_f32(lane, vdupq_n_f32(scale))))
            });
            let narrow = |quarter: &[int32x4_t]| {
                vcombine_s8(
                    vqmovn_s16(vcombine_s16(vqmovn_s32(quarter[0]), vqmovn_s32(quarter[1]))),
                    vqmovn_s16(vcombine_s16(vqmovn_s32(quarter[2]), vqmovn_s32(quarter[3]))),
                )
            };
            vst1q_s8(out.as_mut_ptr(), narrow(&levels[..4]));
            vst1q_s8(out.as_mut_ptr().add(16), narrow(&levels[4..]));
            let sums = levels
                .iter()
                .fold(vdupq_n_s32(0), |sums, &level| vaddq_s32(sums, level));
            vaddvq_s32(sums)
        }
    }

    #[inline(always)]
    fn seed(correction: i32) -> int32x4_t {
        // SAFETY: NEON is always available on aarch64; no memory is accessed.
        unsafe { vsetq_lane_s32::<0>(correction, vdupq_n_s32(0)) }
    }

    #[inline(always)]
    fn dot(dots: int32x4_t, activations: &Self::Activations, weights: &Self::Weights) -> int32x4_t {
        // SAFETY: `dotprod` is enabled for this module (cfg); no memory is accessed.
        unsafe {
            let low = vdotq_s32(dots, activations.0, weights.0);
            vdotq_s32(low, activations.1, weights.1)
        }
    }

    #[inline(always)]
    fn scale(sums: float32x4_t, dots: int32x4_t, scale: f32) -> float32x4_t {
        // SAFETY: NEON is always available on aarch64; no memory is accessed.
        unsafe { vfmaq_n_f32(sums, vcvtq_f32_s32(dots), scale) }
    }

    #[inline(always)]
    fn sum(sums: float32x4_t) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { vaddvq_f32(sums) }
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len(), "dot product of equal lengths");
        let chunks = left.len() / 4;
        // SAFETY: i·4 + 4 ≤ len for both slices (asserted equal); NEON loads are unaligned.
        let head = unsafe {
            let mut sums = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let x = vld1q_f32(left.as_ptr().add(i * 4));
                let y = vld1q_f32(right.as_ptr().add(i * 4));
                sums = vfmaq_f32(sums, x, y);
            }
            vaddvq_f32(sums)
        };
        let tail = left[chunks * 4..].iter().zip(&right[chunks * 4..]);
        head + tail.map(|(x, y)| x * y).sum::<f32>()
    }

    fn dot_f32_4(left: &[f32], rights: [&[f32]; 4]) -> [f32; 4] {
        assert!(
            rights.iter().all(|right| right.len() == left.len()),
            "dot products of equal lengths"
        );
        let chunks = left.len() / 4;
        // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); NEON loads are unaligned.
        let heads = unsafe {
            let mut sums = [vdupq_n_f32(0.0); 4];
            for i in 0..chunks {
                let x = vld1q_f32(left.as_ptr().add(i * 4));
                for (sum, right) in sums.iter_mut().zip(rights) {
                    *sum = vfmaq_f32(*sum, x, vld1q_f32(right.as_ptr().add(i * 4)));
                }
            }
            sums.map(|sum| vaddvq_f32(sum))
        };
        let mut out = [0.0; 4];
        for ((value, head), right) in out.iter_mut().zip(heads).zip(rights) {
            let tail = left[chunks * 4..].iter().zip(&right[chunks * 4..]);
            *value = head + tail.map(|(x, y)| x * y).sum::<f32>();
        }
        out
    }
}
