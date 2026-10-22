#![expect(unsafe_code, reason = "NEON intrinsics and loads through raw pointers")]

use std::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint8x16_t, vabsq_f32, vaddq_f32, vaddq_s32, vaddvq_f32,
    vaddvq_s32, vandq_u8, vcombine_s8, vcombine_s16, vcvtq_f32_s32, vcvtq_s32_f32, vdivq_f32,
    vdotq_s32, vdupq_n_f32, vdupq_n_s32, vdupq_n_u8, vfmaq_f32, vgetq_lane_f32, vld1q_dup_s32,
    vld1q_f32, vld1q_s8, vld1q_u8, vmaxq_f32, vmaxvq_f32, vminq_f32, vmulq_f32, vmulq_n_f32,
    vqmovn_s16, vqmovn_s32, vreinterpretq_f32_s32, vreinterpretq_s8_s32, vreinterpretq_s8_u8,
    vrndnq_f32, vshlq_n_s32, vshrq_n_u8, vst1q_f32, vst1q_s8, vsubq_f32,
};

use super::{CHUNKS, DOTS, EXP_RUN, LANES};
use crate::ops::{
    EXP_BIAS, EXP_MAX, EXP_MIN, EXP_POLYNOMIAL, GELU_TANH_CUBIC, GELU_TANH_SCALE, LN2_HIGH, LN2_LOW,
};

pub(crate) struct Neon;

impl super::Backend for Neon {
    type Chunk = int8x16_t;
    type Dots = int32x4_t;
    type Sums = float32x4_t;
    const TILE: usize = 6;

    #[inline(always)]
    unsafe fn load_q7(at: *const u8) -> [int8x16_t; CHUNKS] {
        // SAFETY: the caller guarantees 128 readable bytes; NEON loads are unaligned. Stored
        // codes are ≤ 127, so reading them as i8 keeps them.
        std::array::from_fn(|chunk| unsafe { vld1q_s8(at.add(16 * chunk).cast()) })
    }

    #[inline(always)]
    unsafe fn load_q4(at: *const u8) -> [int8x16_t; CHUNKS] {
        // SAFETY: the caller guarantees 64 readable bytes; NEON loads are unaligned.
        unsafe {
            let words: [uint8x16_t; CHUNKS / 2] =
                std::array::from_fn(|word| vld1q_u8(at.add(16 * word)));
            std::array::from_fn(|chunk| {
                vreinterpretq_s8_u8(if chunk < CHUNKS / 2 {
                    vandq_u8(words[chunk], vdupq_n_u8(0x0f))
                } else {
                    vshrq_n_u8::<4>(words[chunk - CHUNKS / 2])
                })
            })
        }
    }

    #[inline(always)]
    unsafe fn load_t2(at: *const u8) -> [int8x16_t; CHUNKS] {
        // SAFETY: the caller guarantees 32 readable bytes; NEON loads are unaligned.
        unsafe {
            let (first, second) = (vld1q_u8(at), vld1q_u8(at.add(16)));
            let codes =
                |shifted: uint8x16_t| vreinterpretq_s8_u8(vandq_u8(shifted, vdupq_n_u8(0b11)));
            [
                codes(first),
                codes(second),
                codes(vshrq_n_u8::<2>(first)),
                codes(vshrq_n_u8::<2>(second)),
                codes(vshrq_n_u8::<4>(first)),
                codes(vshrq_n_u8::<4>(second)),
                codes(vshrq_n_u8::<6>(first)),
                codes(vshrq_n_u8::<6>(second)),
            ]
        }
    }

    #[inline(always)]
    unsafe fn broadcast(at: *const i8) -> int8x16_t {
        // SAFETY: the caller guarantees 4 readable bytes; NEON loads are unaligned.
        unsafe { vreinterpretq_s8_s32(vld1q_dup_s32(at.cast())) }
    }

    #[inline(always)]
    unsafe fn load_scales(at: *const f32) -> float32x4_t {
        // SAFETY: the caller guarantees 4 readable f32s; NEON loads are unaligned.
        unsafe { vld1q_f32(at) }
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

    fn maximum(x: &[f32]) -> f32 {
        let (chunks, rest) = x.as_chunks::<4>();
        // SAFETY: four readable f32s per chunk; NEON loads are unaligned.
        let head = unsafe {
            let max = chunks
                .iter()
                .fold(vdupq_n_f32(f32::NEG_INFINITY), |max, chunk| {
                    vmaxq_f32(max, vld1q_f32(chunk.as_ptr()))
                });
            vmaxvq_f32(max)
        };
        rest.iter().fold(head, |max, &value| max.max(value))
    }

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
        unsafe { vdupq_n_s32(correction) }
    }

    #[inline(always)]
    fn dot(dots: int32x4_t, activations: int8x16_t, weights: int8x16_t) -> int32x4_t {
        // SAFETY: `dotprod` is enabled for this module (cfg); no memory is accessed.
        unsafe { vdotq_s32(dots, activations, weights) }
    }

    #[inline(always)]
    fn scale(
        sums: float32x4_t,
        dots: int32x4_t,
        column_scales: float32x4_t,
        row_scale: f32,
    ) -> float32x4_t {
        // SAFETY: NEON is always available on aarch64; no memory is accessed.
        unsafe {
            vfmaq_f32(
                sums,
                vcvtq_f32_s32(dots),
                vmulq_n_f32(column_scales, row_scale),
            )
        }
    }

    #[inline(always)]
    fn lanes(sums: float32x4_t) -> [f32; LANES] {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            [
                vgetq_lane_f32::<0>(sums),
                vgetq_lane_f32::<1>(sums),
                vgetq_lane_f32::<2>(sums),
                vgetq_lane_f32::<3>(sums),
            ]
        }
    }

    fn exp_sweep(x: &mut [f32], shift: f32) -> [f32; EXP_RUN] {
        // SAFETY: whole runs of f32s, read then written; NEON loads and stores are unaligned.
        unsafe {
            let shift = vdupq_n_f32(shift);
            let mut sums = [vdupq_n_f32(0.0); EXP_RUN / 4];
            for run in x.as_chunks_mut::<EXP_RUN>().0 {
                for (half, sum) in run.as_chunks_mut::<4>().0.iter_mut().zip(&mut sums) {
                    let exps = exp(vsubq_f32(vld1q_f32(half.as_ptr()), shift));
                    vst1q_f32(half.as_mut_ptr(), exps);
                    *sum = vaddq_f32(*sum, exps);
                }
            }
            let [low, high] = sums.map(Self::lanes);
            [
                low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
            ]
        }
    }

    fn gelu_tanh(x: [f32; LANES]) -> [f32; LANES] {
        // SAFETY: four readable f32s; NEON loads are unaligned. No fused multiply-add, so the
        // rounding is the scalar's.
        unsafe {
            let x = vld1q_f32(x.as_ptr());
            let cubic = vmulq_f32(vmulq_f32(vmulq_n_f32(x, GELU_TANH_CUBIC), x), x);
            let inner = vmulq_n_f32(vaddq_f32(x, cubic), GELU_TANH_SCALE);
            let exps = exp(vmulq_n_f32(inner, -2.0));
            Self::lanes(vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exps)))
        }
    }

    fn axpy_f32_many(out: &mut [f32], weights: [f32; DOTS], rights: [&[f32]; DOTS]) {
        assert!(
            rights.iter().all(|right| right.len() == out.len()),
            "rows of the output's length"
        );
        let chunks = out.len() / 4;
        for i in 0..chunks {
            // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); NEON loads and stores are
            // unaligned. No fused multiply-add, so the rounding is the scalar's.
            unsafe {
                let at = out.as_mut_ptr().add(i * 4);
                let mut sum = vld1q_f32(at);
                for (weight, right) in weights.iter().zip(rights) {
                    let value = vld1q_f32(right.as_ptr().add(i * 4));
                    sum = vaddq_f32(sum, vmulq_n_f32(value, *weight));
                }
                vst1q_f32(at, sum);
            }
        }
        for i in chunks * 4..out.len() {
            for (weight, right) in weights.iter().zip(rights) {
                out[i] += weight * right[i];
            }
        }
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

    fn dot_f32_many(left: &[f32], rights: [&[f32]; DOTS]) -> [f32; DOTS] {
        assert!(
            rights.iter().all(|right| right.len() == left.len()),
            "dot products of equal lengths"
        );
        let chunks = left.len() / 4;
        // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); NEON loads are unaligned.
        let heads = unsafe {
            let mut sums = [vdupq_n_f32(0.0); DOTS];
            for i in 0..chunks {
                let x = vld1q_f32(left.as_ptr().add(i * 4));
                for (sum, right) in sums.iter_mut().zip(rights) {
                    *sum = vfmaq_f32(*sum, x, vld1q_f32(right.as_ptr().add(i * 4)));
                }
            }
            sums.map(|sum| vaddvq_f32(sum))
        };
        let mut out = [0.0; DOTS];
        for ((value, head), right) in out.iter_mut().zip(heads).zip(rights) {
            let tail = left[chunks * 4..].iter().zip(&right[chunks * 4..]);
            *value = head + tail.map(|(x, y)| x * y).sum::<f32>();
        }
        out
    }
}

/// `ops::exp` on four lanes, operation for operation (no fused multiply-adds).
///
/// # Safety
/// NEON is always available on aarch64; no memory is accessed.
unsafe fn exp(x: float32x4_t) -> float32x4_t {
    // SAFETY: forwarded from the caller.
    unsafe {
        let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(EXP_MAX)), vdupq_n_f32(EXP_MIN));
        let exponent = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
        let remainder = vsubq_f32(x, vmulq_n_f32(exponent, LN2_HIGH));
        let remainder = vsubq_f32(remainder, vmulq_n_f32(exponent, LN2_LOW));
        let mut poly = vdupq_n_f32(EXP_POLYNOMIAL[0]);
        for &coefficient in &EXP_POLYNOMIAL[1..] {
            poly = vaddq_f32(vmulq_f32(poly, remainder), vdupq_n_f32(coefficient));
        }
        let poly = vmulq_f32(vmulq_f32(poly, remainder), remainder);
        let poly = vaddq_f32(vaddq_f32(poly, remainder), vdupq_n_f32(1.0));
        let scale = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(exponent), vdupq_n_s32(EXP_BIAS)));
        vmulq_f32(poly, vreinterpretq_f32_s32(scale))
    }
}
