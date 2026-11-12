#![expect(unsafe_code, reason = "SIMD loads and stores through raw pointers")]

use std::hint::black_box;

use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_max, f32x4_min,
    f32x4_mul, f32x4_nearest, f32x4_neg, f32x4_splat, f32x4_sub, i8x16_shuffle, i16x8_shuffle,
    i32x4_add, i32x4_extract_lane, i32x4_shl, i32x4_shuffle, i32x4_splat, i32x4_trunc_sat_f32x4,
    u8x16_shr, u8x16_splat, v128, v128_and, v128_load, v128_load32_splat, v128_store,
};
#[cfg(target_feature = "relaxed-simd")]
use core::arch::wasm32::{f32x4_relaxed_madd, i32x4_relaxed_dot_i8x16_i7x16_add};
#[cfg(not(target_feature = "relaxed-simd"))]
use core::arch::wasm32::{
    i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_extadd_pairwise_i16x8,
};

use super::{Backend, DOTS, EXP_RUN, LANES};
use crate::ops::{
    EXP_MAX, EXP_MIN, EXP_POLYNOMIAL, EXP_TAYLOR, GELU_TANH_CUBIC, GELU_TANH_SCALE, LN2_HIGH,
    LN2_LOW, gelu_tanh, silu,
};
use crate::quant::BLOCK;

pub(crate) struct Wasm;

/// `1.5 · 2²³`: f32s from `2²³` to `2²⁴` are the integers, so adding it to a value within
/// `±2²²` rounds the value to its nearest integer, ties to even, and leaves that integer in
/// the sum's low bits.
const ROUNDING: f32 = 12_582_912.0;

impl super::Backend for Wasm {
    type Chunk = v128;
    type Dots = v128;
    type Sums = v128;
    const TILE: usize = 4;

    #[inline(always)]
    unsafe fn load_word(at: *const u8) -> v128 {
        // SAFETY: the caller guarantees 16 readable bytes; v128.load is unaligned.
        unsafe { v128_load(at.cast()) }
    }

    #[inline(always)]
    fn nibbles(word: v128, part: usize) -> v128 {
        if part == 0 {
            v128_and(word, u8x16_splat(0x0f))
        } else {
            u8x16_shr(word, 4)
        }
    }

    #[inline(always)]
    fn bit_pairs(word: v128, part: usize) -> v128 {
        v128_and(u8x16_shr(word, 2 * part as u32), u8x16_splat(0b11))
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
        // Reduced as vectors: `f32::max` skips NaNs, a compare and a select per operand.
        let pairs = f32x4_max(max, i32x4_shuffle::<2, 3, 2, 3>(max, max));
        let all = f32x4_max(pairs, i32x4_shuffle::<1, 1, 1, 1>(pairs, pairs));
        f32x4_extract_lane::<0>(all)
    }

    fn maximum(x: &[f32]) -> f32 {
        // Four maxima in flight: one would wait for itself at every load.
        let (runs, rest) = x.as_chunks::<16>();
        let mut maxima = [f32x4_splat(f32::NEG_INFINITY); 4];
        for run in runs {
            for (max, quad) in maxima.iter_mut().zip(run.as_chunks::<4>().0) {
                // SAFETY: four readable f32s; v128.load is unaligned.
                *max = f32x4_max(*max, unsafe { v128_load(quad.as_ptr().cast()) });
            }
        }
        let [first, second, third, fourth] = maxima;
        let lanes = Self::lanes(f32x4_max(
            f32x4_max(first, second),
            f32x4_max(third, fourth),
        ));
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
            // A quotient plus `ROUNDING` is `ROUNDING` plus the quotient's nearest integer, ties
            // to even (floats are a unit apart there): its bits are `ROUNDING`'s plus the level,
            // one addition where rounding and converting are two.
            let levels: [v128; 8] = std::array::from_fn(|i| {
                let lane = v128_load(x.as_ptr().add(4 * i).cast());
                f32x4_add(f32x4_div(lane, f32x4_splat(scale)), f32x4_splat(ROUNDING))
            });
            // A level is within ±127 (`scale` is at least the block's largest magnitude over
            // 127) and `ROUNDING`'s low byte is zero, so the low byte of the bits is the level:
            // the even bytes of the even halves. Saturating narrows compile to two clamps per
            // level before the same shuffles.
            let narrow = |quarter: &[v128]| {
                i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(
                    i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(quarter[0], quarter[1]),
                    i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(quarter[2], quarter[3]),
                )
            };
            v128_store(out.as_mut_ptr().cast(), narrow(&levels[..4]));
            v128_store(out.as_mut_ptr().add(16).cast(), narrow(&levels[4..]));
            let sums = levels
                .iter()
                .fold(i32x4_splat(0), |sums, &level| i32x4_add(sums, level));
            let bits = i32x4_extract_lane::<0>(sums)
                .wrapping_add(i32x4_extract_lane::<1>(sums))
                .wrapping_add(i32x4_extract_lane::<2>(sums))
                .wrapping_add(i32x4_extract_lane::<3>(sums));
            bits.wrapping_sub((ROUNDING.to_bits() as i32).wrapping_mul(BLOCK as i32))
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
        let exponential = Exp::new();
        sweep(x, shift, |lanes| exponential.of(lanes))
    }

    fn exp_sweep_fast(x: &mut [f32], shift: f32) -> [f32; EXP_RUN] {
        let exponential = FastExp::new();
        sweep(x, shift, |lanes| exponential.of(lanes))
    }

    fn gelu_tanh_sweep(x: &mut [f32]) {
        let exponential = Exp::new();
        let scalars = black_box([GELU_TANH_CUBIC, GELU_TANH_SCALE, -2.0]);
        let [cubic, scale, minus_two] = scalars.map(|value| f32x4_splat(value));
        let gelu = |x: v128| {
            let cube = f32x4_mul(f32x4_mul(f32x4_mul(cubic, x), x), x);
            let inner = f32x4_mul(scale, f32x4_add(x, cube));
            let exps = exponential.of(f32x4_mul(minus_two, inner));
            f32x4_div(x, f32x4_add(exponential.one, exps))
        };
        activate(x, gelu, gelu_tanh);
    }

    fn silu_sweep(x: &mut [f32]) {
        let exponential = Exp::new();
        let silu_lanes = |x: v128| {
            let exps = exponential.of(f32x4_neg(x));
            f32x4_div(x, f32x4_add(exponential.one, exps))
        };
        activate(x, silu_lanes, silu);
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

    fn decay_rows(mut rows: [&mut [f32]; DOTS], decay: f32, deltas: [f32; DOTS], key: &[f32]) {
        assert!(
            rows.iter().all(|row| row.len() == key.len()),
            "rows of the key's length"
        );
        let decay_lanes = f32x4_splat(decay);
        let delta_lanes = deltas.map(|delta| f32x4_splat(delta));
        let chunks = key.len() / 4;
        for i in 0..chunks {
            // SAFETY: i·4 + 4 ≤ len for every slice (asserted equal); v128 loads and stores are
            // unaligned.
            unsafe {
                let keys = v128_load(key.as_ptr().add(i * 4).cast());
                for (row, delta) in rows.iter_mut().zip(delta_lanes) {
                    let at = row.as_mut_ptr().add(i * 4).cast();
                    let decayed = f32x4_mul(decay_lanes, v128_load(at));
                    v128_store(at, f32x4_add(decayed, f32x4_mul(delta, keys)));
                }
            }
        }
        for (row, delta) in rows.into_iter().zip(deltas) {
            for (value, key) in row[chunks * 4..].iter_mut().zip(&key[chunks * 4..]) {
                *value = decay * *value + delta * key;
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

fn sweep(x: &mut [f32], shift: f32, exponential: impl Fn(v128) -> v128) -> [f32; EXP_RUN] {
    let shift = f32x4_splat(shift);
    let mut sums = [f32x4_splat(0.0); EXP_RUN / 4];
    let (pairs, tail) = x.as_chunks_mut::<{ 2 * EXP_RUN }>();
    for pair in pairs {
        let [a_low, a_high, b_low, b_high] = exp_quads(pair, shift, &exponential);
        sums[0] = f32x4_add(sums[0], a_low);
        sums[1] = f32x4_add(sums[1], a_high);
        sums[0] = f32x4_add(sums[0], b_low);
        sums[1] = f32x4_add(sums[1], b_high);
    }
    if let Some(run) = tail.first_chunk_mut::<EXP_RUN>() {
        let [low, high] = exp_quads(run, shift, &exponential);
        sums[0] = f32x4_add(sums[0], low);
        sums[1] = f32x4_add(sums[1], high);
    }
    let [low, high] = sums.map(Wasm::lanes);
    [
        low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
    ]
}

fn exp_quads<const QUADS: usize>(
    x: &mut [f32],
    shift: v128,
    exponential: impl Fn(v128) -> v128,
) -> [v128; QUADS] {
    let quads = x.as_chunks_mut::<4>().0;
    std::array::from_fn(|index| {
        let quad = &mut quads[index];
        // SAFETY: the four f32s of `quad`, read then written; v128 loads and stores are
        // unaligned.
        unsafe {
            let exps = exponential(f32x4_sub(v128_load(quad.as_ptr().cast()), shift));
            v128_store(quad.as_mut_ptr().cast(), exps);
            exps
        }
    })
}

/// `x[i] = scalar(x[i])`, four at a time through `lanes`, which is `scalar` on every lane.
fn activate(x: &mut [f32], lanes: impl Fn(v128) -> v128, scalar: fn(f32) -> f32) {
    let (quads, rest) = x.as_chunks_mut::<LANES>();
    for quad in quads {
        // SAFETY: the four f32s of `quad`, read then written; v128 loads and stores are
        // unaligned.
        unsafe {
            let values = v128_load(quad.as_ptr().cast());
            v128_store(quad.as_mut_ptr().cast(), lanes(values));
        }
    }
    for value in rest {
        *value = scalar(*value);
    }
}

/// `ops::exp`'s constants on four lanes, as values the compiler cannot see: V8 keeps a value
/// in a register across a loop, and builds a constant again where it is used.
struct Exp {
    max: v128,
    min: v128,
    log2_e: v128,
    ln2_high: v128,
    ln2_low: v128,
    polynomial: [v128; EXP_POLYNOMIAL.len()],
    one: v128,
}

impl Exp {
    fn new() -> Self {
        let scalars = [
            EXP_MAX,
            EXP_MIN,
            std::f32::consts::LOG2_E,
            LN2_HIGH,
            LN2_LOW,
            1.0,
        ];
        let [max, min, log2_e, ln2_high, ln2_low, one] =
            black_box(scalars).map(|value| f32x4_splat(value));
        Self {
            max,
            min,
            log2_e,
            ln2_high,
            ln2_low,
            polynomial: black_box(EXP_POLYNOMIAL).map(|value| f32x4_splat(value)),
            one,
        }
    }

    /// `ops::exp` of each lane: its operations in its order, the last product as the
    /// addition of the exponents.
    #[inline(always)]
    fn of(&self, x: v128) -> v128 {
        let x = f32x4_max(f32x4_min(x, self.max), self.min);
        let exponent = f32x4_nearest(f32x4_mul(x, self.log2_e));
        let remainder = f32x4_sub(x, f32x4_mul(exponent, self.ln2_high));
        let remainder = f32x4_sub(remainder, f32x4_mul(exponent, self.ln2_low));
        let mut poly = self.polynomial[0];
        for &coefficient in &self.polynomial[1..] {
            poly = f32x4_add(f32x4_mul(poly, remainder), coefficient);
        }
        let poly = f32x4_mul(f32x4_mul(poly, remainder), remainder);
        let poly = f32x4_add(f32x4_add(poly, remainder), self.one);
        times_power_of_two(poly, exponent)
    }
}

/// `ops::exp_fast`'s constants on four lanes, as values the compiler cannot see: V8 keeps a
/// value in a register across a loop, and builds a constant again where it is used (seven
/// times an iteration of the sweep).
struct FastExp {
    min: v128,
    log2_e: v128,
    minus_ln_2: v128,
    taylor: [v128; EXP_TAYLOR.len()],
}

impl FastExp {
    fn new() -> Self {
        let [min, log2_e, minus_ln_2] =
            black_box([EXP_MIN, std::f32::consts::LOG2_E, -std::f32::consts::LN_2])
                .map(|value| f32x4_splat(value));
        Self {
            min,
            log2_e,
            minus_ln_2,
            taylor: black_box(EXP_TAYLOR).map(|value| f32x4_splat(value)),
        }
    }

    /// `ops::exp_fast` of each lane, operation for operation.
    #[inline(always)]
    fn of(&self, x: v128) -> v128 {
        let x = f32x4_max(x, self.min);
        let exponent = f32x4_nearest(f32x4_mul(x, self.log2_e));
        let remainder = multiply_add(exponent, self.minus_ln_2, x);
        let mut poly = self.taylor[0];
        for &coefficient in &self.taylor[1..] {
            poly = multiply_add(poly, remainder, coefficient);
        }
        times_power_of_two(poly, exponent)
    }
}

#[cfg(target_feature = "relaxed-simd")]
fn multiply_add(left: v128, right: v128, addend: v128) -> v128 {
    f32x4_relaxed_madd(left, right, addend)
}

#[cfg(not(target_feature = "relaxed-simd"))]
fn multiply_add(left: v128, right: v128, addend: v128) -> v128 {
    f32x4_add(f32x4_mul(left, right), addend)
}

/// `value · 2^exponent` (the lanes of `exponent` integral) as the addition of `exponent` to
/// `value`'s own: the product's bits where both are normal, as the exponentials' clamps keep
/// them (`e⁻⁸⁷` is above the smallest normal f32 and `e⁸⁸` below the largest).
fn times_power_of_two(value: v128, exponent: v128) -> v128 {
    i32x4_add(value, i32x4_shl(i32x4_trunc_sat_f32x4(exponent), 23))
}
