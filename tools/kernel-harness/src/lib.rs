//! Scratch microbenchmarks for V8 on this machine: tier-up behaviour, the relaxed dot's
//! throughput, and int8 matmul tile variants against a replica of fusor-ai's kernel.
#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;

#[unsafe(no_mangle)]
pub extern "C" fn arena(len: usize) -> *mut u8 {
    Box::leak(vec![0u8; len].into_boxed_slice()).as_mut_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn tier_work(iters: u32, seed: u32) -> u32 {
    let mut state = seed | 1;
    for _ in 0..iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
    }
    state
}

/// `iters` × 8 independent relaxed dots on registers: the instruction's throughput ceiling.
#[unsafe(no_mangle)]
pub extern "C" fn dot_regs(iters: u32, x: i32) -> i32 {
    let a = i32x4_splat(x);
    let w = v128_and(i32x4_splat(x.wrapping_mul(0x1E37_79B9)), u8x16_splat(0x7f));
    let mut acc = [i32x4_splat(0); 8];
    for i in 0..iters {
        let a = i32x4_add(a, i32x4_splat(i as i32));
        for lane in &mut acc {
            *lane = i32x4_relaxed_dot_i8x16_i7x16_add(a, w, *lane);
        }
    }
    let total = acc.iter().fold(i32x4_splat(0), |s, &v| i32x4_add(s, v));
    i32x4_extract_lane::<0>(total)
        + i32x4_extract_lane::<1>(total)
        + i32x4_extract_lane::<2>(total)
        + i32x4_extract_lane::<3>(total)
}

/// Same as `dot_regs` with the i16x8 lowering (SIMD128 path of the kernel) for comparison.
#[unsafe(no_mangle)]
pub extern "C" fn dot_regs_i16(iters: u32, x: i32) -> i32 {
    let a = i32x4_splat(x);
    let w = v128_and(i32x4_splat(x.wrapping_mul(0x1E37_79B9)), u8x16_splat(0x7f));
    let mut acc = [i32x4_splat(0); 8];
    for i in 0..iters {
        let a = i32x4_add(a, i32x4_splat(i as i32));
        for lane in &mut acc {
            let pairs = i16x8_add(i16x8_extmul_low_i8x16(a, w), i16x8_extmul_high_i8x16(a, w));
            *lane = i32x4_add(*lane, i32x4_extadd_pairwise_i16x8(pairs));
        }
    }
    let total = acc.iter().fold(i32x4_splat(0), |s, &v| i32x4_add(s, v));
    i32x4_extract_lane::<0>(total)
}

#[repr(C)]
pub struct Params {
    a: *const i8,
    a_scales: *const f32,
    a_corr: *const i32,
    w: *const u8,
    w_scales: *const f32,
    out: *mut f32,
    m: usize,
    blocks: usize,
    n: usize,
}

trait Weights {
    const BYTES: usize;
    unsafe fn load(at: *const u8) -> (v128, v128);
}
struct Q7;
struct Q4;
impl Weights for Q7 {
    const BYTES: usize = 32;
    #[inline(always)]
    unsafe fn load(at: *const u8) -> (v128, v128) {
        unsafe { (v128_load(at.cast()), v128_load(at.add(16).cast())) }
    }
}
impl Weights for Q4 {
    const BYTES: usize = 16;
    #[inline(always)]
    unsafe fn load(at: *const u8) -> (v128, v128) {
        let packed = unsafe { v128_load(at.cast()) };
        (v128_and(packed, u8x16_splat(0x0f)), u8x16_shr(packed, 4))
    }
}

#[inline(always)]
unsafe fn load_a(at: *const i8) -> (v128, v128) {
    unsafe { (v128_load(at.cast()), v128_load(at.add(16).cast())) }
}

#[inline(always)]
fn dot(acc: v128, a: (v128, v128), w: (v128, v128)) -> v128 {
    let low = i32x4_relaxed_dot_i8x16_i7x16_add(a.0, w.0, acc);
    i32x4_relaxed_dot_i8x16_i7x16_add(a.1, w.1, low)
}

#[inline(always)]
fn hsum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

/// Replica of fusor-ai's `quant_tile`: 4 rows × 2 columns, per-block f32 scaling.
#[inline(never)]
unsafe fn mm_4x2<W: Weights, const MADD: bool, const ROWS: usize>(p: &Params) {
    let (m, blocks, n) = (p.m, p.blocks, p.n);
    let mut tile = 0;
    while tile < m {
        let mut col = 0;
        while col < n {
            let second = col + 1;
            let mut s0 = [f32x4_splat(0.0); ROWS];
            let mut s1 = [f32x4_splat(0.0); ROWS];
            for block in 0..blocks {
                let seeds: [v128; ROWS] = core::array::from_fn(|r| unsafe {
                    i32x4(*p.a_corr.add((tile + r) * blocks + block), 0, 0, 0)
                });
                let (mut d0, mut d1) = (seeds, seeds);
                let w0 = unsafe { W::load(p.w.add((col * blocks + block) * W::BYTES)) };
                let w1 = unsafe { W::load(p.w.add((second * blocks + block) * W::BYTES)) };
                for r in 0..ROWS {
                    let a = unsafe { load_a(p.a.add(((tile + r) * blocks + block) * 32)) };
                    d0[r] = dot(d0[r], a, w0);
                    d1[r] = dot(d1[r], a, w1);
                }
                let ws0 = unsafe { *p.w_scales.add(col * blocks + block) };
                let ws1 = unsafe { *p.w_scales.add(second * blocks + block) };
                for r in 0..ROWS {
                    let s = unsafe { *p.a_scales.add((tile + r) * blocks + block) };
                    if MADD {
                        s0[r] = f32x4_relaxed_madd(f32x4_convert_i32x4(d0[r]), f32x4_splat(ws0 * s), s0[r]);
                        s1[r] = f32x4_relaxed_madd(f32x4_convert_i32x4(d1[r]), f32x4_splat(ws1 * s), s1[r]);
                    } else {
                        s0[r] = f32x4_add(s0[r], f32x4_mul(f32x4_convert_i32x4(d0[r]), f32x4_splat(ws0 * s)));
                        s1[r] = f32x4_add(s1[r], f32x4_mul(f32x4_convert_i32x4(d1[r]), f32x4_splat(ws1 * s)));
                    }
                }
            }
            for r in 0..ROWS {
                unsafe {
                    *p.out.add((tile + r) * n + col) += hsum(s0[r]);
                    *p.out.add((tile + r) * n + second) += hsum(s1[r]);
                }
            }
            col += 2;
        }
        tile += ROWS;
    }
}

/// 4 rows × 4 columns, per-block scaling with relaxed madd.
#[inline(never)]
unsafe fn mm_4x4<W: Weights>(p: &Params) {
    let (m, blocks, n) = (p.m, p.blocks, p.n);
    let mut tile = 0;
    while tile < m {
        let mut col = 0;
        while col < n {
            let mut sums = [[f32x4_splat(0.0); 4]; 4];
            for block in 0..blocks {
                let seeds: [v128; 4] = core::array::from_fn(|r| unsafe {
                    i32x4(*p.a_corr.add((tile + r) * blocks + block), 0, 0, 0)
                });
                let mut dots = [seeds; 4];
                let w: [(v128, v128); 4] = core::array::from_fn(|c| unsafe {
                    W::load(p.w.add(((col + c) * blocks + block) * W::BYTES))
                });
                for r in 0..4 {
                    let a = unsafe { load_a(p.a.add(((tile + r) * blocks + block) * 32)) };
                    for c in 0..4 {
                        dots[c][r] = dot(dots[c][r], a, w[c]);
                    }
                }
                let ws: [v128; 4] = core::array::from_fn(|c| unsafe {
                    f32x4_splat(*p.w_scales.add((col + c) * blocks + block))
                });
                for r in 0..4 {
                    let s = unsafe { f32x4_splat(*p.a_scales.add((tile + r) * blocks + block)) };
                    for c in 0..4 {
                        sums[c][r] = f32x4_relaxed_madd(f32x4_convert_i32x4(dots[c][r]), f32x4_mul(ws[c], s), sums[c][r]);
                    }
                }
            }
            for r in 0..4 {
                for c in 0..4 {
                    unsafe { *p.out.add((tile + r) * n + col + c) += hsum(sums[c][r]) };
                }
            }
            col += 4;
        }
        tile += 4;
    }
}

/// Hypothetical: one activation scale per row (applied at the end), weight scales per block.
#[inline(never)]
unsafe fn mm_4x4_rowscale<W: Weights>(p: &Params) {
    let (m, blocks, n) = (p.m, p.blocks, p.n);
    let mut tile = 0;
    while tile < m {
        let mut col = 0;
        while col < n {
            let mut sums = [[f32x4_splat(0.0); 4]; 4];
            for block in 0..blocks {
                let seeds: [v128; 4] = core::array::from_fn(|r| unsafe {
                    i32x4(*p.a_corr.add((tile + r) * blocks + block), 0, 0, 0)
                });
                let mut dots = [seeds; 4];
                let w: [(v128, v128); 4] = core::array::from_fn(|c| unsafe {
                    W::load(p.w.add(((col + c) * blocks + block) * W::BYTES))
                });
                for r in 0..4 {
                    let a = unsafe { load_a(p.a.add(((tile + r) * blocks + block) * 32)) };
                    for c in 0..4 {
                        dots[c][r] = dot(dots[c][r], a, w[c]);
                    }
                }
                for c in 0..4 {
                    let ws = unsafe { f32x4_splat(*p.w_scales.add((col + c) * blocks + block)) };
                    for r in 0..4 {
                        sums[c][r] = f32x4_relaxed_madd(f32x4_convert_i32x4(dots[c][r]), ws, sums[c][r]);
                    }
                }
            }
            for r in 0..4 {
                let s = unsafe { *p.a_scales.add((tile + r) * blocks) };
                for c in 0..4 {
                    unsafe { *p.out.add((tile + r) * n + col + c) += hsum(sums[c][r]) * s };
                }
            }
            col += 4;
        }
        tile += 4;
    }
}

/// Hypothetical ceiling: integer accumulation over all of K, no per-block scaling at all.
#[inline(never)]
unsafe fn mm_4x4_int<W: Weights>(p: &Params) {
    let (m, blocks, n) = (p.m, p.blocks, p.n);
    let mut tile = 0;
    while tile < m {
        let mut col = 0;
        while col < n {
            let mut dots = [[i32x4_splat(0); 4]; 4];
            for block in 0..blocks {
                let w: [(v128, v128); 4] = core::array::from_fn(|c| unsafe {
                    W::load(p.w.add(((col + c) * blocks + block) * W::BYTES))
                });
                for r in 0..4 {
                    let a = unsafe { load_a(p.a.add(((tile + r) * blocks + block) * 32)) };
                    for c in 0..4 {
                        dots[c][r] = dot(dots[c][r], a, w[c]);
                    }
                }
            }
            for r in 0..4 {
                let s = unsafe { *p.a_scales.add((tile + r) * blocks) };
                for c in 0..4 {
                    let ws = unsafe { *p.w_scales.add((col + c) * blocks) };
                    unsafe { *p.out.add((tile + r) * n + col + c) += hsum(f32x4_convert_i32x4(dots[c][r])) * s * ws };
                }
            }
            col += 4;
        }
        tile += 4;
    }
}

/// Lane-per-column layout: `w` holds, per group of 4 columns and per block, 8 chunks of
/// 16 bytes `[c0 k..k+4, c1 k..k+4, c2 k..k+4, c3 k..k+4]` (Q7), or 4 words whose low
/// nibbles are chunk `c` and high nibbles chunk `c + 4` (Q4); `w_scales` holds `[s0..s3]` per
/// group and block. `ROWS` rows × 4 column groups per tile; the activation chunk is broadcast.
#[inline(never)]
unsafe fn mm_lanes<W: Weights, const ROWS: usize, const GROUPS: usize>(p: &Params) {
    let (m, blocks, n) = (p.m, p.blocks, p.n);
    let group_bytes = 128 / (32 / W::BYTES);
    let mut tile = 0;
    while tile < m {
        let mut col = 0;
        while col < n {
            let mut sums = [[f32x4_splat(0.0); GROUPS]; ROWS];
            for block in 0..blocks {
                let mut dots: [[v128; GROUPS]; ROWS] = core::array::from_fn(|r| {
                    let correction = unsafe { *p.a_corr.add((tile + r) * blocks + block) };
                    [i32x4_splat(correction); GROUPS]
                });
                let chunks: [[v128; 8]; GROUPS] = core::array::from_fn(|g| unsafe {
                    let at = p.w.add(((col / 4 + g) * blocks + block) * group_bytes);
                    if W::BYTES == 32 {
                        core::array::from_fn(|c| v128_load(at.add(16 * c).cast()))
                    } else {
                        let words: [v128; 4] = core::array::from_fn(|c| v128_load(at.add(16 * c).cast()));
                        core::array::from_fn(|c| {
                            if c < 4 { v128_and(words[c], u8x16_splat(0x0f)) } else { u8x16_shr(words[c - 4], 4) }
                        })
                    }
                });
                for r in 0..ROWS {
                    let row = unsafe { p.a.add(((tile + r) * blocks + block) * 32) };
                    for c in 0..8 {
                        let a = unsafe { v128_load32_splat(row.add(4 * c).cast()) };
                        for g in 0..GROUPS {
                            dots[r][g] = i32x4_relaxed_dot_i8x16_i7x16_add(a, chunks[g][c], dots[r][g]);
                        }
                    }
                }
                let col_scales: [v128; GROUPS] = core::array::from_fn(|g| unsafe {
                    v128_load(p.w_scales.add(((col / 4 + g) * blocks + block) * 4).cast())
                });
                for r in 0..ROWS {
                    let s = unsafe { f32x4_splat(*p.a_scales.add((tile + r) * blocks + block)) };
                    for g in 0..GROUPS {
                        sums[r][g] = f32x4_relaxed_madd(f32x4_convert_i32x4(dots[r][g]), f32x4_mul(col_scales[g], s), sums[r][g]);
                    }
                }
            }
            for r in 0..ROWS {
                for g in 0..GROUPS {
                    let at = unsafe { p.out.add((tile + r) * n + col + 4 * g) };
                    unsafe { v128_store(at.cast(), f32x4_add(v128_load(at.cast()), sums[r][g])) };
                }
            }
            col += 4 * GROUPS;
        }
        tile += ROWS;
    }
}

macro_rules! export_mm {
    ($name:ident, $call:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(p: *const Params, repeats: u32) {
            let p = unsafe { &*p };
            for _ in 0..repeats {
                unsafe { $call(p) };
            }
        }
    };
}
export_mm!(mm_4x2_q7, mm_4x2::<Q7, false, 4>);
export_mm!(mm_1x2_q7, mm_4x2::<Q7, false, 1>);
export_mm!(mm_1x2_q4, mm_4x2::<Q4, false, 1>);
export_mm!(mm_4x2_q7_madd, mm_4x2::<Q7, true, 4>);
export_mm!(mm_4x4_q7, mm_4x4::<Q7>);
export_mm!(mm_4x4_q7_rowscale, mm_4x4_rowscale::<Q7>);
export_mm!(mm_4x4_q7_int, mm_4x4_int::<Q7>);
export_mm!(mm_4x2_q4, mm_4x2::<Q4, false, 4>);
export_mm!(mm_4x2_q4_madd, mm_4x2::<Q4, true, 4>);
export_mm!(mm_4x4_q4, mm_4x4::<Q4>);
export_mm!(mm_4x4_q4_rowscale, mm_4x4_rowscale::<Q4>);
export_mm!(mm_4x4_q4_int, mm_4x4_int::<Q4>);
export_mm!(mm_lanes_q7_r2, mm_lanes::<Q7, 2, 4>);
export_mm!(mm_lanes_q7_r4, mm_lanes::<Q7, 4, 4>);
export_mm!(mm_lanes_q4_r2, mm_lanes::<Q4, 2, 4>);
export_mm!(mm_lanes_q4_r4, mm_lanes::<Q4, 4, 4>);

/// fusor-ai's `kernels::ops::exp`, as written there (scalar, auto-vectorized).
#[inline(always)]
fn exp_ref(x: f32) -> f32 {
    const LN2_HIGH: f32 = 0.693_359_4;
    const LN2_LOW: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    let exponent = (x * core::f32::consts::LOG2_E).round_ties_even();
    let remainder = x - exponent * LN2_HIGH - exponent * LN2_LOW;
    let poly = 1.987_569_1e-4;
    let poly = poly * remainder + 1.398_199_9e-3;
    let poly = poly * remainder + 8.333_452e-3;
    let poly = poly * remainder + 4.166_579_6e-2;
    let poly = poly * remainder + 0.166_666_66;
    let poly = poly * remainder + 0.5;
    let poly = poly * remainder * remainder + remainder + 1.0;
    poly * f32::from_bits(((exponent as i32 + 127) << 23) as u32)
}

/// fusor-ai's `attention::exponentials` over `n` scores into `out`, `repeats` times; returns
/// the last sum.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exponentials_ref(scores: *const f32, out: *mut f32, n: usize, repeats: u32) -> f32 {
    const LANES: usize = 8;
    let mut total = 0.0;
    for _ in 0..repeats {
        let scores = unsafe { core::slice::from_raw_parts(scores, n) };
        let out = unsafe { core::slice::from_raw_parts_mut(out, n) };
        let mut maxima = [f32::NEG_INFINITY; LANES];
        let (lanes, rest) = scores.as_chunks::<LANES>();
        for chunk in lanes {
            for (max, score) in maxima.iter_mut().zip(chunk) {
                *max = max.max(*score);
            }
        }
        let max = rest.iter().chain(&maxima).fold(f32::NEG_INFINITY, |max, &score| max.max(score));
        let mut sums = [0.0; LANES];
        let (lanes, rest) = scores.as_chunks::<LANES>();
        let (out_lanes, out_rest) = out.as_chunks_mut::<LANES>();
        for (chunk, out_chunk) in lanes.iter().zip(out_lanes) {
            for ((sum, score), out) in sums.iter_mut().zip(chunk).zip(out_chunk) {
                *out = exp_ref(*score - max);
                *sum += *out;
            }
        }
        for (score, out) in rest.iter().zip(out_rest.iter_mut()) {
            *out = exp_ref(*score - max);
        }
        total = out_rest.iter().chain(&sums).sum::<f32>();
    }
    total
}

/// Hand-vectorized exp with relaxed madd and f32x4.nearest, same algorithm.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exponentials_simd(scores: *const f32, out: *mut f32, n: usize, repeats: u32) -> f32 {
    let mut total = 0.0;
    for _ in 0..repeats {
        let scores = unsafe { core::slice::from_raw_parts(scores, n) };
        let mut max4 = f32x4_splat(f32::NEG_INFINITY);
        let chunks = n / 4;
        for i in 0..chunks {
            max4 = f32x4_max(max4, unsafe { v128_load(scores.as_ptr().add(4 * i).cast()) });
        }
        let max = f32x4_extract_lane::<0>(max4)
            .max(f32x4_extract_lane::<1>(max4))
            .max(f32x4_extract_lane::<2>(max4))
            .max(f32x4_extract_lane::<3>(max4));
        let max4 = f32x4_splat(max);
        let mut sum4 = [f32x4_splat(0.0); 2];
        let log2e = f32x4_splat(core::f32::consts::LOG2_E);
        let ln2_high = f32x4_splat(-0.693_359_4);
        let ln2_low = f32x4_splat(2.121_944_4e-4);
        let c = [
            f32x4_splat(1.987_569_1e-4),
            f32x4_splat(1.398_199_9e-3),
            f32x4_splat(8.333_452e-3),
            f32x4_splat(4.166_579_6e-2),
            f32x4_splat(0.166_666_66),
            f32x4_splat(0.5),
        ];
        let one = f32x4_splat(1.0);
        let bias = i32x4_splat(127);
        for i in 0..chunks {
            let at = unsafe { out.add(4 * i) };
            let x = f32x4_sub(unsafe { v128_load(scores.as_ptr().add(4 * i).cast()) }, max4);
            let x = f32x4_max(x, f32x4_splat(-87.0));
            let exponent = f32x4_nearest(f32x4_mul(x, log2e));
            let r = f32x4_relaxed_madd(exponent, ln2_high, x);
            let r = f32x4_relaxed_madd(exponent, ln2_low, r);
            let mut poly = c[0];
            for k in &c[1..] {
                poly = f32x4_relaxed_madd(poly, r, *k);
            }
            let poly = f32x4_relaxed_madd(f32x4_mul(poly, r), r, f32x4_add(r, one));
            let scale = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(exponent), bias), 23);
            let e = f32x4_mul(poly, scale);
            unsafe { v128_store(at.cast(), e) };
            sum4[i & 1] = f32x4_add(sum4[i & 1], e);
        }
        let sum = f32x4_add(sum4[0], sum4[1]);
        total = hsum(sum);
    }
    total
}

/// Hand-vectorized exp with relaxed madd and f32x4.nearest, same algorithm.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exponentials_simd_scalarmax(scores: *const f32, out: *mut f32, n: usize, repeats: u32) -> f32 {
    let mut total = 0.0;
    for _ in 0..repeats {
        let scores = unsafe { core::slice::from_raw_parts(scores, n) };
        let mut maxima = [f32::NEG_INFINITY; 8];
        let (lanes, rest) = scores.as_chunks::<8>();
        for chunk in lanes {
            for (max, score) in maxima.iter_mut().zip(chunk) {
                *max = max.max(*score);
            }
        }
        let max = rest.iter().chain(&maxima).fold(f32::NEG_INFINITY, |max, &score| max.max(score));
        let chunks = n / 4;
        let max4 = f32x4_splat(max);
        let mut sum4 = [f32x4_splat(0.0); 2];
        let log2e = f32x4_splat(core::f32::consts::LOG2_E);
        let ln2_high = f32x4_splat(-0.693_359_4);
        let ln2_low = f32x4_splat(2.121_944_4e-4);
        let c = [
            f32x4_splat(1.987_569_1e-4),
            f32x4_splat(1.398_199_9e-3),
            f32x4_splat(8.333_452e-3),
            f32x4_splat(4.166_579_6e-2),
            f32x4_splat(0.166_666_66),
            f32x4_splat(0.5),
        ];
        let one = f32x4_splat(1.0);
        let bias = i32x4_splat(127);
        for i in 0..chunks {
            let at = unsafe { out.add(4 * i) };
            let x = f32x4_sub(unsafe { v128_load(scores.as_ptr().add(4 * i).cast()) }, max4);
            let x = f32x4_max(x, f32x4_splat(-87.0));
            let exponent = f32x4_nearest(f32x4_mul(x, log2e));
            let r = f32x4_relaxed_madd(exponent, ln2_high, x);
            let r = f32x4_relaxed_madd(exponent, ln2_low, r);
            let mut poly = c[0];
            for k in &c[1..] {
                poly = f32x4_relaxed_madd(poly, r, *k);
            }
            let poly = f32x4_relaxed_madd(f32x4_mul(poly, r), r, f32x4_add(r, one));
            let scale = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(exponent), bias), 23);
            let e = f32x4_mul(poly, scale);
            unsafe { v128_store(at.cast(), e) };
            sum4[i & 1] = f32x4_add(sum4[i & 1], e);
        }
        let sum = f32x4_add(sum4[0], sum4[1]);
        total = hsum(sum);
    }
    total
}

// ---- language-model attention replicas: one query over `keys` cached keys/values of 128 ----

const HEAD: usize = 128;

#[inline(always)]
fn dot128(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = f32x4_splat(0.0);
    for i in 0..HEAD / 4 {
        let x = unsafe { v128_load(a.as_ptr().add(4 * i).cast()) };
        let y = unsafe { v128_load(b.as_ptr().add(4 * i).cast()) };
        sums = f32x4_add(sums, f32x4_mul(x, y));
    }
    hsum(sums)
}

#[inline(always)]
fn dot128_many(a: &[f32], bs: [&[f32]; 4]) -> [f32; 4] {
    let mut sums = [f32x4_splat(0.0); 4];
    for i in 0..HEAD / 4 {
        let x = unsafe { v128_load(a.as_ptr().add(4 * i).cast()) };
        for (sum, b) in sums.iter_mut().zip(bs) {
            let y = unsafe { v128_load(b.as_ptr().add(4 * i).cast()) };
            *sum = f32x4_add(*sum, f32x4_mul(x, y));
        }
    }
    sums.map(hsum)
}

fn softmax_scalar(scores: &mut [f32]) {
    let max = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
    let mut total = 0.0;
    for s in scores.iter_mut() {
        *s = exp_ref(*s - max);
        total += *s;
    }
    for s in scores.iter_mut() {
        *s /= total;
    }
}

/// As fusor-ai's `attend` after attempt 3: a closure per key with a modulo, 4 dots at a time,
/// scalar softmax, the value accumulation as an auto-vectorized loop over `out`.
/// `kernels::ops::attend` as the language model calls it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn real_attend(q: *const f32, keys: *const f32, values: *const f32, count: usize, capacity: usize, first: usize, _scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, HEAD) };
    let keys = unsafe { core::slice::from_raw_parts(keys, capacity * HEAD) };
    let values = unsafe { core::slice::from_raw_parts(values, capacity * HEAD) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, HEAD) };
    let mut scores = Vec::with_capacity(count);
    for _ in 0..repeats {
        let key_value = |j: usize| {
            let slot = ((first + j) % capacity) * HEAD;
            (&keys[slot..slot + HEAD], &values[slot..slot + HEAD])
        };
        kernels::ops::attend(q, count, key_value, &mut scores, out);
    }
}

fn dot_dyn(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The page's attention: head size a run-time value, `kernels::ops::attend` as is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_dyn_real(q: *const f32, keys: *const f32, values: *const f32, count: usize, capacity: usize, head: usize, _scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, head) };
    let keys = unsafe { core::slice::from_raw_parts(keys, capacity * head) };
    let values = unsafe { core::slice::from_raw_parts(values, capacity * head) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, head) };
    let mut scores = Vec::with_capacity(count);
    for _ in 0..repeats {
        let key_value = |j: usize| {
            let slot = (j % capacity) * head;
            (&keys[slot..slot + head], &values[slot..slot + head])
        };
        kernels::ops::attend(q, count, key_value, &mut scores, out);
    }
}

/// Run-time head size, the new softmax (`kernels::ops::softmax`), the old P·V (one key at a time).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_dyn_newsoftmax_oldpv(q: *const f32, keys: *const f32, values: *const f32, count: usize, capacity: usize, head: usize, scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, head) };
    let keys = unsafe { core::slice::from_raw_parts(keys, capacity * head) };
    let values = unsafe { core::slice::from_raw_parts(values, capacity * head) };
    let scores = unsafe { core::slice::from_raw_parts_mut(scores, count) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, head) };
    for _ in 0..repeats {
        let key_value = |j: usize| {
            let slot = (j % capacity) * head;
            (&keys[slot..slot + head], &values[slot..slot + head])
        };
        let scale = 1.0 / (head as f32).sqrt();
        for j in 0..count {
            scores[j] = dot_dyn(q, key_value(j).0) * scale;
        }
        kernels::ops::softmax(scores);
        out.fill(0.0);
        for (j, weight) in scores.iter().enumerate() {
            for (output, component) in out.iter_mut().zip(key_value(j).1) {
                *output += weight * component;
            }
        }
    }
}

/// Run-time head size, the old scalar softmax, the new P·V (four keys at a time).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_dyn_oldsoftmax_newpv(q: *const f32, keys: *const f32, values: *const f32, count: usize, capacity: usize, head: usize, scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, head) };
    let keys = unsafe { core::slice::from_raw_parts(keys, capacity * head) };
    let values = unsafe { core::slice::from_raw_parts(values, capacity * head) };
    let scores = unsafe { core::slice::from_raw_parts_mut(scores, count) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, head) };
    for _ in 0..repeats {
        let key_value = |j: usize| {
            let slot = (j % capacity) * head;
            (&keys[slot..slot + head], &values[slot..slot + head])
        };
        let scale = 1.0 / (head as f32).sqrt();
        for j in 0..count {
            scores[j] = dot_dyn(q, key_value(j).0) * scale;
        }
        softmax_scalar(scores);
        out.fill(0.0);
        let (blocks, rest) = scores.as_chunks::<4>();
        for (block, weights) in blocks.iter().enumerate() {
            let values: [&[f32]; 4] = core::array::from_fn(|offset| &key_value(block * 4 + offset).1[..out.len()]);
            for (index, output) in out.iter_mut().enumerate() {
                for (weight, value) in weights.iter().zip(&values) {
                    *output += weight * value[index];
                }
            }
        }
        for (j, weight) in (blocks.len() * 4..).zip(rest) {
            for (output, component) in out.iter_mut().zip(key_value(j).1) {
                *output += weight * component;
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_closure(q: *const f32, keys: *const f32, values: *const f32, count: usize, capacity: usize, first: usize, scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, HEAD) };
    let keys = unsafe { core::slice::from_raw_parts(keys, capacity * HEAD) };
    let values = unsafe { core::slice::from_raw_parts(values, capacity * HEAD) };
    let scores = unsafe { core::slice::from_raw_parts_mut(scores, count) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, HEAD) };
    for _ in 0..repeats {
        let key_value = |j: usize| {
            let slot = ((first + j) % capacity) * HEAD;
            (&keys[slot..slot + HEAD], &values[slot..slot + HEAD])
        };
        let scale = 1.0 / (HEAD as f32).sqrt();
        let grouped = count - count % 4;
        for start in (0..grouped).step_by(4) {
            let group = core::array::from_fn(|o| key_value(start + o).0);
            for (o, v) in dot128_many(q, group).into_iter().enumerate() {
                scores[start + o] = v * scale;
            }
        }
        for j in grouped..count {
            scores[j] = dot128(q, key_value(j).0) * scale;
        }
        softmax_scalar(scores);
        out.fill(0.0);
        for (j, weight) in scores.iter().enumerate() {
            for (o, v) in out.iter_mut().zip(key_value(j).1) {
                *o += weight * v;
            }
        }
    }
}

/// Keys and values as contiguous rows (no closure, no modulo), otherwise the same.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_strided(q: *const f32, keys: *const f32, values: *const f32, count: usize, scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, HEAD) };
    let keys = unsafe { core::slice::from_raw_parts(keys, count * HEAD) };
    let values = unsafe { core::slice::from_raw_parts(values, count * HEAD) };
    let scores = unsafe { core::slice::from_raw_parts_mut(scores, count) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, HEAD) };
    for _ in 0..repeats {
        let scale = 1.0 / (HEAD as f32).sqrt();
        let grouped = count - count % 4;
        for start in (0..grouped).step_by(4) {
            let group = core::array::from_fn(|o| &keys[(start + o) * HEAD..][..HEAD]);
            for (o, v) in dot128_many(q, group).into_iter().enumerate() {
                scores[start + o] = v * scale;
            }
        }
        for j in grouped..count {
            scores[j] = dot128(q, &keys[j * HEAD..][..HEAD]) * scale;
        }
        softmax_scalar(scores);
        out.fill(0.0);
        for (j, weight) in scores.iter().enumerate() {
            for (o, v) in out.iter_mut().zip(&values[j * HEAD..][..HEAD]) {
                *o += weight * v;
            }
        }
    }
}

/// Strided, with the value accumulation in registers: 32 channels (8 vectors) per pass over the keys.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn attend_blocked(q: *const f32, keys: *const f32, values: *const f32, count: usize, scores: *mut f32, out: *mut f32, repeats: u32) {
    let q = unsafe { core::slice::from_raw_parts(q, HEAD) };
    let keys = unsafe { core::slice::from_raw_parts(keys, count * HEAD) };
    let values = unsafe { core::slice::from_raw_parts(values, count * HEAD) };
    let scores = unsafe { core::slice::from_raw_parts_mut(scores, count) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, HEAD) };
    for _ in 0..repeats {
        let scale = 1.0 / (HEAD as f32).sqrt();
        let grouped = count - count % 4;
        for start in (0..grouped).step_by(4) {
            let group = core::array::from_fn(|o| &keys[(start + o) * HEAD..][..HEAD]);
            for (o, v) in dot128_many(q, group).into_iter().enumerate() {
                scores[start + o] = v * scale;
            }
        }
        for j in grouped..count {
            scores[j] = dot128(q, &keys[j * HEAD..][..HEAD]) * scale;
        }
        softmax_scalar(scores);
        const SPAN: usize = 32;
        for first in (0..HEAD).step_by(SPAN) {
            let mut acc = [f32x4_splat(0.0); SPAN / 4];
            for (j, &weight) in scores.iter().enumerate() {
                let w = f32x4_splat(weight);
                let row = unsafe { values.as_ptr().add(j * HEAD + first) };
                for (k, acc) in acc.iter_mut().enumerate() {
                    let v = unsafe { v128_load(row.add(4 * k).cast()) };
                    *acc = f32x4_add(*acc, f32x4_mul(w, v));
                }
            }
            for (k, acc) in acc.iter().enumerate() {
                unsafe { v128_store(out.as_mut_ptr().add(first + 4 * k).cast(), *acc) };
            }
        }
    }
}
export_mm!(mm_lanes_q7_r1, mm_lanes::<Q7, 1, 4>);
export_mm!(mm_lanes_q4_r1, mm_lanes::<Q4, 1, 4>);
export_mm!(mm_lanes_q7_r1g1, mm_lanes::<Q7, 1, 1>);
export_mm!(mm_lanes_q4_r1g1, mm_lanes::<Q4, 1, 1>);
export_mm!(mm_lanes_q7_r1g2, mm_lanes::<Q7, 1, 2>);

// ---- the real fusor-ai kernel on synthetic weights ----

use std::cell::RefCell;

struct Real {
    table: kernels::Table,
    x: Vec<f32>,
    rows: kernels::QuantizedRows,
    out: Vec<f32>,
}

thread_local! {
    static REAL: RefCell<Vec<Real>> = const { RefCell::new(Vec::new()) };
}

fn pseudo(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Prepares a `[n, k]` matrix of `dtype` (7 or 4 bits) and `m` activation rows; returns its slot.
#[unsafe(no_mangle)]
pub extern "C" fn real_prepare(m: usize, k: usize, n: usize, bits: u32) -> usize {
    let dtype = if bits == 7 { kernels::DType::Q7b32 } else { kernels::DType::Q4b32 };
    let tensor = kernels::Tensor::encode(dtype, vec![n, k], &pseudo(n * k, 1)).unwrap();
    let real = Real {
        table: kernels::Table::new(tensor).unwrap(),
        x: pseudo(m * k, 7),
        rows: kernels::QuantizedRows::default(),
        out: vec![0.0; m * n],
    };
    REAL.with(|reals| {
        reals.borrow_mut().push(real);
        reals.borrow().len() - 1
    })
}

/// `repeats` matmuls over every output column (the activations quantized once).
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |row, col, values| {
                real.out[row * n + col..][..4].copy_from_slice(&values);
            });
        }
        real.out[0]
    })
}

/// As `real_matmul` with the page's QK store: `scale` applied per value into a strided buffer.
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul_scaled(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        let scale = std::hint::black_box(0.117851_f32);
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |row, col, values| {
                let scores = &mut real.out[row * n + col..][..4];
                for (score, value) in scores.iter_mut().zip(values) {
                    *score = value * scale;
                }
            });
        }
        real.out[0]
    })
}

/// As `real_matmul_scaled` with the scaling written as a map over the quad.
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul_scaled_map(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        let scale = std::hint::black_box(0.117851_f32);
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |row, col, values| {
                real.out[row * n + col..][..4].copy_from_slice(&values.map(|value| value * scale));
            });
        }
        real.out[0]
    })
}

/// As `real_matmul` through `Linear::project`'s per-element adapter (range filter, bias, one
/// store per value), as a layer's projections run.
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul_adapter(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        let cols = std::hint::black_box(0..n);
        let bias: Option<&[f32]> = std::hint::black_box(None);
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |row, first, values| {
                for (col, value) in (first..).zip(values).filter(|(col, _)| cols.contains(col)) {
                    real.out[row * n + col] = value + bias.map_or(0.0, |bias| bias[col]);
                }
            });
        }
        real.out[0]
    })
}

/// `real_matmul_adapter` with GELU through a function pointer, as `FeedForward::Plain` stores.
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul_adapter_gelu(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        let cols = std::hint::black_box(0..n);
        let bias: Option<&[f32]> = std::hint::black_box(None);
        let activation: fn(f32) -> f32 = std::hint::black_box(kernels::ops::gelu_erf);
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |row, first, values| {
                for (col, value) in (first..).zip(values).filter(|(col, _)| cols.contains(col)) {
                    real.out[row * n + col] = activation(value + bias.map_or(0.0, |bias| bias[col]));
                }
            });
        }
        real.out[0]
    })
}

/// As `real_matmul` with the cheapest store that keeps the work alive.
#[unsafe(no_mangle)]
pub extern "C" fn real_matmul_nostore(slot: usize, repeats: u32) -> f32 {
    REAL.with(|reals| {
        let mut reals = reals.borrow_mut();
        let real = &mut reals[slot];
        let matrix = real.table.matrix();
        let n = matrix.rows;
        let input = real.rows.input(&matrix, &real.x);
        let mut acc = 0.0;
        for _ in 0..repeats {
            kernels::matmul(&matrix, &input, 0..n, |_, _, values| {
                acc += values[0];
            });
        }
        acc
    })
}

export_mm!(mm_lanes_q7_r6, mm_lanes::<Q7, 6, 4>);
export_mm!(mm_lanes_q7_r8, mm_lanes::<Q7, 8, 4>);
export_mm!(mm_lanes_q4_r6, mm_lanes::<Q4, 6, 4>);
export_mm!(mm_lanes_q4_r8, mm_lanes::<Q4, 8, 4>);
export_mm!(mm_lanes_q4_r4g2, mm_lanes::<Q4, 4, 2>);
export_mm!(mm_lanes_q4_r8g2, mm_lanes::<Q4, 8, 2>);
