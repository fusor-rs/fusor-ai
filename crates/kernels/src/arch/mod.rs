//! The only per-target code: one `Backend` per instruction set — wasm relaxed SIMD
//! (`i32x4.relaxed_dot_i8x16_i7x16_add_s`; weights are 7-bit, so the result is defined on every
//! engine) or SIMD128 (`extmul` and pairwise adds), aarch64 NEON (`sdot`), and scalar. `Active`
//! is chosen at compile time from the target features; the scalar backend is also built for
//! tests, which check every backend against an f64 reference.

#![expect(
    unsafe_code,
    reason = "the backend contract loads blocks through raw pointers"
)]

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
mod neon;
#[cfg(any(
    test,
    not(any(
        target_arch = "wasm32",
        all(target_arch = "aarch64", target_feature = "dotprod")
    ))
))]
mod scalar;
#[cfg(target_arch = "wasm32")]
mod wasm;

use crate::quant::{BLOCK, CHUNK_INPUTS, QUAD, group_scale};

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub(crate) type Active = neon::Neon;
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_feature = "dotprod")
)))]
pub(crate) type Active = scalar::Scalar;
#[cfg(test)]
pub(crate) use scalar::Scalar;
#[cfg(target_arch = "wasm32")]
pub(crate) type Active = wasm::Wasm;

/// Lanes of a dot product: the rows of a quad, one output column each.
pub(crate) const LANES: usize = QUAD;
/// Chunks of a block.
pub(crate) const CHUNKS: usize = BLOCK / CHUNK_INPUTS;
/// Values per run of `exp_sweep`: the partial sums it returns, one per offset in a run.
pub(crate) const EXP_RUN: usize = 8;

/// The int8 dot-product contract every instruction set implements over the quad layout
/// (`quant`): a chunk is 16 codes, four consecutive inputs of each of a quad's four rows; the
/// activations that meet it are one row's same four inputs, broadcast to every lane; a dot
/// adds each lane's four products to it. The blocks of a group share a weight scale per lane
/// and one activation scale, so their integer dot products add up before one conversion to
/// f32.
pub(crate) trait Backend {
    type Chunk: Copy;
    /// Four i32 partial dot products, one per lane.
    type Dots: Copy;
    /// Four f32 partial sums, one per lane.
    type Sums: Copy;

    /// Activation rows per matmul tile (each weight chunk is loaded once for all of them): as
    /// many as the target keeps in registers, measured per instruction set (at most 6).
    const TILE: usize;

    /// # Safety
    /// 128 readable bytes at `at`.
    unsafe fn load_q7(at: *const u8) -> [Self::Chunk; CHUNKS];

    /// # Safety
    /// 64 readable bytes at `at`.
    unsafe fn load_q4(at: *const u8) -> [Self::Chunk; CHUNKS];

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_t2(at: *const u8) -> [Self::Chunk; CHUNKS];

    /// Four activations, broadcast to every lane.
    ///
    /// # Safety
    /// 4 readable bytes at `at`.
    unsafe fn broadcast(at: *const i8) -> Self::Chunk;

    /// A quad's scales for one group.
    ///
    /// # Safety
    /// 4 readable f32s at `at`.
    unsafe fn load_scales(at: *const f32) -> Self::Sums;

    fn zero() -> Self::Sums;

    /// The largest magnitude of 32 values.
    fn absmax(x: &[f32]) -> f32;

    /// The largest of `x` (`−∞` of none).
    fn maximum(x: &[f32]) -> f32;

    /// Quantizes 32 values to int8 at scale `d` (`round_ties_even(x / d)`); returns `Σq`.
    fn quantize_block(x: &[f32], scale: f32, out: &mut [i8]) -> i32;

    /// The dot products of a group before its first block: `correction`, its `−zero_point · Σa`,
    /// in every lane.
    fn seed(correction: i32) -> Self::Dots;

    /// `dots + Σ a·w` over one chunk, per lane.
    fn dot(dots: Self::Dots, activations: Self::Chunk, weights: Self::Chunk) -> Self::Dots;

    /// `sums + dots · column_scales · row_scale`.
    fn scale(
        sums: Self::Sums,
        dots: Self::Dots,
        column_scales: Self::Sums,
        row_scale: f32,
    ) -> Self::Sums;

    fn lanes(sums: Self::Sums) -> [f32; LANES];

    /// `x[i] = exp(x[i] − shift)` with `ops::exp`'s arithmetic over the whole runs of `EXP_RUN`
    /// values at the start of `x` (the rest untouched); returns the sums of the results at
    /// each offset in a run.
    fn exp_sweep(x: &mut [f32], shift: f32) -> [f32; EXP_RUN];

    /// `ops::gelu_tanh` of each value, with its arithmetic.
    fn gelu_tanh(x: [f32; LANES]) -> [f32; LANES];

    fn dot_f32(left: &[f32], right: &[f32]) -> f32;

    /// `dot_f32(left, right)` for each of `rights`, each summed in exactly its order.
    fn dot_f32_many(left: &[f32], rights: [&[f32]; DOTS]) -> [f32; DOTS] {
        rights.map(|right| Self::dot_f32(left, right))
    }

    /// `out[i] += weights[k] · rights[k][i]` for `k` in order, each product added on its own
    /// (no fused multiply-add), as adding the rows one at a time would.
    fn axpy_f32_many(out: &mut [f32], weights: [f32; DOTS], rights: [&[f32]; DOTS]) {
        for (weight, right) in weights.iter().zip(rights) {
            for (output, value) in out.iter_mut().zip(right) {
                *output += weight * value;
            }
        }
    }
}

/// The dot product of two f32 slices with the target's SIMD.
pub fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    <Active as Backend>::dot_f32(left, right)
}

/// How many dot products `dot_f32_many` takes at once.
pub const DOTS: usize = 4;

/// `dot_f32(left, right)` for each of `rights`, reading `left` once.
pub fn dot_f32_many(left: &[f32], rights: [&[f32]; DOTS]) -> [f32; DOTS] {
    <Active as Backend>::dot_f32_many(left, rights)
}

/// `each(i, dot_f32(left, right(i)))` for `i < count`, `DOTS` dot products at a time, each
/// summed in exactly `dot_f32`'s order.
pub fn dot_f32_each<'rows>(
    left: &[f32],
    count: usize,
    right: impl Fn(usize) -> &'rows [f32],
    each: impl FnMut(usize, f32),
) {
    dot_f32_each_with::<Active>(left, count, right, each);
}

pub(crate) fn dot_f32_each_with<'rows, B: Backend>(
    left: &[f32],
    count: usize,
    right: impl Fn(usize) -> &'rows [f32],
    mut each: impl FnMut(usize, f32),
) {
    let grouped = count - count % DOTS;
    for first in (0..grouped).step_by(DOTS) {
        let rights = std::array::from_fn(|offset| right(first + offset));
        for (offset, value) in B::dot_f32_many(left, rights).into_iter().enumerate() {
            each(first + offset, value);
        }
    }
    for index in grouped..count {
        each(index, B::dot_f32(left, right(index)));
    }
}

const ACTIVATION_MAX: f32 = 127.0;

/// `d = absmax / 127` (1 for an all-zero group): an activation group's scale.
pub(crate) fn activation_scale(absmax: f32) -> f32 {
    group_scale(absmax, ACTIVATION_MAX)
}
