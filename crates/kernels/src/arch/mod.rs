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

use crate::quant::group_scale;

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

/// The int8 block-dot contract every instruction set implements. A block is 32 values: 32 weight
/// bytes (`Q7b32`, values ≤ 127), 16 packed nibble bytes (`Q4b32`) or 8 bytes of two-bit codes
/// (`T2b128`), and 32 int8 activations. The blocks of a group share a weight scale and an
/// activation scale, so their integer dot products add up before one conversion to f32.
pub(crate) trait Backend {
    type Weights;
    type Activations;
    /// Four i32 partial dot products.
    type Dots: Copy;
    /// Four f32 partial sums.
    type Sums: Copy;

    /// Activation rows per matmul tile (each weight block is unpacked once for all of them): as
    /// many as the target keeps in registers, measured per instruction set (at most 6).
    const TILE: usize;

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_q7(at: *const u8) -> Self::Weights;

    /// # Safety
    /// 16 readable bytes at `at`.
    unsafe fn load_q4(at: *const u8) -> Self::Weights;

    /// # Safety
    /// 8 readable bytes at `at`.
    unsafe fn load_t2(at: *const u8) -> Self::Weights;

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_activations(at: *const i8) -> Self::Activations;

    fn zero() -> Self::Sums;

    /// The largest magnitude of 32 values.
    fn absmax(x: &[f32]) -> f32;

    /// Quantizes 32 values to int8 at scale `d` (`round_ties_even(x / d)`); returns `Σq`.
    fn quantize_block(x: &[f32], scale: f32, out: &mut [i8]) -> i32;

    /// The dot products of a group before its first block: `correction`, its `−zero_point · Σa`.
    fn seed(correction: i32) -> Self::Dots;

    /// `dots + Σ a·w` over one block.
    fn dot(
        dots: Self::Dots,
        activations: &Self::Activations,
        weights: &Self::Weights,
    ) -> Self::Dots;

    /// `sums + dots · scale`.
    fn scale(sums: Self::Sums, dots: Self::Dots, scale: f32) -> Self::Sums;

    fn sum(sums: Self::Sums) -> f32;

    fn dot_f32(left: &[f32], right: &[f32]) -> f32;

    /// `dot_f32(left, right)` for each of `rights`, each summed in exactly its order.
    fn dot_f32_many(left: &[f32], rights: [&[f32]; DOTS]) -> [f32; DOTS] {
        rights.map(|right| Self::dot_f32(left, right))
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

const ACTIVATION_MAX: f32 = 127.0;

/// `d = absmax / 127` (1 for an all-zero group): an activation group's scale.
pub(crate) fn activation_scale(absmax: f32) -> f32 {
    group_scale(absmax, ACTIVATION_MAX)
}
