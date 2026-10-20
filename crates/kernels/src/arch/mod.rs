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
/// bytes (`Q7b32`, values ≤ 127) or 16 packed nibble bytes (`Q4b32`), and 32 int8 activations.
pub(crate) trait Backend {
    type Weights;
    type Activations;
    /// Four f32 partial sums.
    type Sums: Copy;

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_q7(at: *const u8) -> Self::Weights;

    /// # Safety
    /// 16 readable bytes at `at`.
    unsafe fn load_q4(at: *const u8) -> Self::Weights;

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_activations(at: *const i8) -> Self::Activations;

    fn zero() -> Self::Sums;

    /// Quantizes 32 values to int8 (`round_ties_even(x / d)`); returns `(d, Σq)`.
    fn quantize_block(x: &[f32], out: &mut [i8]) -> (f32, i32);

    /// `sums + (Σ a·w + correction) · scale`, the correction being `−zero_point · Σa`.
    fn accumulate(
        sums: Self::Sums,
        activations: &Self::Activations,
        weights: &Self::Weights,
        correction: i32,
        scale: f32,
    ) -> Self::Sums;

    fn sum(sums: Self::Sums) -> f32;

    fn dot_f32(left: &[f32], right: &[f32]) -> f32;
}

/// The dot product of two f32 slices with the target's SIMD.
pub fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    <Active as Backend>::dot_f32(left, right)
}

const ACTIVATION_MAX: f32 = 127.0;

/// `d = absmax / 127` (1 for an all-zero block): an activation block's scale.
fn activation_scale(absmax: f32) -> f32 {
    if absmax == 0.0 {
        1.0
    } else {
        absmax / ACTIVATION_MAX
    }
}
