//! The only per-target code: wasm relaxed SIMD (`i32x4.relaxed_dot_i8x16_i7x16_add_s`; weights
//! are 7-bit, so the result is defined on every engine), wasm SIMD128 (`extmul` and pairwise
//! adds), aarch64 NEON (`sdot`) and a scalar fallback. The path is chosen at compile time from
//! the target features.

#[cfg(target_arch = "wasm32")]
#[path = "wasm.rs"]
mod target;

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[path = "neon.rs"]
mod target;

#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_feature = "dotprod")
)))]
#[path = "scalar.rs"]
mod target;

pub use target::dot_f32;
pub(crate) use target::{
    accumulate, load_activations, load_q4, load_q7, quantize_block, sum, zero,
};

const ACTIVATION_MAX: f32 = 127.0;

/// `d = absmax / 127` (1 for an all-zero block): an activation block's scale.
fn activation_scale(absmax: f32) -> f32 {
    if absmax == 0.0 {
        1.0
    } else {
        absmax / ACTIVATION_MAX
    }
}
