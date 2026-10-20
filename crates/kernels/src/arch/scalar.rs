#![expect(
    unsafe_code,
    reason = "block loads through raw pointers, as on the SIMD targets"
)]

pub(crate) struct Weights([i32; 32]);
pub(crate) struct Activations([i32; 32]);

pub(crate) struct Scalar;

impl super::Backend for Scalar {
    type Weights = Weights;
    type Activations = Activations;
    type Sums = f32;

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_q7(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 32 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, 32) };
        Weights(std::array::from_fn(|j| i32::from(bytes[j])))
    }

    /// # Safety
    /// 16 readable bytes at `at`.
    unsafe fn load_q4(at: *const u8) -> Self::Weights {
        // SAFETY: the caller guarantees 16 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, 16) };
        Weights(std::array::from_fn(|j| {
            i32::from(if j < 16 {
                bytes[j] & 0x0f
            } else {
                bytes[j - 16] >> 4
            })
        }))
    }

    /// # Safety
    /// 32 readable bytes at `at`.
    unsafe fn load_activations(at: *const i8) -> Self::Activations {
        // SAFETY: the caller guarantees 32 readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(at, 32) };
        Activations(std::array::from_fn(|j| i32::from(bytes[j])))
    }

    fn zero() -> f32 {
        0.0
    }

    /// Quantizes 32 values to int8 (`round_ties_even(x / d)`); returns `(d, Σq)`.
    fn quantize_block(x: &[f32], out: &mut [i8]) -> (f32, i32) {
        let absmax = x.iter().fold(0.0f32, |max, value| max.max(value.abs()));
        let scale = super::activation_scale(absmax);
        let mut sum = 0;
        for (output, value) in out.iter_mut().zip(x) {
            let limit = super::ACTIVATION_MAX;
            let level = (value / scale).round_ties_even().clamp(-limit, limit) as i8;
            sum += i32::from(level);
            *output = level;
        }
        (scale, sum)
    }

    fn accumulate(
        sums: f32,
        activations: &Self::Activations,
        weights: &Self::Weights,
        correction: i32,
        scale: f32,
    ) -> f32 {
        let products = activations.0.iter().zip(&weights.0);
        let dot: i32 = products
            .map(|(activation, weight)| activation * weight)
            .sum();
        sums + (dot + correction) as f32 * scale
    }

    fn sum(sums: f32) -> f32 {
        sums
    }

    fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}
