//! f32 operations of transformer inference: normalization, rotary embeddings, activations and
//! attention.

use crate::arch::dot_f32;

pub const NORM_EPS: f32 = 1e-6;

/// `out = x · rsqrt(mean(x²) + eps) · weight` for every row of `x` (rows of `weight.len()`).
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    let dim = weight.len();
    for (row, out_row) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean = dot_f32(row, row) / dim as f32;
        let inverse = 1.0 / (mean + NORM_EPS).sqrt();
        for ((output, value), scale) in out_row.iter_mut().zip(row).zip(weight) {
            *output = value * inverse * scale;
        }
    }
}

/// RoPE frequencies `base^(−2i/head_dim)` for `i < head_dim / 2`, computed in f32 like
/// `1 / base ** (arange(0, d, 2).float() / d)`.
pub fn rope_inv_freq(base: f32, head_dim: usize) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect()
}

/// Cos and sin of `position · inv_freq[i]` (angles in f32).
pub fn rope_angles(position: usize, inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    for ((frequency, cosine), sine) in inv_freq.iter().zip(cos.iter_mut()).zip(sin.iter_mut()) {
        let angle = position as f32 * frequency;
        *cosine = angle.cos();
        *sine = angle.sin();
    }
}

/// Rotation by halves (`rotate_half`, Llama/Qwen): element `i` pairs with `i + d/2`.
pub fn rope_half(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = head.len() / 2;
    let (first, second) = head.split_at_mut(half);
    for i in 0..half {
        let (x, y) = (first[i], second[i]);
        first[i] = x * cos[i] - y * sin[i];
        second[i] = y * cos[i] + x * sin[i];
    }
}

/// Interleaved rotation (x-transformers): element `2i` pairs with `2i + 1`, both rotated by
/// angle `i`.
pub fn rope_interleaved(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    for (i, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (x, y) = (pair[0], pair[1]);
        pair[0] = x * cos[i] - y * sin[i];
        pair[1] = y * cos[i] + x * sin[i];
    }
}

/// `eˣ` within about 2 ulp for `x ∈ [−87, 88]` (clamped outside), branch-free so that loops
/// over it vectorize: `2ⁿ · p(r)` with `n = round(x / ln 2)` and a degree-6 polynomial on
/// `|r| ≤ ln 2 / 2` (Cephes `expf` coefficients).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LN2_HIGH: f32 = 0.693_359_4;
    const LN2_LOW: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    let exponent = (x * std::f32::consts::LOG2_E).round_ties_even();
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

/// GELU, tanh approximation (`nn.GELU(approximate="tanh")`), as `x · σ(2u)`
/// (`0.5 · (1 + tanh u) = σ(2u)`).
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    x / (1.0 + exp(-2.0 * inner))
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

/// Softmax attention of one query over `keys` positions, with `1/√d` scaling. `key_value(j)`
/// gives position `j`'s key and value (each `query.len()` long); `scores` is scratch.
pub fn attend<'cache>(
    query: &[f32],
    keys: usize,
    key_value: impl Fn(usize) -> (&'cache [f32], &'cache [f32]),
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let scale = 1.0 / (query.len() as f32).sqrt();
    scores.clear();
    scores.extend((0..keys).map(|j| dot_f32(query, key_value(j).0) * scale));
    let max = scores
        .iter()
        .fold(f32::NEG_INFINITY, |max, &score| max.max(score));
    let mut total = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }
    out.fill(0.0);
    for (j, score) in scores.iter().enumerate() {
        let weight = score / total;
        for (output, component) in out.iter_mut().zip(key_value(j).1) {
            *output += weight * component;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{exp, gelu_tanh};

    #[test]
    fn exp_and_gelu_match_libm() {
        for i in -2000..=2000 {
            let x = i as f32 / 100.0;
            let want = f64::from(x).exp();
            assert!(((f64::from(exp(x)) - want) / want).abs() < 4e-7, "exp({x})");
            let wide = f64::from(x);
            let gelu =
                0.5 * wide * (1.0 + (0.797_884_6 * (wide + 0.044_715 * wide.powi(3))).tanh());
            let error = (f64::from(gelu_tanh(x)) - gelu).abs();
            assert!(error < 1e-6 * (1.0 + gelu.abs()), "gelu({x})");
        }
    }
}
