//! f32 operations of transformer inference: normalization, rotary embeddings, activations and
//! attention.

use crate::arch::dot_f32;

/// `out = x · rsqrt(mean(x²) + eps) · weight` for every row of `x` (rows of `weight.len()`).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = weight.len();
    for (row, out_row) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean = dot_f32(row, row) / dim as f32;
        let inverse = 1.0 / (mean + eps).sqrt();
        for ((output, value), scale) in out_row.iter_mut().zip(row).zip(weight) {
            *output = value * inverse * scale;
        }
    }
}

/// `out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias` for every row of `x` (rows of
/// `weight.len()`; the variance is the biased one, as `nn.LayerNorm`).
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let dim = weight.len();
    for (row, out_row) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / dim as f32;
        let inverse = 1.0 / (variance + eps).sqrt();
        let affine = weight.iter().zip(bias);
        for ((output, value), (scale, shift)) in out_row.iter_mut().zip(row).zip(affine) {
            *output = (value - mean) * inverse * scale + shift;
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

/// `erf(x)` within about 2e-7 absolute, branch-free: the odd rational approximation on
/// `[−4, 4]` that Eigen and XLA use for f32 (`erf` is ±1 in f32 beyond).
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    const NUMERATOR: [f32; 7] = [
        -2.726_142_3e-10,
        2.770_681_4e-8,
        -2.101_024e-6,
        -5.692_506_4e-5,
        -7.349_906_3e-4,
        -2.954_600_1e-3,
        -1.609_603_3e-2,
    ];
    const DENOMINATOR: [f32; 5] = [
        -1.456_607_2e-5,
        -2.133_740_6e-4,
        -1.682_827e-3,
        -7.373_329_2e-3,
        -1.426_474e-2,
    ];
    let x = x.clamp(-4.0, 4.0);
    let square = x * x;
    let numerator = NUMERATOR.iter().fold(0.0, |sum, &term| sum * square + term);
    let denominator = DENOMINATOR
        .iter()
        .fold(0.0, |sum, &term| sum * square + term);
    x * numerator / denominator
}

/// GELU with the exact erf (`nn.GELU()`): `x · (1 + erf(x / √2)) / 2`.
#[inline(always)]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// CLIP's GELU approximation, `x · σ(1.702 x)`.
#[inline(always)]
pub fn quick_gelu(x: f32) -> f32 {
    const SHARPNESS: f32 = 1.702;
    x / (1.0 + exp(-SHARPNESS * x))
}

/// ELU with `α = 1`: `x` for `x > 0`, `eˣ − 1` otherwise.
#[inline(always)]
pub fn elu(x: f32) -> f32 {
    let negative = exp(x.min(0.0)) - 1.0;
    if x > 0.0 { x } else { negative }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `scores` replaced by their softmax.
pub fn softmax(scores: &mut [f32]) {
    let max = scores
        .iter()
        .fold(f32::NEG_INFINITY, |max, &score| max.max(score));
    let mut total = 0.0;
    for score in scores.iter_mut() {
        *score = exp(*score - max);
        total += *score;
    }
    scores.iter_mut().for_each(|score| *score /= total);
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
    softmax(scores);
    out.fill(0.0);
    for (j, weight) in scores.iter().enumerate() {
        for (output, component) in out.iter_mut().zip(key_value(j).1) {
            *output += weight * component;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{elu, exp, gelu_erf, gelu_tanh, layer_norm};

    #[test]
    fn exp_and_activations_match_the_f64_reference() {
        for i in -2000..=2000 {
            let x = i as f32 / 100.0;
            let want = f64::from(x).exp();
            assert!(((f64::from(exp(x)) - want) / want).abs() < 4e-7, "exp({x})");
            let elu_want = if x > 0.0 { f64::from(x) } else { want - 1.0 };
            assert!((f64::from(elu(x)) - elu_want).abs() < 1e-7, "elu({x})");
            let wide = f64::from(x);
            let gelu =
                0.5 * wide * (1.0 + (0.797_884_6 * (wide + 0.044_715 * wide.powi(3))).tanh());
            let error = (f64::from(gelu_tanh(x)) - gelu).abs();
            assert!(error < 1e-6 * (1.0 + gelu.abs()), "gelu({x})");
        }
        let erf_gelu = [
            (-3.0, -0.004_049_694_950_713),
            (-1.0, -0.158_655_253_931_457),
            (-0.5, -0.154_268_769_362_993),
            (0.5, 0.345_731_230_637_007),
            (1.0, 0.841_344_746_068_543),
            (2.0, 1.954_499_736_103_642),
            (5.0, 4.999_998_566_734_9),
        ];
        for (x, want) in erf_gelu {
            let error = (f64::from(gelu_erf(x as f32)) - want).abs();
            assert!(error < 3e-7 * (1.0 + want.abs()), "gelu_erf({x})");
        }
    }

    #[test]
    fn layer_norm_centers_scales_and_shifts() {
        let mut out = [0.0; 4];
        layer_norm(
            &[0.0, 0.0, 2.0, 2.0],
            &[1.0, 2.0, 1.0, 1.0],
            &[0.0, 0.0, 1.0, 0.0],
            0.0,
            &mut out,
        );
        let want = [-1.0, -2.0, 2.0, 1.0];
        for (got, want) in out.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{out:?}");
        }
    }
}
