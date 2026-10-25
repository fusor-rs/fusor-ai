//! f32 operations of transformer inference: normalization, rotary embeddings, activations and
//! attention.

use crate::arch::{Active, Backend, DOTS, EXP_RUN, dot_f32, dot_f32_each};
use crate::quant::QUAD;

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

/// `exp`'s range: clamped outside `[EXP_MIN, EXP_MAX]`; `exp_fast` clamps below at `EXP_MIN` too.
pub(crate) const EXP_MIN: f32 = -87.0;
pub(crate) const EXP_MAX: f32 = 88.0;
/// `ln 2` in two parts, so that `x − n · ln 2` stays exact.
pub(crate) const LN2_HIGH: f32 = 0.693_359_4;
pub(crate) const LN2_LOW: f32 = -2.121_944_4e-4;
/// Cephes `expf`'s polynomial on `|r| ≤ ln 2 / 2`, highest degree first; `r² · p(r) + r + 1`
/// completes it.
pub(crate) const EXP_POLYNOMIAL: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_66,
    0.5,
];
/// f32's exponent bias.
pub(crate) const EXP_BIAS: i32 = 127;

/// `eˣ` within about 2 ulp for `x ∈ [EXP_MIN, EXP_MAX]` (clamped outside), branch-free so that
/// loops over it vectorize: `2ⁿ · p(r)` with `n = round(x / ln 2)` and a degree-6 polynomial on
/// `|r| ≤ ln 2 / 2` (Cephes `expf` coefficients). The backends' `exp_sweep` is the same
/// arithmetic on four lanes.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(EXP_MIN, EXP_MAX);
    let exponent = (x * std::f32::consts::LOG2_E).round_ties_even();
    let remainder = x - exponent * LN2_HIGH - exponent * LN2_LOW;
    let mut poly = EXP_POLYNOMIAL[0];
    for coefficient in &EXP_POLYNOMIAL[1..] {
        poly = poly * remainder + coefficient;
    }
    let poly = poly * remainder * remainder + remainder + 1.0;
    poly * f32::from_bits(((exponent as i32 + EXP_BIAS) << 23) as u32)
}

/// `exp_fast`'s Taylor polynomial of `eʳ`, degree 4, highest first; the last two are the linear
/// and constant terms.
pub(crate) const EXP_TAYLOR: [f32; 5] = [1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];

/// `eˣ` for `x ≤ 0` within 6e-5 relative (measured in the kernel harness), a hundred times under
/// the 1/127 step of the int8 probabilities it feeds: `2ⁿ · p(r)` with `n = round(x / ln 2)`,
/// `r = x − n · ln 2` in one fused operation, and `p` a degree-4 Taylor polynomial evaluated
/// with fused multiply-adds. The backends' `exp_sweep_fast` is the same arithmetic on four lanes,
/// its multiply-adds fused where the target fuses them (NEON, relaxed SIMD, the scalar
/// `mul_add`); SIMD128 multiplies then adds. Both are within the 6e-5 bound.
///
/// On x86-64 the scalar `mul_add` and `round_ties_even` are library calls, so the scalar
/// backend's fast sweep is slow there.
#[inline(always)]
pub(crate) fn exp_fast(x: f32) -> f32 {
    // No upper clamp: its callers subtract the row's maximum first.
    let x = x.max(EXP_MIN);
    let exponent = (x * std::f32::consts::LOG2_E).round_ties_even();
    let remainder = (-exponent).mul_add(std::f32::consts::LN_2, x);
    let mut poly = EXP_TAYLOR[0];
    for coefficient in &EXP_TAYLOR[1..] {
        poly = poly.mul_add(remainder, *coefficient);
    }
    poly * f32::from_bits(((exponent as i32 + EXP_BIAS) << 23) as u32)
}

/// The exponential of a row of scores less their maximum: `Exact` is `exp`, within 2 ulp;
/// `Fast` is `exp_fast`, for scores whose exponentials are quantized to int8 next, where its
/// error is far under the quantization's step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Exponentials {
    Exact,
    Fast,
}

impl Exponentials {
    /// `scores` replaced by the exponential of each score less their maximum; returns their sum.
    pub fn apply(self, scores: &mut [f32]) -> f32 {
        let max = Active::maximum(scores);
        let sums = self.exponentiate(scores, max);
        let (_, rest) = scores.as_chunks::<EXP_RUN>();
        rest.iter().chain(&sums).sum()
    }

    /// `scores` replaced by the variant's exponential of `score − max`: the backend's sweep over
    /// the whole runs of `EXP_RUN` values, its scalar twin over the rest; returns the sweep's
    /// sums of its results at each offset in a run.
    fn exponentiate(self, scores: &mut [f32], max: f32) -> [f32; EXP_RUN] {
        let (sums, exponential): (_, fn(f32) -> f32) = match self {
            Self::Exact => (Active::exp_sweep(scores, max), exp),
            Self::Fast => (Active::exp_sweep_fast(scores, max), exp_fast),
        };
        let (_, rest) = scores.as_chunks_mut::<EXP_RUN>();
        for score in rest.iter_mut() {
            *score = exponential(*score - max);
        }
        sums
    }
}

/// `gelu_tanh`'s `u = √(2/π) · (x + 0.044715 · x³)`.
pub(crate) const GELU_TANH_SCALE: f32 = 0.797_884_6;
pub(crate) const GELU_TANH_CUBIC: f32 = 0.044_715;

/// GELU, tanh approximation (`nn.GELU(approximate="tanh")`), as `x · σ(2u)`
/// (`0.5 · (1 + tanh u) = σ(2u)`).
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    let inner = GELU_TANH_SCALE * (x + GELU_TANH_CUBIC * x * x * x);
    x / (1.0 + exp(-2.0 * inner))
}

/// `gelu_tanh` of four values at once, each bit for bit the scalar's.
pub fn gelu_tanh_lanes(x: [f32; QUAD]) -> [f32; QUAD] {
    Active::gelu_tanh(x)
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

/// HardSwish as ONNX defines it, `x · HardSigmoid(x)` with `HardSigmoid(x) = clamp(x/6 + 1/2,
/// 0, 1)`.
#[inline(always)]
pub fn hard_swish(x: f32) -> f32 {
    const SLOPE: f32 = 1.0 / 6.0;
    x * (SLOPE * x + 0.5).clamp(0.0, 1.0)
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `scores` replaced by their softmax, the exponentials summed in their order.
pub fn softmax(scores: &mut [f32]) {
    let max = Active::maximum(scores);
    Exponentials::Exact.exponentiate(scores, max);
    let total: f32 = scores.iter().sum();
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
    scores.resize(keys, 0.0);
    dot_f32_each(
        query,
        keys,
        |j| key_value(j).0,
        |j, dot| scores[j] = dot * scale,
    );
    softmax(scores);
    out.fill(0.0);
    let (blocks, rest) = scores.as_chunks::<DOTS>();
    for (block, weights) in blocks.iter().enumerate() {
        let values = std::array::from_fn(|offset| key_value(block * DOTS + offset).1);
        Active::axpy_f32_many(out, *weights, values);
    }
    for (j, weight) in (blocks.len() * DOTS..).zip(rest) {
        for (output, component) in out.iter_mut().zip(key_value(j).1) {
            *output += weight * component;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Exponentials, elu, exp, exp_fast, gelu_erf, gelu_tanh, gelu_tanh_lanes, layer_norm,
    };
    use crate::arch::{Active, Backend, DOTS};

    #[test]
    fn exp_and_activations_match_the_f64_reference() {
        for i in -2000..=2000 {
            let x = i as f32 / 100.0;
            let want = f64::from(x).exp();
            assert!(((f64::from(exp(x)) - want) / want).abs() < 4e-7, "exp({x})");
            if x <= 0.0 {
                let error = (f64::from(exp_fast(x)) - want) / want;
                assert!(error.abs() < 6e-5, "exp_fast({x})");
            }
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
    fn axpy_f32_many_adds_the_rows_one_at_a_time_bit_for_bit() {
        let rows: [Vec<f32>; DOTS] = std::array::from_fn(|row| {
            (0..37)
                .map(|i| ((i * 7 + row * 3) % 11) as f32 / 3.0 - 1.5)
                .collect()
        });
        let weights = [0.3, -1.25, 0.7, 2.0];
        let mut want: Vec<f32> = (0..37).map(|i| i as f32 / 5.0).collect();
        let mut got = want.clone();
        for (weight, row) in weights.iter().zip(&rows) {
            for (output, value) in want.iter_mut().zip(row) {
                *output += weight * value;
            }
        }
        Active::axpy_f32_many(&mut got, weights, rows.each_ref().map(Vec::as_slice));
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn decay_rows_updates_the_rows_one_at_a_time_bit_for_bit() {
        let mut want: [Vec<f32>; DOTS] = std::array::from_fn(|row| {
            (0..37)
                .map(|i| ((i * 7 + row * 3) % 11) as f32 / 3.0 - 1.5)
                .collect()
        });
        let mut got = want.clone();
        let key: Vec<f32> = (0..37).map(|i| ((i * 5) % 13) as f32 / 4.0 - 1.0).collect();
        let decay = 0.93;
        let deltas = [0.3, -1.25, 0.7, 2.0];
        for (row, delta) in want.iter_mut().zip(deltas) {
            for (value, key) in row.iter_mut().zip(&key) {
                *value = decay * *value + delta * key;
            }
        }
        Active::decay_rows(got.each_mut().map(Vec::as_mut_slice), decay, deltas, &key);
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn gelu_tanh_lanes_are_the_scalar_gelu_bit_for_bit() {
        for i in (-2000..=2000).step_by(4) {
            let x = [i, i + 1, i + 2, i + 3].map(|i| i as f32 / 100.0);
            let want = x.map(gelu_tanh).map(f32::to_bits);
            assert_eq!(
                gelu_tanh_lanes(x).map(f32::to_bits),
                want,
                "gelu_tanh({x:?})"
            );
        }
    }

    #[test]
    fn exponentials_are_the_scalar_exp_of_every_score_less_their_maximum() {
        let variants = [
            (Exponentials::Exact, exp as fn(f32) -> f32, 1e-6),
            (Exponentials::Fast, exp_fast, 1e-4),
        ];
        for (exponentials, exponential, tolerance) in variants {
            for length in [37, 45] {
                let scores: Vec<f32> = (0..length)
                    .map(|i| (i as f32).mul_add(-0.37, -1.5))
                    .collect();
                let mut swept = scores.clone();
                let sum = exponentials.apply(&mut swept);
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for (score, value) in scores.iter().zip(&swept) {
                    assert_eq!(
                        value.to_bits(),
                        exponential(score - max).to_bits(),
                        "{exponentials:?}: exp({score} − {max})"
                    );
                }
                let want: f64 = scores
                    .iter()
                    .map(|score| f64::from(score - max).exp())
                    .sum();
                assert!(
                    (f64::from(sum) - want).abs() < tolerance * want,
                    "{exponentials:?}: {sum} vs {want}"
                );
            }
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
