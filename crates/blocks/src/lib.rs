//! Architecture blocks: shape-parameterized pieces of model architectures, written once on the
//! fusor-ai kernels and thread team. A block never names a checkpoint, a tensor of a particular
//! model, a sample rate or a language; the models in `models/` configure and wire them.
//!
//! Every block loads itself from a bundle with `bundle::Load`, under the tensor names relative to
//! a prefix that its documentation lists.

pub mod adaln_mlp;
pub mod apg;
pub mod attention;
pub mod audio;
mod causal;
pub mod code2wav;
pub mod codec;
pub mod conformer;
pub mod depth;
pub mod dit;
pub mod dpm;
pub mod ecapa;
pub mod flow;
pub mod hybrid;
pub mod interleaved;
pub mod istft;
mod layer;
pub mod lm;
pub mod mmdit;
pub mod norm;
mod pixel_attention;
pub mod pyramid_jit;
pub mod reading;
mod resnet;
pub mod rotary;
mod row_work;
pub mod seanet;
pub mod t5;
mod time_embedding;
pub mod transformer;
pub mod unet;
pub mod vae;
pub mod video;
pub mod vit;
pub mod wan_dit;
pub mod wan_vae;

/// The base of the sinusoids that encode a position or a time (`10000^(−2i/d)`).
pub const SINUSOID_BASE: f32 = 10_000.0;

/// Chunk `index` of the `hidden` values each of `values`.
pub(crate) fn section(values: &[f32], hidden: usize, index: usize) -> &[f32] {
    &values[index * hidden..(index + 1) * hidden]
}

/// `value`, or `1 + value` where `chunk` is one of `scales`.
pub(crate) fn scaled(chunk: usize, scales: &[usize], value: f32) -> f32 {
    if scales.contains(&chunk) {
        1.0 + value
    } else {
        value
    }
}

/// The timestep's sinusoids, `cos(t · fᵢ)` then `sin(t · fᵢ)` for `fᵢ = period^(−i / half)`, in
/// f32 like diffusers' `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)` (period 10 000:
/// Stable Diffusion's UNet, FLUX.1's, HunyuanVideo's and Motif-Video's transformers; 10: Iris's).
/// `timestep_sinusoid_f64` computes it in f64.
pub fn timestep_sinusoid(timestep: f32, period: f32, out: &mut [f32]) {
    let half = out.len() / 2;
    let (cosines, sines) = out.split_at_mut(half);
    for (i, (cosine, sine)) in cosines.iter_mut().zip(sines).enumerate() {
        let exponent = -period.ln() * i as f32 / half as f32;
        let angle = timestep * exponent.exp();
        *cosine = angle.cos();
        *sine = angle.sin();
    }
}

/// `timestep_sinusoid` in f64, rounded to f32 at the end, as Wan's and P-JiT's
/// `sinusoidal_embedding_1d` compute it.
pub fn timestep_sinusoid_f64(timestep: f32, period: f32, out: &mut [f32]) {
    let half = out.len() / 2;
    let (cosines, sines) = out.split_at_mut(half);
    for (i, (cosine, sine)) in cosines.iter_mut().zip(sines).enumerate() {
        let decay = f64::from(period).powf(-(i as f64) / half as f64);
        let angle = f64::from(timestep) * decay;
        *cosine = angle.cos() as f32;
        *sine = angle.sin() as f32;
    }
}
