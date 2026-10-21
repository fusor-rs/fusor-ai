//! Pixels for the page: the decoder's image, and a preview of the clean latents a step predicts
//! at the latents' resolution, each latent pixel through a linear map to RGB. The map is a least
//! squares fit to the decoder's images average-pooled to that resolution (realistic-vision's
//! `scripts/preview-factors.py`): rough colours and shapes, at no cost beside a step's.

use crate::model::{IMAGE_CHANNELS, LATENT_CHANNELS};

/// Per output channel (red, green, blue): the weights of the latent channels, then a bias, into
/// the decoder's range `[−1, 1]`.
const PREVIEW: [[f32; LATENT_CHANNELS + 1]; IMAGE_CHANNELS] = [
    [0.2884, 0.1866, -0.0835, -0.2140, -0.0278],
    [0.2422, 0.2384, 0.1535, -0.2968, -0.0868],
    [0.2481, 0.1367, 0.2215, -0.5044, -0.1628],
];
/// Bytes of an image pixel: red, green, blue, opaque alpha.
const RGBA: usize = 4;

/// Pixel-major RGB in `[−1, 1]` as RGBA bytes: `round(255 · clamp(x / 2 + 1/2, 0, 1))`.
pub(crate) fn rgba(image: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(image.len() / IMAGE_CHANNELS * RGBA);
    for pixel in image.as_chunks::<IMAGE_CHANNELS>().0 {
        push(&mut out, *pixel);
    }
    out
}

/// Pixel-major latents as the RGBA bytes of their preview.
pub(crate) fn preview(latents: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(latents.len() / LATENT_CHANNELS * RGBA);
    for latent in latents.as_chunks::<LATENT_CHANNELS>().0 {
        let colour = PREVIEW.map(|weights| {
            let (weights, bias) = weights.split_at(LATENT_CHANNELS);
            let products = weights
                .iter()
                .zip(latent)
                .map(|(weight, value)| weight * value);
            products.sum::<f32>() + bias[0]
        });
        push(&mut out, colour);
    }
    out
}

fn push(out: &mut Vec<u8>, colour: [f32; IMAGE_CHANNELS]) {
    let bytes = colour.map(|value| byte(value / 2.0 + 0.5));
    out.extend(bytes.into_iter().chain([u8::MAX]));
}

fn byte(intensity: f32) -> u8 {
    (intensity.clamp(0.0, 1.0) * f32::from(u8::MAX)).round() as u8
}
