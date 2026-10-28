//! The speech tokenizer's tensors (`speech_tokenizer/model.safetensors`): the encoder's
//! (`encoder`), the decoder's (`decoder`) and the vocoder's (`vocoder`).

mod decoder;
mod encoder;
mod vocoder;

use compile::{Checkpoint, Error, Tensors, scale_rows};

use crate::Profile;

/// The floor a codebook's cluster usage is clamped to when the centroids are made.
const USAGE_FLOOR: f32 = 1e-5;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    encoder::write(checkpoint, tensors)?;
    decoder::write(checkpoint, tensors)?;
    vocoder::write(checkpoint, tensors)
}

/// A codebook's centroids, `[entries, channels]` row-major: its running sums over its cluster
/// usage.
fn centroids(checkpoint: &Checkpoint, (sums, usage): (&str, &str)) -> Result<Vec<f32>, Error> {
    let mut centroids = checkpoint.f32(sums)?;
    let usage: Vec<f32> = checkpoint
        .f32(usage)?
        .into_iter()
        .map(|used| used.max(USAGE_FLOOR).recip())
        .collect();
    scale_rows(&mut centroids, &usage);
    Ok(centroids)
}

/// The weight `name` with its rows scaled by the layer scale `scale`.
fn scaled(checkpoint: &Checkpoint, name: &str, scale: &str) -> Result<Vec<f32>, Error> {
    let mut weight = checkpoint.f32(name)?;
    scale_rows(&mut weight, &checkpoint.f32(scale)?);
    Ok(weight)
}
