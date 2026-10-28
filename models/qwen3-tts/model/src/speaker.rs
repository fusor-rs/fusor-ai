//! The speaker encoder: a voice clip's log-mel features as the reference computes them (24 kHz
//! samples reflect-padded by `REFLECT` on both sides, the mel magnitudes `√(power + 1e-9)` of
//! `media::mel`'s filterbank, floored and logged) through `blocks::ecapa` into the voice's
//! 1,024-value embedding.
//!
//! Tensors: `blocks::ecapa`'s under `speaker`.

use blocks::ecapa::{self, Buffers, Ecapa};
use bundle::{Load, Store};
use media::mel::{self, MelSpectrogram};
use team::Team;

use crate::Error;
use crate::model::SAMPLE_RATE;

pub const MEL: mel::Shape = mel::Shape {
    sample_rate: SAMPLE_RATE,
    fft: 1024,
    hop: 256,
    mels: 128,
};
/// Samples reflected past each end of the clip: `(fft − hop) / 2`.
const REFLECT: usize = 384;
const MAGNITUDE_FLOOR: f32 = 1e-9;
const LOG_FLOOR: f32 = 1e-5;

pub const ECAPA: ecapa::Shape = ecapa::Shape {
    mels: MEL.mels,
    channels: [512, 512, 512, 512, 1536],
    kernels: [5, 3, 3, 3, 1],
    dilations: [1, 2, 3, 4, 1],
    attention: 128,
    scale: 8,
    squeeze: 128,
    embedding: 1024,
};

pub(crate) struct Speaker {
    spectrogram: MelSpectrogram,
    ecapa: Ecapa,
}

impl Speaker {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            spectrogram: MelSpectrogram::new(MEL),
            ecapa: Ecapa::load(store, "speaker", ECAPA)?,
        })
    }

    /// The log-mel features of `samples`, `frames × mels`.
    pub(crate) fn features(&self, samples: &[f32]) -> Vec<f32> {
        let last = samples.len() - 1;
        let left = samples[1..=REFLECT].iter().rev();
        let right = samples[last - REFLECT..last].iter().rev();
        let padded: Vec<f32> = left.chain(samples).chain(right).copied().collect();
        let mut features = Vec::new();
        self.spectrogram
            .magnitude(&padded, MAGNITUDE_FLOOR, &mut features);
        for value in &mut features {
            *value = value.max(LOG_FLOOR).ln();
        }
        features
    }

    /// The embedding of `features` (`features`' output).
    pub(crate) fn embed(&self, team: &Team, features: &[f32]) -> Vec<f32> {
        let frames = features.len() / MEL.mels;
        let buffers = Buffers::new(&ECAPA, frames, team.threads());
        team.run(|ctx| self.ecapa.embed(ctx, &buffers, features, frames));
        buffers.embedding.read().to_vec()
    }
}
