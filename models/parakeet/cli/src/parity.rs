//! `parakeet parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own input (its features, its encoded frames), so errors do not
//! compound from one stage to the next. Tensors are compared by relative L2
//! (`‖ours − ref‖ / ‖ref‖`); the decoder's decisions must be the reference's, one for one.

use std::path::Path;

use anyhow::{Result, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use parakeet::Step;
use parakeet::model::MELS;
use parakeet::pipeline::Stage;
use parakeet::session::Engine;

/// The reference's tensors for one recording: its samples, the features (with the zero frame
/// past the recording), the subsampler's frames, the speech probabilities and the encoded frames
/// of the valid frames, and the decoder's decisions.
struct Goldens {
    pcm: Vec<f32>,
    features: Vec<f32>,
    subsampled: Vec<f32>,
    speech: Vec<f32>,
    encoded: Vec<f32>,
    steps: Vec<Step>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let tokens = file.i64("tokens")?;
        let durations = file.i64("durations")?;
        let steps = tokens
            .into_iter()
            .zip(durations)
            .map(|(token, frames)| {
                Ok(Step {
                    token: u32::try_from(token)?,
                    frames: u32::try_from(frames)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            pcm: file.f32("pcm")?,
            features: file.f32("features")?,
            subsampled: file.f32("subsampled")?,
            speech: file.f32("speech")?,
            encoded: file.f32("encoded")?,
            steps,
        })
    }
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let ours = engine.features(&goldens.pcm)?;
    let features = &goldens.features[..ours.len()];
    let frames = ours.len() / MELS;
    let seconds = goldens.pcm.len() as f64 / f64::from(parakeet::SAMPLE_RATE);
    println!("{seconds:.2}s of audio, {frames} mel frames");
    for &stage in Stage::ALL {
        let report = match stage {
            Stage::Features => compare("features", &ours, features)?,
            Stage::Subsampling => compare(
                "subsampling",
                &engine.subsample(features)?,
                &goldens.subsampled,
            )?,
            Stage::Speech => compare("speech", &engine.speech(features)?, &goldens.speech)?,
            Stage::Encoder => compare("encoder", &engine.encode(features)?, &goldens.encoded)?,
            Stage::Decoder => decoder(engine, &goldens)?,
        };
        println!("{report}");
    }
    Ok(())
}

/// The decisions on the reference's encoded frames.
fn decoder(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let ours = engine.decode(&goldens.encoded)?;
    let agree = ours
        .iter()
        .zip(&goldens.steps)
        .take_while(|(ours, reference)| ours == reference);
    let agreed = agree.count();
    ensure!(
        agreed == ours.len() && ours.len() == goldens.steps.len(),
        "decoder: {} decisions, reference {}; they part at {agreed}: ours {:?}, reference {:?}",
        ours.len(),
        goldens.steps.len(),
        ours.get(agreed),
        goldens.steps.get(agreed)
    );
    let tokens = ours
        .iter()
        .filter(|step| step.token as usize != parakeet::model::BLANK);
    Ok(format!(
        "decoder: {} decisions ({} tokens) identical",
        ours.len(),
        tokens.count()
    ))
}
