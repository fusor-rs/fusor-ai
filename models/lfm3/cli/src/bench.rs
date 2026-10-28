//! `lfm3 bench`: the milliseconds of each stage on a conversation's first turn (its first image
//! seen, its first clip heard, the prompt read from its first row, one token of the reply, one
//! frame spoken), and of replying to it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use kernels::numpy::argmax;
use lfm3::audio;
use lfm3::media::{self, IMAGE};
use lfm3::pipeline::Stage;
use lfm3::prompt::{Opening, Turn};
use lfm3::session::{Engine, Message};
use lfm3::{Patches, Pixels};

const CALLS: usize = 3;

/// The turn benched: its message, images and clips.
pub struct Inputs<'inputs> {
    pub message: &'inputs Message,
    pub images: &'inputs [Pixels],
    pub clips: &'inputs [Vec<f32>],
}

pub fn run(engine: &mut Engine, inputs: &Inputs) -> Result<()> {
    let patches = inputs
        .images
        .iter()
        .map(media::patches)
        .collect::<Result<Vec<_>, _>>()?;
    let mut tokens = Vec::new();
    for image in &patches {
        tokens.extend(see(engine, image)?);
    }
    for clip in inputs.clips {
        tokens.extend(hear(engine, clip)?);
    }
    let message = inputs.message;
    let turn = Turn {
        images: patches
            .iter()
            .map(|image| IMAGE.tokens(image.grid.patches()))
            .collect(),
        audio: inputs
            .clips
            .iter()
            .map(|clip| audio::tokens(clip.len()))
            .collect(),
        text: message.text.clone(),
    };
    let opening = Opening {
        mode: message.mode,
        thinking: message.thinking,
    };
    let prompt = engine.prompt(&message.system, &[], &turn, opening)?;
    let prefill = engine.prefill(&prompt, &tokens);
    let first = argmax(&prefill.logits) as u32;
    println!(
        "threads {}, {} rows a prompt",
        engine.threads(),
        prompt.len()
    );
    for &stage in Stage::ALL {
        let milliseconds = match (stage, patches.first(), inputs.clips.first()) {
            (Stage::See, None, _) | (Stage::Hear, _, None) => continue,
            (Stage::See, Some(image), _) => time(CALLS, || see(engine, image))?,
            (Stage::Hear, _, Some(clip)) => time(CALLS, || hear(engine, clip))?,
            (Stage::Prefill, ..) => timed(|| engine.prefill(&prompt, &tokens)),
            (Stage::Decode, ..) => timed(|| engine.force(&prompt, &[first])),
            (Stage::Speak, ..) if !message.mode.speaks() => continue,
            (Stage::Speak, ..) => time(CALLS, || speak(engine, &prefill.hidden))?,
        };
        row(stage, milliseconds, "");
    }
    whole(engine, inputs)
}

/// An image's tokens, through the tower and the projector.
fn see(engine: &mut Engine, image: &Patches) -> Result<Vec<f32>, lfm3::Error> {
    let embedded = engine.embed(image)?;
    let encoded = engine.encode(&embedded, image.grid)?;
    engine.project(&encoded, image.grid)
}

/// A clip's tokens, through its features, the encoder and the projector.
fn hear(engine: &mut Engine, clip: &[f32]) -> Result<Vec<f32>, lfm3::Error> {
    let features = engine.audio_features(clip)?;
    let encoded = engine.encode_audio(&features)?;
    engine.project_audio(&encoded)
}

/// A frame drawn from `hidden`, and its samples.
fn speak(engine: &mut Engine, hidden: &[f32]) -> Result<Vec<f32>, lfm3::Error> {
    let frame = engine.frame(hidden, 0)?;
    if audio::ends(&frame) {
        return Ok(Vec::new());
    }
    engine.speak(&[frame])
}

/// The turn replied to, whole, in a new conversation.
fn whole(engine: &mut Engine, inputs: &Inputs) -> Result<()> {
    engine.reset();
    let start = Instant::now();
    engine.start(inputs.message, inputs.images, inputs.clips)?;
    let mut progress = engine.step()?;
    while !progress.done {
        progress = engine.step()?;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "  reply whole  {:9.1} ms, {} tokens and {} frames ({:.1} rows/s)",
        seconds * 1000.0,
        progress.tokens,
        progress.frames,
        (progress.tokens + progress.frames) as f64 / seconds
    );
    Ok(())
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, std::convert::Infallible>(call()));
    milliseconds
}
