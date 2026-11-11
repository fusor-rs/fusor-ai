//! `wan2-1 bench`: milliseconds of each stage at one size, and what a generation of `STEPS`
//! steps adds up to.
//!
//! The prompts' conditioning of the transformer (about 0.3 TFLOP, 0.2 % of a generation) runs
//! before the timings and is not part of them.

use std::path::Path;

use anyhow::Result;
use bundle::Engine as _;
use harness::bench::{row, time};
use wan2_1::model::{
    DEFAULT_GUIDANCE, DEFAULT_SHIFT, LATENT_CHANNELS, NEGATIVE, TEXT_BUNDLE, VIDEO_BUNDLE,
};
use wan2_1::pipeline::Stage;
use wan2_1::session::{Branch, Request, TextEngine, VideoEngine};

const CALLS: usize = 2;
const STEPS: usize = 20;
const PROMPT: &str =
    "A red fox trots through fresh snow in a birch forest at golden hour, cinematic";

pub fn run(bundle: &Path, (width, height, frames): (usize, usize, usize)) -> Result<()> {
    let video = Request {
        width,
        height,
        frames,
        steps: STEPS,
        guidance: DEFAULT_GUIDANCE,
        shift: DEFAULT_SHIFT,
        seed: 0,
    }
    .video()?;
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!(
        "threads {}, {width}×{height}, {frames} frames",
        text.threads()
    );
    let tokens = time(CALLS, || {
        Ok::<_, wan2_1::Error>((text.tokens(PROMPT)?, text.tokens(NEGATIVE)?))
    })?;
    let mut states = [Vec::new(), Vec::new()];
    let encoded = time(CALLS, || {
        states = [text.encode(PROMPT)?, text.encode(NEGATIVE)?];
        Ok::<_, wan2_1::Error>(())
    })?;
    drop(text);
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    let [prompt, negative] = &states;
    engine.condition(Branch::Prompt, prompt)?;
    engine.condition(Branch::Negative, negative)?;
    let latents: Vec<f32> = (0..video.pixels() * LATENT_CHANNELS)
        .map(|at| (at as f32 * 0.37).sin())
        .collect();
    let middle = engine.schedule(STEPS, DEFAULT_SHIFT).timestep(STEPS / 2);
    let dit = time(CALLS, || {
        engine.predict((&latents, video), middle, Branch::Prompt)
    })?;
    let sampler = time(CALLS, || {
        engine.sample((&latents, video), (1, DEFAULT_GUIDANCE, DEFAULT_SHIFT))
    })?;
    let decode = time(CALLS, || engine.decode((&latents, video)))?;
    let rows = [
        (Stage::Prompt, tokens),
        (Stage::Text, encoded),
        (Stage::Dit, dit),
        (Stage::Sampler, sampler),
        (Stage::Decode, decode),
    ];
    for (stage, milliseconds) in rows {
        row(stage, milliseconds, "");
    }
    let total = encoded + STEPS as f64 * sampler + decode;
    println!(
        "  a generation of {STEPS} steps at guidance {DEFAULT_GUIDANCE}: {:.1} s (text, the \
         sampler's steps and decoding)",
        total / 1000.0
    );
    Ok(())
}
