//! `phonon bench`: milliseconds per call of each stage on this machine (the voice and text
//! prompt once per chunk, the rest per frame), and the per-frame total they add up to.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use phonon::model::LATENT;
use phonon::pipeline::Stage;
use phonon::prompt::VOICES;
use phonon::session::Engine;
use phonon::{FRAME_SAMPLES, SAMPLE_RATE};

use crate::parity::SENTENCE_A;

const CALLS: usize = 20;

pub fn run(engine: &mut Engine) -> Result<()> {
    let tokens = engine.chunks(SENTENCE_A)?.swap_remove(0).tokens;
    println!("threads {}", engine.threads());
    engine.prompt(VOICES[0].0, &tokens)?;
    let start = Instant::now();
    engine.prompt(VOICES[0].0, &tokens)?;
    println!(
        "  voice + text prompt ({} tokens) {:.2} ms per chunk",
        tokens.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );
    let latent: Vec<f32> = (0..LATENT)
        .map(|i| ((i * 7919 % 113) as f32 / 56.0) - 1.0)
        .collect();
    let mut per_frame = 0.0;
    for &stage in Stage::ALL {
        let ms = match stage {
            Stage::Voice | Stage::Text => continue,
            Stage::Flow => time(CALLS, || {
                anyhow::Ok(engine.flow_step(Some(&latent), &latent))
            })?,
            Stage::Decode => {
                let latents = latent.repeat(CALLS);
                let start = Instant::now();
                drop(engine.decode(&latents));
                start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64
            }
        };
        row(stage, ms, "/frame");
        per_frame += ms;
    }
    let frame_ms = 1000.0 * FRAME_SAMPLES as f64 / f64::from(SAMPLE_RATE);
    println!(
        "  frame        {per_frame:7.2} ms → RTF {:.3}",
        per_frame / frame_ms
    );
    Ok(())
}
