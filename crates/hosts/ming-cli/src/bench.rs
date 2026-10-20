//! `ming bench`: milliseconds per call of each component on this machine (the AR loop's
//! per-frame work), and the frame total they add up to.

use std::time::Instant;

use anyhow::Result;
use ming::dit::LATENT;
use ming::prompt::Voice;
use ming::session::{Engine, Preset};
use ming::vocoder::FRAME_SAMPLES;

const SENTENCE: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";
const CALLS: usize = 20;

fn time(mut f: impl FnMut()) -> f64 {
    f();
    let start = Instant::now();
    for _ in 0..CALLS {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64
}

pub fn run(engine: &mut Engine) -> Result<()> {
    let prompt = engine
        .prompter()
        .build(SENTENCE, &Voice::Builtin("US_James".into()), None)?;
    let start = Instant::now();
    let z = engine.prefill(&prompt);
    let prefill = start.elapsed().as_secs_f64() * 1000.0;
    let noise: Vec<f32> = (0..LATENT)
        .map(|i| ((i * 7919 % 113) as f32 / 56.0) - 1.0)
        .collect();
    let history = vec![0.0; LATENT];
    let latent = engine.sample(&z, &history, &noise, Preset::Quality);
    let embedding = engine.aggregate(&latent);
    let mut position = prompt.ids.len();
    let llm = time(|| {
        engine.decode(&embedding, position);
        position += 1;
    });
    let flows: Vec<(Preset, f64)> = [Preset::Quality, Preset::Balanced, Preset::Fast]
        .into_iter()
        .map(|p| (p, time(|| drop(engine.sample(&z, &history, &noise, p)))))
        .collect();
    let aggregator = time(|| drop(engine.aggregate(&latent)));
    let latents = latent.repeat(CALLS);
    let start = Instant::now();
    let samples = engine.vocode(&latents).len();
    let vocoder = start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64;
    debug_assert!(samples >= (CALLS - 1) * FRAME_SAMPLES);

    println!(
        "threads {}: prefill {} tokens {prefill:.0} ms",
        engine.threads(),
        prompt.ids.len()
    );
    println!("  llm decode   {llm:6.2} ms/frame");
    for (preset, ms) in &flows {
        println!(
            "  flow {:9} {ms:6.2} ms/frame",
            format!("{preset:?}").to_lowercase()
        );
    }
    println!("  aggregator   {aggregator:6.2} ms/frame");
    println!("  vocoder      {vocoder:6.2} ms/frame");
    let frame_ms = 1000.0 * FRAME_SAMPLES as f64 / 44_100.0;
    for (preset, ms) in &flows {
        let total = llm + ms + aggregator + vocoder;
        println!(
            "  frame ({:8}) {total:6.2} ms → RTF {:.3}",
            format!("{preset:?}").to_lowercase(),
            total / frame_ms
        );
    }
    Ok(())
}
