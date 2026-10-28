//! `qwen3-tts bench`: milliseconds each stage of a request takes on this machine (the speaker
//! embedding, the clip's frames, the prompt and the likeness once; the frame on average over
//! `FRAMES`), and the real-time factor the frame stage gives.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::row;
use qwen3_tts::model::{FRAME_SAMPLES, Language, SAMPLE_RATE};
use qwen3_tts::pipeline::Stage;
use qwen3_tts::session::{Engine, Request};

/// The frames timed.
const FRAMES: usize = 20;
const SENTENCE: &str = "The morning train was late again, so we walked along the river.";

pub fn run(engine: &mut Engine, voice: &[f32], reference_text: &str) -> Result<()> {
    println!("threads {}", engine.threads());
    let request = Request {
        text: SENTENCE.into(),
        language: Language::English,
        reference_text: reference_text.into(),
        seed: 0,
        max_frames: Some(FRAMES + 1),
    };
    let mut frame_ms = 0.0;
    let mut frames = 0;
    engine.start(&request, voice)?;
    loop {
        let start = Instant::now();
        let progress = engine.step()?;
        let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
        let stage = Stage::ALL
            .iter()
            .copied()
            .find(|stage| stage.name() == progress.stage)
            .expect("a step reports one of the stages");
        match stage {
            Stage::Speak => {
                frame_ms += milliseconds;
                frames += 1;
            }
            _ => row(stage, milliseconds, ""),
        }
        if progress.done {
            break;
        }
    }
    let per_frame = frame_ms / f64::from(frames);
    println!("  {:12} {per_frame:9.2} ms/frame", Stage::Speak.name());
    let frame_seconds = FRAME_SAMPLES as f64 / f64::from(SAMPLE_RATE);
    println!("  RTF {:.3}", per_frame / 1000.0 / frame_seconds);
    Ok(())
}
