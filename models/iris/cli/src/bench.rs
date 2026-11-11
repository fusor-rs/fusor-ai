//! `iris bench`: milliseconds of each stage, and what a Turbo image adds up to.
//!
//! The text adapter runs once per image, the transformer's pass once per step. A Turbo image is
//! timed at 1024 × 1024, its default size, where a pass is also a restoration's tile.

use std::path::Path;

use anyhow::Result;
use bundle::Engine as _;
use harness::bench::{row, time};
use iris::Size;
use iris::model::{IMAGE_CHANNELS, TEXT_BUNDLE, TILE, TRAINING_TIMESTEPS, TURBO_LEVELS, Variant};
use iris::pipeline::Stage;
use iris::session::{Branch, ImageEngine, TextEngine};

const CALLS: usize = 2;
const PROMPT: &str = "A red fox sleeping in fresh snow, golden hour";

pub fn run(bundle: &Path) -> Result<()> {
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!("threads {}", text.threads());
    let tokens = time(CALLS, || text.tokens(PROMPT))?;
    let mut states = Vec::new();
    let encoded = time(CALLS, || {
        states = text.encode(PROMPT)?;
        Ok::<_, iris::Error>(())
    })?;
    drop(text);
    let mut engine: ImageEngine = harness::load(&bundle.join(Variant::Turbo.name()))?;
    let adapter = time(CALLS, || engine.condition(Branch::Prompt, &states))?;
    let middle = TURBO_LEVELS[TURBO_LEVELS.len() / 2] * TRAINING_TIMESTEPS as f32;
    let pass = time(CALLS, || predict(&mut engine, TILE, middle))?;
    for (stage, milliseconds, unit) in [
        (Stage::Prompt, tokens, ""),
        (Stage::Text, encoded, ""),
        (Stage::Adapter, adapter, ""),
        (Stage::Dit, pass, " per step"),
        (Stage::Restore, pass, " per tile"),
    ] {
        row(stage, milliseconds, unit);
    }
    let steps = (TURBO_LEVELS.len() - 1) as f64;
    println!(
        "  a Turbo image at {TILE}×{TILE}: {:.1} s (text, adapter and {steps} steps)",
        (encoded + adapter + steps * pass) / 1000.0
    );
    Ok(())
}

/// One transformer pass over a `side × side` pattern.
fn predict(engine: &mut ImageEngine, side: usize, timestep: f32) -> Result<Vec<f32>, iris::Error> {
    let image: Vec<f32> = (0..side * side * IMAGE_CHANNELS)
        .map(|at| (at as f32 * 0.37).sin())
        .collect();
    let size = Size {
        width: side,
        height: side,
    };
    engine.predict((&image, size), timestep, Branch::Prompt)
}
