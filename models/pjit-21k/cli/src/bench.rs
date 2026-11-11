//! `pjit-21k bench`: milliseconds of each stage at one size, and what a generation of
//! `DEFAULT_STEPS` steps adds up to.
//!
//! The branches' conditioning of the transformer (two small products per request) runs before the
//! timings and is not part of them.

use std::path::Path;

use anyhow::Result;
use bundle::Engine as _;
use harness::bench::{row, time};
use pjit_21k::Size;
use pjit_21k::model::{
    CHANNELS, DEFAULT_GUIDANCE, DEFAULT_STEPS, IMAGE_BUNDLE, NEGATIVE, TEXT_BUNDLE,
};
use pjit_21k::pipeline::Stage;
use pjit_21k::session::{Branch, ImageEngine, TextEngine, knots, timestep};

const CALLS: usize = 2;
const CAPTION: &str = "Close-up portrait of an elderly Middle Eastern woman looking calmly \
    into the lens, her face filling the tall frame. A cream headscarf and the collar of a dark \
    green cardigan frame her chin. Soft overcast daylight smooths the fine lines of her skin, \
    and the olive grove behind melts into silvery blur.";

pub fn run(bundle: &Path, image: Size) -> Result<()> {
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!(
        "threads {}, {}×{}",
        text.threads(),
        image.width,
        image.height
    );
    let tokens = time(CALLS, || {
        Ok::<_, pjit_21k::Error>((text.tokens(CAPTION)?, text.tokens(NEGATIVE)?))
    })?;
    let mut states = [Vec::new(), Vec::new()];
    let encoded = time(CALLS, || {
        states = [text.encode(CAPTION)?, text.encode(NEGATIVE)?];
        Ok::<_, pjit_21k::Error>(())
    })?;
    drop(text);
    let mut engine: ImageEngine = harness::load(&bundle.join(IMAGE_BUNDLE))?;
    let [prompt, negative] = &states;
    engine.condition(Branch::Prompt, prompt)?;
    engine.condition(Branch::Negative, negative)?;
    let pixels: Vec<f32> = (0..image.pixels() * CHANNELS)
        .map(|at| (at as f32 * 0.37).sin())
        .collect();
    let middle = timestep(knots(DEFAULT_STEPS)[DEFAULT_STEPS / 2]);
    let dit = time(CALLS, || {
        engine.predict((&pixels, image), middle, Branch::Prompt)
    })?;
    let sampler = time(CALLS, || {
        engine.sample((&pixels, image), (1, DEFAULT_GUIDANCE))
    })?;
    let rows = [
        (Stage::Prompt, tokens),
        (Stage::Text, encoded),
        (Stage::Dit, dit),
        (Stage::Sampler, sampler),
    ];
    for (stage, milliseconds) in rows {
        row(stage, milliseconds, "");
    }
    let total = encoded + DEFAULT_STEPS as f64 * sampler;
    println!(
        "  a generation of {DEFAULT_STEPS} steps at guidance {DEFAULT_GUIDANCE}: {:.1} s (the \
         text, then the sampler's steps)",
        total / 1000.0
    );
    Ok(())
}
