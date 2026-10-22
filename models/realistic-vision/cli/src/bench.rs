//! `realistic-vision bench`: milliseconds of each stage at one size, and what a generation of
//! `STEPS` steps adds up to.

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use realistic_vision::Grid;
use realistic_vision::model::{LATENT_CHANNELS, LATENT_FACTOR};
use realistic_vision::pipeline::Stage;
use realistic_vision::session::Engine;

const CALLS: usize = 3;
const STEPS: usize = 20;
const GUIDANCE: f32 = 5.0;
const PROMPT: &str = "RAW photo, a lighthouse on a rocky coast at sunset, waves, 8k uhd, dslr";

pub fn run(engine: &mut Engine, (width, height): (usize, usize)) -> Result<()> {
    let grid = Grid {
        images: 1,
        height: height / LATENT_FACTOR,
        width: width / LATENT_FACTOR,
    };
    println!("threads {}, {width}×{height}", engine.threads());
    let latents: Vec<f32> = (0..grid.pixels() * LATENT_CHANNELS)
        .map(|at| (at as f32 * 0.37).sin())
        .collect();
    let tokens = engine.tokens(PROMPT)?.ids;
    let mut text = engine.encode_text(&engine.tokens("")?.ids)?;
    text.extend(engine.encode_text(&tokens)?);
    let both = [latents.as_slice(), &latents].concat();
    let branches = Grid { images: 2, ..grid };
    let middle = engine.schedule(STEPS).timestep(STEPS / 2);
    let mut total = 0.0;
    for &stage in Stage::ALL {
        let (ms, per_image) = match stage {
            Stage::Prompt => (time(CALLS, || engine.tokens(PROMPT))?, 2.0),
            Stage::Text => (time(CALLS, || engine.encode_text(&tokens))?, 2.0),
            Stage::Unet => (
                time(CALLS, || {
                    engine.predict_noise((&both, branches), middle, &text)
                })?,
                0.0,
            ),
            Stage::Sampler => {
                let step = time(CALLS, || {
                    engine.sample(&text, (&latents, grid), (1, GUIDANCE))
                })?;
                (step, STEPS as f64)
            }
            Stage::Decode => (time(CALLS, || engine.decode((&latents, grid)))?, 1.0),
        };
        total += ms * per_image;
        row(stage, ms, "");
    }
    println!(
        "  a generation of {STEPS} steps at guidance {GUIDANCE}: {:.1} s (the sampler's step \
         includes the UNet)",
        total / 1000.0
    );
    Ok(())
}
