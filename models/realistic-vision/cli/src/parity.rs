//! `realistic-vision parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own input (its tokens, its text states, its noise, its last
//! latents), so errors do not compound from one stage to the next, but for the sampler, which
//! runs every step from the reference's starting noise. Tensors are compared by relative L2
//! (`‖ours − ref‖ / ‖ref‖`); the tokens must be the reference's, one for one.

use std::path::Path;

use anyhow::{Result, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use realistic_vision::Grid;
use realistic_vision::model::{IMAGE_CHANNELS, LATENT_CHANNELS};
use realistic_vision::pipeline::Stage;
use realistic_vision::session::Engine;

/// The reference's tensors for one generation: its prompts (UTF-8) and their tokens, the text
/// encoder's states (the negative prompt's, then the prompt's), the starting noise, the
/// schedule, the UNet's noise for both prompts at the first step, the last latents and the
/// image (NCHW, as the reference holds them).
struct Goldens {
    prompts: [String; 2],
    tokens: [Vec<u32>; 2],
    text: Vec<f32>,
    noise: Vec<f32>,
    grid: Grid,
    timesteps: Vec<f32>,
    levels: Vec<f32>,
    guidance: f32,
    predicted: Vec<f32>,
    latents: Vec<f32>,
    image: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let text = |name: &str| Ok::<_, anyhow::Error>(String::from_utf8(file.u8(name)?)?);
        let tokens = |name: &str| -> Result<Vec<u32>> {
            let ids = file.i64(name)?;
            Ok(ids
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?)
        };
        let &[_, _, height, width] = file.shape("noise")? else {
            anyhow::bail!("noise: [1, channels, height, width] expected");
        };
        Ok(Self {
            prompts: [text("negative")?, text("prompt")?],
            tokens: [tokens("negative_tokens")?, tokens("prompt_tokens")?],
            text: file.f32("text")?,
            noise: file.f32("noise")?,
            grid: Grid {
                images: 1,
                height,
                width,
            },
            timesteps: file.f32("timesteps")?,
            levels: file.f32("sigmas")?,
            guidance: file.f32("guidance")?[0],
            predicted: file.f32("predicted")?,
            latents: file.f32("latents")?,
            image: file.f32("image")?,
        })
    }
}

/// Channel-major images (`[images, channels, pixels]`) as pixel-major rows.
fn pixel_major(values: &[f32], (channels, pixels): (usize, usize)) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for (at, value) in values.iter().enumerate() {
        let (plane, pixel) = (at / pixels, at % pixels);
        let (image, channel) = (plane / channels, plane % channels);
        out[(image * pixels + pixel) * channels + channel] = *value;
    }
    out
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let steps = goldens.timesteps.len();
    let grid = goldens.grid;
    println!(
        "{}×{} latents, {steps} steps, guidance {}",
        grid.width, grid.height, goldens.guidance
    );
    let latent = (LATENT_CHANNELS, grid.area());
    let noise = pixel_major(&goldens.noise, latent);
    for &stage in Stage::ALL {
        let report = match stage {
            Stage::Prompt => prompts(engine, &goldens)?,
            Stage::Text => {
                let negative = engine.encode_text(&goldens.tokens[0])?;
                let ours = [negative, engine.encode_text(&goldens.tokens[1])?].concat();
                compare("text", &ours, &goldens.text)?
            }
            Stage::Unet => {
                let both = [noise.as_slice(), &noise].concat();
                let branches = Grid { images: 2, ..grid };
                let timestep = goldens.timesteps[0];
                let ours = engine.predict_noise((&both, branches), timestep, &goldens.text)?;
                let reference = pixel_major(&goldens.predicted, latent);
                compare("unet", &ours, &reference)?
            }
            Stage::Sampler => sampler(engine, &goldens, &noise)?,
            Stage::Decode => {
                let latents = pixel_major(&goldens.latents, latent);
                let ours = engine.decode((&latents, grid))?;
                let image = (IMAGE_CHANNELS, ours.len() / IMAGE_CHANNELS);
                compare("decode", &ours, &pixel_major(&goldens.image, image))?
            }
        };
        println!("{report}");
    }
    Ok(())
}

/// Both prompts' tokens, which must be the reference's.
fn prompts(engine: &Engine, goldens: &Goldens) -> Result<String> {
    for (text, reference) in goldens.prompts.iter().zip(&goldens.tokens) {
        let ours = engine.tokens(text)?.ids;
        ensure!(
            &ours == reference,
            "prompt {text:?}: tokens {ours:?}, reference {reference:?}"
        );
    }
    Ok("prompt: both prompts' tokens identical".into())
}

/// The schedule, then every step from the reference's noise to the last latents.
fn sampler(engine: &mut Engine, goldens: &Goldens, noise: &[f32]) -> Result<String> {
    let steps = goldens.timesteps.len();
    let schedule = engine.schedule(steps);
    let timesteps: Vec<f32> = (0..steps).map(|step| schedule.timestep(step)).collect();
    ensure!(
        timesteps == goldens.timesteps,
        "timesteps {timesteps:?}, reference {:?}",
        goldens.timesteps
    );
    let levels = compare("levels", schedule.levels(), &goldens.levels)?;
    let at = (noise, goldens.grid);
    let ours = engine.sample(&goldens.text, at, (steps, goldens.guidance))?;
    let reference = pixel_major(&goldens.latents, (LATENT_CHANNELS, goldens.grid.area()));
    Ok(format!(
        "sampler: timesteps identical; {levels}; {}",
        compare("latents", &ours, &reference)?
    ))
}
