//! `pjit-21k parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its tokens, its text states, its image at the first
//! step), so errors do not compound from one stage to the next, but for the sampler, which runs
//! every step from the reference's starting noise. Tensors are compared by relative L2
//! (`‖ours − ref‖ / ‖ref‖`); the tokens must be the reference's, one for one. The text engine is
//! dropped before the image engine loads: the two bundles do not fit memory together.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use compile::Checkpoint;
use harness::parity::{compare, pixel_major};
use pjit_21k::Size;
use pjit_21k::model::{CHANNELS, IMAGE_BUNDLE, TEXT_BUNDLE};
use pjit_21k::session::{Branch, ImageEngine, TextEngine, knots, timestep};

/// The reference's tensors for one generation: its two prompts (UTF-8) and their tokens, Qwen3.5's
/// states of each, the starting noise, the DiT's input and its clean image for each prompt at the
/// first step, the knots and timesteps, and the last image (channel-major, as the reference holds
/// them).
struct Goldens {
    prompts: [String; 2],
    tokens: [Vec<u32>; 2],
    text: [Vec<f32>; 2],
    image: Size,
    noise: Vec<f32>,
    seen: Vec<f32>,
    times: Vec<f32>,
    timesteps: Vec<f32>,
    guidance: f32,
    prediction: Vec<f32>,
    sample: Vec<f32>,
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
        let &[_, height, width] = file.shape("noise")? else {
            bail!("noise: [channels, height, width] expected");
        };
        Ok(Self {
            prompts: [text("prompt")?, text("negative")?],
            tokens: [tokens("prompt_tokens")?, tokens("negative_tokens")?],
            text: [file.f32("prompt_text")?, file.f32("negative_text")?],
            image: Size { width, height },
            noise: file.f32("noise")?,
            seen: file.f32("seen")?,
            times: file.f32("times")?,
            timesteps: file.f32("timesteps")?,
            guidance: file.f32("guidance")?[0],
            prediction: file.f32("prediction")?,
            sample: file.f32("sample")?,
        })
    }
}

pub fn run(bundle: &Path, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    println!(
        "{}×{}, {} steps, guidance {}",
        goldens.image.width,
        goldens.image.height,
        goldens.timesteps.len(),
        goldens.guidance
    );
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!("{}", prompts(&text, &goldens)?);
    println!("{}", encode(&mut text, &goldens)?);
    drop(text);
    let mut engine: ImageEngine = harness::load(&bundle.join(IMAGE_BUNDLE))?;
    println!("{}", dit(&mut engine, &goldens)?);
    println!("{}", sampler(&mut engine, &goldens)?);
    Ok(())
}

/// Both prompts' tokens, which must be the reference's.
fn prompts(engine: &TextEngine, goldens: &Goldens) -> Result<String> {
    for (text, reference) in goldens.prompts.iter().zip(&goldens.tokens) {
        let ours = engine.tokens(text)?.ids;
        ensure!(
            &ours == reference,
            "prompt {text:?}: tokens {ours:?}, reference {reference:?}"
        );
    }
    Ok("prompt: both prompts' tokens identical".into())
}

/// Qwen3.5-4B's states of the reference's tokens, the caption's then the negative prompt's.
fn encode(engine: &mut TextEngine, goldens: &Goldens) -> Result<String> {
    let [prompt, negative] = &goldens.tokens;
    let ours = [
        engine.encode_tokens(prompt)?,
        engine.encode_tokens(negative)?,
    ]
    .concat();
    let reference = goldens.text.concat();
    Ok(compare("text", &ours, &reference)?)
}

/// The DiT's clean image for the reference's input at its first timestep under each prompt.
fn dit(engine: &mut ImageEngine, goldens: &Goldens) -> Result<String> {
    engine.condition(Branch::Prompt, &goldens.text[0])?;
    engine.condition(Branch::Negative, &goldens.text[1])?;
    let image = goldens.image;
    let seen = pixel_major(&goldens.seen, (CHANNELS, image.pixels()));
    let timestep = goldens.timesteps[0];
    let at = (seen.as_slice(), image);
    let prompt = engine.predict(at, timestep, Branch::Prompt)?;
    let negative = engine.predict(at, timestep, Branch::Negative)?;
    let reference = pixel_major(&goldens.prediction, (CHANNELS, image.pixels()));
    Ok(compare("dit", &[prompt, negative].concat(), &reference)?)
}

/// The knots and timesteps, then every step from the reference's noise to its last image.
fn sampler(engine: &mut ImageEngine, goldens: &Goldens) -> Result<String> {
    let steps = goldens.timesteps.len();
    let times = knots(steps);
    ensure!(
        times == goldens.times,
        "knots {times:?}, reference {:?}",
        goldens.times
    );
    let timesteps: Vec<f32> = times[..steps].iter().map(|&time| timestep(time)).collect();
    ensure!(
        timesteps == goldens.timesteps,
        "timesteps {timesteps:?}, reference {:?}",
        goldens.timesteps
    );
    let image = goldens.image;
    let noise = pixel_major(&goldens.noise, (CHANNELS, image.pixels()));
    let ours = engine.sample((&noise, image), (steps, goldens.guidance))?;
    let reference = pixel_major(&goldens.sample, (CHANNELS, image.pixels()));
    Ok(format!(
        "sampler: knots and timesteps identical; {}",
        compare("sample", &ours, &reference)?
    ))
}
