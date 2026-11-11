//! `wan2-1 parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its tokens, its text states, its noise, its last
//! latents), so errors do not compound from one stage to the next, but for the sampler, which
//! runs every step from the reference's starting noise. Tensors are compared by relative L2
//! (`‖ours − ref‖ / ‖ref‖`); the tokens must be the reference's, one for one. The text engine
//! is dropped before the video engine loads: the two bundles do not fit memory together.

use std::path::Path;

use anyhow::{Result, ensure};
use compile::Checkpoint;
use harness::parity::{compare, pixel_major};
use wan2_1::Video;
use wan2_1::model::{IMAGE_CHANNELS, LATENT_CHANNELS, TEXT_BUNDLE, VIDEO_BUNDLE};
use wan2_1::session::{Branch, TextEngine, VideoEngine};

/// The reference's tensors for one generation: its two prompts (UTF-8) and their tokens, umT5's
/// states of each, the starting noise, the schedule, the transformer's velocity for the prompt
/// and the negative prompt at the first step, the last latents and the video (channel-major, as
/// the reference holds them).
struct Goldens {
    prompts: [String; 2],
    tokens: [Vec<u32>; 2],
    text: [Vec<f32>; 2],
    noise: Vec<f32>,
    video: Video,
    timesteps: Vec<f32>,
    levels: Vec<f32>,
    guidance: f32,
    shift: f32,
    velocity: Vec<f32>,
    latents: Vec<f32>,
    frames: Vec<f32>,
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
        let &[_, frames, height, width] = file.shape("noise")? else {
            anyhow::bail!("noise: [channels, frames, height, width] expected");
        };
        Ok(Self {
            prompts: [text("prompt")?, text("negative")?],
            tokens: [tokens("prompt_tokens")?, tokens("negative_tokens")?],
            text: [file.f32("prompt_text")?, file.f32("negative_text")?],
            noise: file.f32("noise")?,
            video: Video {
                frames,
                height,
                width,
            },
            timesteps: file.f32("timesteps")?,
            levels: file.f32("sigmas")?,
            guidance: file.f32("guidance")?[0],
            shift: file.f32("shift")?[0],
            velocity: file.f32("velocity")?,
            latents: file.f32("latents")?,
            frames: file.f32("video")?,
        })
    }
}

pub fn run(bundle: &Path, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let video = goldens.video;
    println!(
        "{}×{} latents, {} latent frames, {} steps, guidance {}, shift {}",
        video.width,
        video.height,
        video.frames,
        goldens.timesteps.len(),
        goldens.guidance,
        goldens.shift
    );
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!("{}", prompts(&text, &goldens)?);
    println!("{}", encode(&mut text, &goldens)?);
    drop(text);
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    println!("{}", dit(&mut engine, &goldens)?);
    println!("{}", sampler(&mut engine, &goldens)?);
    println!("{}", decode(&mut engine, &goldens)?);
    Ok(())
}

/// Both prompts' tokens, which must be the reference's.
fn prompts(engine: &TextEngine, goldens: &Goldens) -> Result<String> {
    for (text, reference) in goldens.prompts.iter().zip(&goldens.tokens) {
        let ours = engine.tokens(text)?;
        ensure!(
            &ours == reference,
            "prompt {text:?}: tokens {ours:?}, reference {reference:?}"
        );
    }
    Ok("prompt: both prompts' tokens identical".into())
}

/// umT5's states of the reference's tokens, the prompt's then the negative prompt's.
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

/// The transformer's velocity for the reference's noise at its first timestep under each prompt.
fn dit(engine: &mut VideoEngine, goldens: &Goldens) -> Result<String> {
    engine.condition(Branch::Prompt, &goldens.text[0])?;
    engine.condition(Branch::Negative, &goldens.text[1])?;
    let video = goldens.video;
    let noise = pixel_major(&goldens.noise, (LATENT_CHANNELS, video.pixels()));
    let timestep = goldens.timesteps[0];
    let at = (noise.as_slice(), video);
    let prompt = engine.predict(at, timestep, Branch::Prompt)?;
    let negative = engine.predict(at, timestep, Branch::Negative)?;
    let reference = pixel_major(&goldens.velocity, (LATENT_CHANNELS, video.pixels()));
    Ok(compare("dit", &[prompt, negative].concat(), &reference)?)
}

/// The schedule, then every step from the reference's noise to the last latents.
fn sampler(engine: &mut VideoEngine, goldens: &Goldens) -> Result<String> {
    let steps = goldens.timesteps.len();
    let schedule = engine.schedule(steps, goldens.shift);
    let timesteps: Vec<f32> = (0..steps).map(|step| schedule.timestep(step)).collect();
    ensure!(
        timesteps == goldens.timesteps,
        "timesteps {timesteps:?}, reference {:?}",
        goldens.timesteps
    );
    let levels = compare("levels", schedule.levels(), &goldens.levels)?;
    let video = goldens.video;
    let noise = pixel_major(&goldens.noise, (LATENT_CHANNELS, video.pixels()));
    let at = (steps, goldens.guidance, goldens.shift);
    let ours = engine.sample((&noise, video), at)?;
    let reference = pixel_major(&goldens.latents, (LATENT_CHANNELS, video.pixels()));
    Ok(format!(
        "sampler: timesteps identical; {levels}; {}",
        compare("latents", &ours, &reference)?
    ))
}

/// The video of the reference's last latents.
fn decode(engine: &mut VideoEngine, goldens: &Goldens) -> Result<String> {
    let video = goldens.video;
    let latents = pixel_major(&goldens.latents, (LATENT_CHANNELS, video.pixels()));
    let ours = engine.decode((&latents, video))?;
    let reference = pixel_major(
        &goldens.frames,
        (IMAGE_CHANNELS, ours.len() / IMAGE_CHANNELS),
    );
    Ok(compare("decode", &ours, &reference)?)
}
