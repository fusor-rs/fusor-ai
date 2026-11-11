//! `motif-video parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`), for a text-to-video case and an image-to-video case.
//!
//! Each stage gets the reference's own input (its tokens, its text states, its first image, its
//! noise, its last latents), so errors do not compound from one stage to the next, but for the
//! sampler, which runs every step from the reference's starting noise. Tensors are compared by
//! relative L2 (`‖ours − ref‖ / ‖ref‖`); the tokens must be the reference's, one for one. The
//! text engine is dropped before the video engine loads: the two bundles do not fit memory
//! together.

use std::path::Path;

use anyhow::{Result, ensure};
use blocks::apg::{Guidance, Settings};
use compile::Checkpoint;
use harness::parity::{compare, pixel_major};
use motif_video::image::{first_frame, vision_pixels};
use motif_video::model::{
    GUIDANCE, IMAGE_CHANNELS, LATENT_CHANNELS, LATENT_FACTOR, TEXT_BUNDLE, VIDEO_BUNDLE,
    VISION_SIDE,
};
use motif_video::session::{Branch, TextEngine, VideoEngine, conditioning, transformer_input};
use motif_video::{Pixels, Size, Video};

const CASES: [&str; 2] = ["t2v", "i2v"];
/// An encoded frame's channels per pixel: the latents' means, then their log-variances.
const MOMENT_CHANNELS: usize = 2 * LATENT_CHANNELS;

/// The first image of an image-to-video case: the image as decoded, the video's first frame,
/// SigLIP's input and last hidden states, and the autoencoder's moments of the frame.
struct ImageGoldens {
    image: Pixels,
    frame: Vec<f32>,
    pixels: Vec<f32>,
    states: Vec<f32>,
    moments: Vec<f32>,
}

/// The reference's tensors for one generation, channel-major as the reference holds them: its two
/// prompts (UTF-8) and their tokens, T5Gemma2's states of each, the starting noise, the first
/// frame's latents (zeros without an image), the schedule, the transformer's velocity for the
/// prompt and the negative prompt at the first step and their guided combination, the last
/// latents and the video.
struct Goldens {
    name: &'static str,
    prompts: [String; 2],
    tokens: [Vec<u32>; 2],
    text: [Vec<f32>; 2],
    noise: Vec<f32>,
    condition: Vec<f32>,
    video: Video,
    timesteps: Vec<f32>,
    levels: Vec<f32>,
    guidance: f32,
    velocity: Vec<f32>,
    guided: Vec<f32>,
    latents: Vec<f32>,
    frames: Vec<f32>,
    image: Option<ImageGoldens>,
}

impl Goldens {
    fn read(name: &'static str, directory: &Path) -> Result<Self> {
        let file = Checkpoint::read(&directory.join(format!("golden_f32_{name}.safetensors")))?;
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
        let image = if file.contains("image") {
            Some(ImageGoldens::read(&file)?)
        } else {
            None
        };
        Ok(Self {
            name,
            prompts: [text("prompt")?, text("negative")?],
            tokens: [tokens("prompt_tokens")?, tokens("negative_tokens")?],
            text: [file.f32("prompt_text")?, file.f32("negative_text")?],
            noise: file.f32("noise")?,
            condition: file.f32("condition")?,
            video: Video {
                frames,
                height,
                width,
            },
            timesteps: file.f32("timesteps")?,
            levels: file.f32("sigmas")?,
            guidance: file.f32("guidance")?[0],
            velocity: file.f32("velocity")?,
            guided: file.f32("guided")?,
            latents: file.f32("latents")?,
            frames: file.f32("video")?,
            image,
        })
    }

    /// The video's frames' size in pixels.
    fn size(&self) -> Size {
        Size {
            width: self.video.width * LATENT_FACTOR,
            height: self.video.height * LATENT_FACTOR,
        }
    }

    /// The video's noise, pixel-major.
    fn noise(&self) -> Vec<f32> {
        pixel_major(&self.noise, (LATENT_CHANNELS, self.video.pixels()))
    }

    /// The last latents, pixel-major.
    fn latents(&self) -> Vec<f32> {
        pixel_major(&self.latents, (LATENT_CHANNELS, self.video.pixels()))
    }

    /// The first frame's latents, pixel-major, when the case has a first image.
    fn first_latents(&self) -> Option<Vec<f32>> {
        self.image.as_ref()?;
        let latents = pixel_major(&self.condition, (LATENT_CHANNELS, self.video.pixels()));
        let first_frame = self.video.height * self.video.width * LATENT_CHANNELS;
        Some(latents[..first_frame].to_vec())
    }
}

impl ImageGoldens {
    /// SigLIP's input, pixel-major.
    fn vision_pixels(&self) -> Vec<f32> {
        pixel_major(&self.pixels, (IMAGE_CHANNELS, VISION_SIDE * VISION_SIDE))
    }

    /// The video's first frame, pixel-major.
    fn frame(&self, size: Size) -> Vec<f32> {
        pixel_major(&self.frame, (IMAGE_CHANNELS, size.width * size.height))
    }

    fn read(file: &Checkpoint) -> Result<Self> {
        let &[height, width, channels] = file.shape("image")? else {
            anyhow::bail!("image: [height, width, channels] expected");
        };
        Ok(Self {
            image: Pixels {
                size: Size { width, height },
                channels,
                values: file.u8("image")?,
            },
            frame: file.f32("frame")?,
            pixels: file.f32("pixels")?,
            states: file.f32("image_states")?,
            moments: file.f32("moments")?,
        })
    }
}

/// The image-to-video cases and their first images.
fn images(cases: &[Goldens]) -> impl Iterator<Item = (&Goldens, &ImageGoldens)> {
    cases
        .iter()
        .filter_map(|case| Some((case, case.image.as_ref()?)))
}

pub fn run(bundle: &Path, directory: &Path) -> Result<()> {
    let cases: Vec<Goldens> = CASES
        .iter()
        .map(|name| Goldens::read(name, directory))
        .collect::<Result<_>>()?;
    for case in &cases {
        println!(
            "{}: {}×{} latents, {} latent frames, {} steps, guidance {}",
            case.name,
            case.video.width,
            case.video.height,
            case.video.frames,
            case.timesteps.len(),
            case.guidance
        );
    }
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!("{}", prompts(&text, &cases)?);
    for image in images(&cases) {
        println!("{}", frame(image)?);
        println!("{}", pixels(image)?);
    }
    println!("{}", encode(&mut text, &cases)?);
    for (_, image) in images(&cases) {
        println!("{}", vision(&mut text, image)?);
    }
    drop(text);
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    for image in images(&cases) {
        println!("{}", moments(&mut engine, image)?);
    }
    for case in &cases {
        let conditioning = condition(&mut engine, case)?;
        println!("{}", dit(&mut engine, case, &conditioning)?);
        println!("{}", guidance(case)?);
        println!("{}", sampler(&mut engine, case, &conditioning)?);
        println!("{}", decode(&mut engine, case)?);
    }
    Ok(())
}

/// Every prompt's tokens, which must be the reference's.
fn prompts(engine: &TextEngine, cases: &[Goldens]) -> Result<String> {
    for case in cases {
        for (text, reference) in case.prompts.iter().zip(&case.tokens) {
            let ours = engine.tokens(text)?;
            ensure!(
                &ours == reference,
                "prompt {text:?}: tokens {ours:?}, reference {reference:?}"
            );
        }
    }
    Ok("prompt: every prompt's tokens identical".into())
}

/// The image resized to the video's size and scaled to [−1, 1].
fn frame((case, image): (&Goldens, &ImageGoldens)) -> Result<String> {
    let size = case.size();
    let ours = first_frame(&image.image, size)?;
    Ok(compare("frame", &ours, &image.frame(size))?)
}

/// SigLIP's input from the reference's first frame.
fn pixels((case, image): (&Goldens, &ImageGoldens)) -> Result<String> {
    let size = case.size();
    let ours = vision_pixels(&image.frame(size), size);
    Ok(compare("pixels", &ours, &image.vision_pixels())?)
}

/// T5Gemma2's states of the reference's tokens, each case's prompt then negative prompt.
fn encode(engine: &mut TextEngine, cases: &[Goldens]) -> Result<String> {
    let mut ours = Vec::new();
    let mut reference = Vec::new();
    for case in cases {
        for (tokens, text) in case.tokens.iter().zip(&case.text) {
            ours.extend(engine.encode_tokens(tokens)?);
            reference.extend_from_slice(text);
        }
    }
    Ok(compare("text", &ours, &reference)?)
}

/// SigLIP's states of the reference's input pixels.
fn vision(engine: &mut TextEngine, image: &ImageGoldens) -> Result<String> {
    let ours = engine.embed_pixels(&image.vision_pixels())?;
    Ok(compare("vision", &ours, &image.states)?)
}

/// The autoencoder's moments of the reference's first frame.
fn moments(engine: &mut VideoEngine, (case, image): (&Goldens, &ImageGoldens)) -> Result<String> {
    let size = case.size();
    let ours = engine.encode_frame(&image.frame(size), size)?;
    let latent_pixels = case.video.height * case.video.width;
    let reference = pixel_major(&image.moments, (MOMENT_CHANNELS, latent_pixels));
    Ok(compare("encode", &ours, &reference)?)
}

/// Conditions the engine on the case's text and first image; the case's conditioning latents.
fn condition(engine: &mut VideoEngine, case: &Goldens) -> Result<Vec<f32>> {
    engine.condition(Branch::Prompt, &case.text[0])?;
    engine.condition(Branch::Negative, &case.text[1])?;
    let states = case.image.as_ref().map(|image| image.states.as_slice());
    engine.condition_image(states)?;
    Ok(conditioning(case.first_latents().as_deref(), case.video))
}

/// The transformer's velocity for the reference's noise at its first timestep under each prompt.
fn dit(engine: &mut VideoEngine, case: &Goldens, conditioning: &[f32]) -> Result<String> {
    let input: Vec<f32> = transformer_input(&case.noise(), conditioning).collect();
    let timestep = case.timesteps[0];
    let at = (input.as_slice(), case.video);
    let prompt = engine.predict(at, timestep, Branch::Prompt)?;
    let negative = engine.predict(at, timestep, Branch::Negative)?;
    let reference = pixel_major(&case.velocity, (LATENT_CHANNELS, case.video.pixels()));
    let name = format!("dit {}", case.name);
    Ok(compare(&name, &[prompt, negative].concat(), &reference)?)
}

/// Adaptive projected guidance of the reference's two velocities at the first step.
fn guidance(case: &Goldens) -> Result<String> {
    let velocity = pixel_major(&case.velocity, (LATENT_CHANNELS, case.video.pixels()));
    let (conditional, unconditional) = velocity.split_at(velocity.len() / 2);
    let settings = Settings {
        scale: case.guidance,
        ..GUIDANCE
    };
    let group = case.video.height * case.video.width * LATENT_CHANNELS;
    let mut ours = vec![0.0; conditional.len()];
    Guidance::new(settings).guide((conditional, unconditional), group, &mut ours);
    let reference = pixel_major(&case.guided, (LATENT_CHANNELS, case.video.pixels()));
    let name = format!("guidance {}", case.name);
    Ok(compare(&name, &ours, &reference)?)
}

/// The schedule, then every step from the reference's noise to the last latents.
fn sampler(engine: &mut VideoEngine, case: &Goldens, conditioning: &[f32]) -> Result<String> {
    let steps = case.timesteps.len();
    let schedule = engine.schedule(steps);
    let timesteps: Vec<f32> = (0..steps).map(|step| schedule.timestep(step)).collect();
    ensure!(
        timesteps == case.timesteps,
        "timesteps {timesteps:?}, reference {:?}",
        case.timesteps
    );
    let levels = compare("levels", schedule.levels(), &case.levels)?;
    let ours = engine.sample(
        (&case.noise(), conditioning),
        case.video,
        (steps, case.guidance),
    )?;
    Ok(format!(
        "sampler {}: timesteps identical; {levels}; {}",
        case.name,
        compare("latents", &ours, &case.latents())?
    ))
}

/// The video of the reference's last latents.
fn decode(engine: &mut VideoEngine, case: &Goldens) -> Result<String> {
    let ours = engine.decode((&case.latents(), case.video))?;
    let reference = pixel_major(&case.frames, (IMAGE_CHANNELS, ours.len() / IMAGE_CHANNELS));
    let name = format!("decode {}", case.name);
    Ok(compare(&name, &ours, &reference)?)
}
