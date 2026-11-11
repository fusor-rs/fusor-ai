//! `motif-video bench`: milliseconds of each stage, and what a generation of `STEPS` steps adds
//! up to, from text and from a first image.
//!
//! The prompts' conditioning of the transformer and of the image's rows (about 0.3 TFLOP and
//! 0.4 TFLOP) runs before the timings and is not part of them.

use std::path::Path;

use anyhow::Result;
use bundle::Engine as _;
use harness::bench::{row, time};
use motif_video::image::first_frame;
use motif_video::model::{
    GUIDANCE, HEIGHT, LATENT_CHANNELS, NEGATIVE, TEXT_BUNDLE, VIDEO_BUNDLE, WIDTH,
};
use motif_video::pipeline::Stage;
use motif_video::session::{
    Branch, Request, TextEngine, VideoEngine, conditioning, transformer_input,
};
use motif_video::{Pixels, Size, Video};

use crate::RGBA_CHANNELS;

const CALLS: usize = 2;
const STEPS: usize = 20;
const PROMPT: &str =
    "A red fox trots through fresh snow in a birch forest at golden hour, cinematic";
/// The size of the synthetic first image: a photograph a page would hand over.
const IMAGE_SIZE: Size = Size {
    width: 1280,
    height: 736,
};
const BYTE_VALUES: usize = 256;

/// Milliseconds per call of each stage.
#[derive(Default)]
struct Timings {
    prompt: f64,
    text: f64,
    frame: f64,
    vision: f64,
    encode: f64,
    dit: f64,
    sampler: f64,
    image_sampler: f64,
    decode: f64,
}

/// What the text engine gives the video engine.
struct Inputs {
    states: [Vec<f32>; 2],
    frame: Vec<f32>,
    image_states: Vec<f32>,
}

pub fn run(bundle: &Path, frames: usize) -> Result<()> {
    let video = Request {
        width: WIDTH,
        height: HEIGHT,
        frames,
        steps: STEPS,
        guidance: GUIDANCE.scale,
        seed: 0,
    }
    .video()?;
    let size = Size {
        width: WIDTH,
        height: HEIGHT,
    };
    let mut timings = Timings::default();
    let inputs = time_text(bundle, size, &mut timings)?;
    time_video(bundle, (video, size), &inputs, &mut timings)?;
    let rows = [
        (Stage::Prompt, timings.prompt, ""),
        (Stage::Text, timings.text, ""),
        (Stage::Frame, timings.frame, ""),
        (Stage::Vision, timings.vision, ""),
        (Stage::Encode, timings.encode, ""),
        (Stage::Dit, timings.dit, ""),
        (Stage::Sampler, timings.sampler, "/step"),
        (
            Stage::Sampler,
            timings.image_sampler,
            "/step with a first image",
        ),
        (Stage::Decode, timings.decode, ""),
    ];
    for (stage, milliseconds, unit) in rows {
        row(stage, milliseconds, unit);
    }
    let total = timings.text + STEPS as f64 * timings.sampler + timings.decode;
    println!(
        "  a generation of {STEPS} steps at guidance {}: {:.1} s",
        GUIDANCE.scale,
        total / 1000.0
    );
    let image_steps = STEPS as f64 * timings.image_sampler;
    let total = timings.text + timings.vision + timings.encode + image_steps + timings.decode;
    println!("  with a first image: {:.1} s", total / 1000.0);
    Ok(())
}

/// The prompts, the text engine and the first image's resize and SigLIP.
fn time_text(bundle: &Path, size: Size, timings: &mut Timings) -> Result<Inputs> {
    let image = Pixels {
        size: IMAGE_SIZE,
        channels: RGBA_CHANNELS,
        values: (0..IMAGE_SIZE.width * IMAGE_SIZE.height * RGBA_CHANNELS)
            .map(|at| (at % BYTE_VALUES) as u8)
            .collect(),
    };
    let mut text: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    println!("threads {}, {}×{}", text.threads(), size.width, size.height);
    timings.prompt = time(CALLS, || {
        Ok::<_, motif_video::Error>((text.tokens(PROMPT)?, text.tokens(NEGATIVE)?))
    })?;
    let mut states = [Vec::new(), Vec::new()];
    timings.text = time(CALLS, || {
        states = [text.encode(PROMPT)?, text.encode(NEGATIVE)?];
        Ok::<_, motif_video::Error>(())
    })?;
    timings.frame = time(CALLS, || first_frame(&image, size))?;
    let frame = first_frame(&image, size)?;
    let mut image_states = Vec::new();
    timings.vision = time(CALLS, || {
        image_states = text.embed_frame(&frame, size)?;
        Ok::<_, motif_video::Error>(())
    })?;
    Ok(Inputs {
        states,
        frame,
        image_states,
    })
}

/// The first frame's encoder, the transformer, the sampler without and with the first image,
/// and the decoder.
fn time_video(
    bundle: &Path,
    (video, size): (Video, Size),
    inputs: &Inputs,
    timings: &mut Timings,
) -> Result<()> {
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    let [prompt, negative] = &inputs.states;
    engine.condition(Branch::Prompt, prompt)?;
    engine.condition(Branch::Negative, negative)?;
    timings.encode = time(CALLS, || engine.encode_frame(&inputs.frame, size))?;
    let latents: Vec<f32> = (0..video.pixels() * LATENT_CHANNELS)
        .map(|at| (at as f32 * 0.37).sin())
        .collect();
    let without = conditioning(None, video);
    let middle = engine.schedule(STEPS).timestep(STEPS / 2);
    let input: Vec<f32> = transformer_input(&latents, &without).collect();
    timings.dit = time(CALLS, || {
        engine.predict((&input, video), middle, Branch::Prompt)
    })?;
    timings.sampler = time(CALLS, || {
        engine.sample((&latents, &without), video, (1, GUIDANCE.scale))
    })?;
    let first_latents = vec![0.0; video.height * video.width * LATENT_CHANNELS];
    let with = conditioning(Some(&first_latents), video);
    engine.condition_image(Some(&inputs.image_states))?;
    timings.image_sampler = time(CALLS, || {
        engine.sample((&latents, &with), video, (1, GUIDANCE.scale))
    })?;
    timings.decode = time(CALLS, || engine.decode((&latents, video)))?;
    Ok(())
}
