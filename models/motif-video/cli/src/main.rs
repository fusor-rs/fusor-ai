//! `motif-video`: the native host of the runtimes.
//!
//! - `compile`: checkpoint → bundles (the `motif-video-compile` crate), `text/` and `video/`.
//! - `generate`: a prompt, and optionally a first image → PNG frames, printing the time of each
//!   step.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use motif_video::Size;
use motif_video::image::first_frame;
use motif_video::model::{GUIDANCE, HEIGHT, NEGATIVE, TEXT_BUNDLE, VIDEO_BUNDLE, WIDTH};
use motif_video::session::{
    FirstImage, MAX_FRAMES, Plan, Progress, Request, TextEngine, VideoEngine,
};

/// Bytes of a frame's pixel in `frames.rgba`.
pub(crate) const RGBA_CHANNELS: usize = 4;

#[derive(Parser)]
#[command(name = "motif-video", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

/// One of the two bundles of a compiled checkpoint.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BundleKind {
    Text,
    Video,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (Motif-Video-2B's files, as the product's
    /// scripts/fetch-checkpoint.sh downloads them) into the bundles `text/` and `video/` of
    /// `out`.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<motif_video_compile::Profile>)]
        profile: motif_video_compile::Profile,
        /// Only this bundle, so that the checkpoint file the other one reads can be deleted
        /// once this one is written.
        #[arg(long)]
        bundle: Option<BundleKind>,
    },
    /// Generate a video into a folder of PNG frames and their raw RGBA.
    Generate {
        /// The folder holding the bundles `text/` and `video/`.
        bundle: PathBuf,
        #[command(flatten)]
        settings: Settings,
        #[arg(long, default_value = "clip")]
        out: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity {
        /// The folder holding the bundles `text/` and `video/`.
        bundle: PathBuf,
        /// The folder holding `golden_f32_t2v.safetensors` and `golden_f32_i2v.safetensors`.
        goldens: PathBuf,
    },
    /// Time each stage.
    Bench {
        bundle: PathBuf,
        #[arg(long, default_value_t = MAX_FRAMES)]
        frames: usize,
    },
}

/// What to generate (`session::Request`).
#[derive(Args)]
struct Settings {
    #[arg(long)]
    prompt: String,
    /// What the video should not show.
    #[arg(long, default_value = NEGATIVE)]
    negative: String,
    /// A PNG or JPEG file for the video's first frame.
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long, default_value_t = MAX_FRAMES)]
    frames: usize,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Adaptive projected guidance's scale; 0 runs the prompt alone.
    #[arg(long, default_value_t = GUIDANCE.scale)]
    guidance: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
            bundle,
        } => compile_bundles(&checkpoint, &out, profile, bundle)?,
        Command::Generate {
            bundle,
            settings,
            out,
        } => generate(&bundle, &settings, &out)?,
        Command::Parity { bundle, goldens } => parity::run(&bundle, &goldens)?,
        Command::Bench { bundle, frames } => bench::run(&bundle, frames)?,
    }
    Ok(())
}

/// The bundle `only` of the checkpoint, or both, into `out`.
fn compile_bundles(
    checkpoint: &Path,
    out: &Path,
    profile: motif_video_compile::Profile,
    only: Option<BundleKind>,
) -> Result<()> {
    if only != Some(BundleKind::Video) {
        let text = out.join(TEXT_BUNDLE);
        harness::compile(|| motif_video_compile::text(checkpoint, &text, profile))?;
    }
    if only != Some(BundleKind::Text) {
        let video = out.join(VIDEO_BUNDLE);
        harness::compile(|| motif_video_compile::video(checkpoint, &video, profile))?;
    }
    Ok(())
}

/// What the text engine gives the video engine.
struct Encoded {
    prompt: Vec<f32>,
    negative: Vec<f32>,
    image: Option<FirstImage>,
}

fn generate(bundle: &Path, settings: &Settings, out: &Path) -> Result<()> {
    let start = Instant::now();
    let encoded = encode(bundle, settings)?;
    eprintln!("prompts encoded in {:.2}s", start.elapsed().as_secs_f64());
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    if let Some(image) = encoded.image {
        engine.first_image(image);
    }
    let request = Request {
        width: WIDTH,
        height: HEIGHT,
        frames: settings.frames,
        steps: settings.steps,
        guidance: settings.guidance,
        seed: settings.seed,
    };
    let plan = engine.start(&request, &encoded.prompt, &encoded.negative)?;
    let mut frames = Vec::new();
    loop {
        let step_start = Instant::now();
        let progress = engine.step()?;
        frames.extend(engine.take_frames()?);
        eprintln!(
            "{} in {:.2}s",
            describe(&progress),
            step_start.elapsed().as_secs_f64()
        );
        if progress.done {
            break;
        }
    }
    write_frames(out, &plan, &frames)?;
    let image = match &settings.image {
        Some(path) => format!(", from {}", path.display()),
        None => String::new(),
    };
    println!(
        "{} ({}×{}, {} frames, {} steps{image}) in {:.1}s",
        out.display(),
        plan.width,
        plan.height,
        plan.frames,
        plan.steps,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The states of the prompt and of the negative prompt (empty at guidance 0, which does not
/// read it) and of the first image, from the text engine, which is dropped before the video
/// engine loads.
fn encode(bundle: &Path, settings: &Settings) -> Result<Encoded> {
    let mut engine: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    let prompt = engine.encode(&settings.prompt)?;
    let negative = if settings.guidance > 0.0 {
        engine.encode(&settings.negative)?
    } else {
        Vec::new()
    };
    let size = Size {
        width: WIDTH,
        height: HEIGHT,
    };
    let image = settings
        .image
        .as_deref()
        .map(|path| embed_image(&mut engine, path, size))
        .transpose()?;
    Ok(Encoded {
        prompt,
        negative,
        image,
    })
}

/// The image file resized to the video's size, and SigLIP's states of it.
fn embed_image(engine: &mut TextEngine, path: &Path, size: Size) -> Result<FirstImage> {
    let frame = first_frame(&harness::image(path)?, size)?;
    let states = engine.embed_frame(&frame, size)?;
    Ok(FirstImage { states, frame })
}

/// What the page shows of a step: the denoising step, or the strips decoded.
fn describe(progress: &Progress) -> String {
    if progress.step < progress.steps {
        format!("step {}/{}", progress.step, progress.steps)
    } else {
        format!("strips {}/{}", progress.decoded, progress.strips)
    }
}

/// Every frame's RGBA rows top to bottom as `frames.rgba`, and each frame as `NNN.png`.
fn write_frames(out: &Path, plan: &Plan, rgba: &[u8]) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| out.display().to_string())?;
    let raw = out.join("frames.rgba");
    std::fs::write(&raw, rgba).with_context(|| raw.display().to_string())?;
    let frame_bytes = plan.width * plan.height * RGBA_CHANNELS;
    for (index, frame) in rgba.chunks_exact(frame_bytes).enumerate() {
        harness::write_png(
            &out.join(format!("{index:03}.png")),
            (plan.width, plan.height),
            frame,
        )?;
    }
    Ok(())
}
