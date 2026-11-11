//! `wan2-1`: the native host of the runtimes.
//!
//! - `compile`: checkpoint → bundles (the `wan2-1-compile` crate), `text/` and `video/`.
//! - `generate`: a prompt → PNG frames, printing the time of each step.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use wan2_1::model::{DEFAULT_GUIDANCE, DEFAULT_SHIFT, NEGATIVE, RGBA, TEXT_BUNDLE, VIDEO_BUNDLE};
use wan2_1::session::{Plan, Progress, Request, TextEngine, VideoEngine};

#[derive(Parser)]
#[command(name = "wan2-1", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (Wan2.1-T2V-1.3B's files, as the product's
    /// scripts/fetch-checkpoint.sh downloads them) into the bundles `text/` and `video/` of `out`.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<wan2_1_compile::Profile>)]
        profile: wan2_1_compile::Profile,
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
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each stage at a size.
    Bench {
        bundle: PathBuf,
        #[arg(long, default_value_t = 320)]
        width: usize,
        #[arg(long, default_value_t = 192)]
        height: usize,
        #[arg(long, default_value_t = 17)]
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
    #[arg(long, default_value_t = 320)]
    width: usize,
    #[arg(long, default_value_t = 192)]
    height: usize,
    #[arg(long, default_value_t = 17)]
    frames: usize,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Classifier-free guidance, at least 1; 1 runs the prompt alone.
    #[arg(long, default_value_t = DEFAULT_GUIDANCE)]
    guidance: f32,
    #[arg(long, default_value_t = DEFAULT_SHIFT)]
    shift: f32,
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
        } => {
            let text = out.join(TEXT_BUNDLE);
            harness::compile(|| wan2_1_compile::text(&checkpoint, &text, profile))?;
            let video = out.join(VIDEO_BUNDLE);
            harness::compile(|| wan2_1_compile::video(&checkpoint, &video, profile))?;
        }
        Command::Generate {
            bundle,
            settings,
            out,
        } => generate(&bundle, &settings, &out)?,
        Command::Parity { bundle, goldens } => parity::run(&bundle, &goldens)?,
        Command::Bench {
            bundle,
            width,
            height,
            frames,
        } => bench::run(&bundle, (width, height, frames))?,
    }
    Ok(())
}

fn generate(bundle: &Path, settings: &Settings, out: &Path) -> Result<()> {
    let start = Instant::now();
    let (prompt, negative) = encode(bundle, settings)?;
    eprintln!("prompts encoded in {:.2}s", start.elapsed().as_secs_f64());
    let mut engine: VideoEngine = harness::load(&bundle.join(VIDEO_BUNDLE))?;
    let request = Request {
        width: settings.width,
        height: settings.height,
        frames: settings.frames,
        steps: settings.steps,
        guidance: settings.guidance,
        shift: settings.shift,
        seed: settings.seed,
    };
    let plan = engine.start(&request, &prompt, &negative)?;
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
    println!(
        "{} ({}×{}, {} frames, {} steps) in {:.1}s",
        out.display(),
        plan.width,
        plan.height,
        plan.frames,
        plan.steps,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The states of the prompt and of the negative prompt (empty at guidance 1, which does not
/// read it), from the text engine, which is dropped before the video engine loads.
fn encode(bundle: &Path, settings: &Settings) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut engine: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    let prompt = engine.encode(&settings.prompt)?;
    let negative = if settings.guidance > 1.0 {
        engine.encode(&settings.negative)?
    } else {
        Vec::new()
    };
    Ok((prompt, negative))
}

/// What the page shows of a step: the denoising step, or the frames decoded.
fn describe(progress: &Progress) -> String {
    if progress.step < progress.steps {
        format!("step {}/{}", progress.step, progress.steps)
    } else {
        format!("frames {}/{}", progress.decoded, progress.frames)
    }
}

/// Every frame's RGBA rows top to bottom as `frames.rgba`, and each frame as `NNN.png`.
fn write_frames(out: &Path, plan: &Plan, rgba: &[u8]) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| out.display().to_string())?;
    let raw = out.join("frames.rgba");
    std::fs::write(&raw, rgba).with_context(|| raw.display().to_string())?;
    let frame_bytes = plan.width * plan.height * RGBA;
    for (index, frame) in rgba.chunks_exact(frame_bytes).enumerate() {
        harness::write_png(
            &out.join(format!("{index:03}.png")),
            (plan.width, plan.height),
            frame,
        )?;
    }
    Ok(())
}
