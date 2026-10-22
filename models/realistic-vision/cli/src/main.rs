//! `realistic-vision`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `realistic-vision-compile` crate).
//! - `generate`: a prompt → a PNG, printing the time of each step.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use realistic_vision::session::{Engine, Request};

#[derive(Parser)]
#[command(name = "realistic-vision", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (the Realistic Vision and SD VAE ft-MSE safetensors,
    /// vocab.json and merges.txt, as the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7")]
        profile: realistic_vision_compile::Profile,
    },
    /// Generate an image into a PNG file.
    Generate {
        bundle: PathBuf,
        #[command(flatten)]
        settings: Settings,
        #[arg(long, default_value = "image.png")]
        out: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each stage at a size.
    Bench {
        bundle: PathBuf,
        #[arg(long, default_value_t = 512)]
        width: usize,
        #[arg(long, default_value_t = 512)]
        height: usize,
    },
}

/// What to generate (`session::Request`).
#[derive(Args)]
struct Settings {
    #[arg(long)]
    prompt: String,
    /// What the image should not show.
    #[arg(long, default_value = "")]
    negative: String,
    #[arg(long, default_value_t = 512)]
    width: usize,
    #[arg(long, default_value_t = 512)]
    height: usize,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Classifier-free guidance; 1 or below runs the prompt alone.
    #[arg(long, default_value_t = 5.0)]
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
        } => harness::compile(|| realistic_vision_compile::run(&checkpoint, &out, profile))?,
        Command::Generate {
            bundle,
            settings,
            out,
        } => generate(&mut harness::load(&bundle)?, settings, &out)?,
        Command::Parity { bundle, goldens } => parity::run(&mut harness::load(&bundle)?, &goldens)?,
        Command::Bench {
            bundle,
            width,
            height,
        } => bench::run(&mut harness::load(&bundle)?, (width, height))?,
    }
    Ok(())
}

fn generate(engine: &mut Engine, settings: Settings, out: &Path) -> Result<()> {
    let request = Request {
        prompt: settings.prompt,
        negative: settings.negative,
        width: settings.width,
        height: settings.height,
        steps: settings.steps,
        guidance: settings.guidance,
        seed: settings.seed,
    };
    let start = Instant::now();
    let plan = engine.start(&request)?;
    for (prompt, dropped) in ["prompt", "negative prompt"].iter().zip(plan.dropped) {
        if dropped > 0 {
            eprintln!("the {prompt}'s last {dropped} tokens are past what CLIP reads");
        }
    }
    eprintln!("prompts encoded in {:.2}s", start.elapsed().as_secs_f64());
    loop {
        let step_start = Instant::now();
        let progress = engine.step()?;
        let what = if progress.done {
            "decoded".to_string()
        } else {
            format!("step {}/{}", progress.step, progress.steps)
        };
        eprintln!("{what} in {:.2}s", step_start.elapsed().as_secs_f64());
        if progress.done {
            break;
        }
    }
    let image = engine.take_image()?;
    write_png(out, (plan.width, plan.height), &image)?;
    println!(
        "{} ({}×{}, {} steps) in {:.1}s",
        out.display(),
        plan.width,
        plan.height,
        plan.steps,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// RGBA rows top to bottom as a PNG file.
fn write_png(path: &Path, (width, height): (usize, usize), rgba: &[u8]) -> Result<()> {
    let file = std::fs::File::create(path).with_context(|| path.display().to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    writer.finish()?;
    Ok(())
}
