//! `pjit-21k`: the native host of the runtimes.
//!
//! - `compile`: checkpoints → bundles (the `pjit-21k-compile` crate), `text/` and `image/`.
//! - `generate`: a caption → a PNG, printing the time of each step.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use pjit_21k::Size;
use pjit_21k::model::{DEFAULT_GUIDANCE, DEFAULT_STEPS, IMAGE_BUNDLE, NEGATIVE, TEXT_BUNDLE};
use pjit_21k::session::{ImageEngine, Plan, Request, TextEngine};

#[derive(Parser)]
#[command(name = "pjit-21k", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

/// One of the two bundles of the compiled checkpoints.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BundleKind {
    Text,
    Image,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (`qwen3.5-4b/` and `pjit-21k/`, as the product's
    /// scripts/fetch-checkpoint.sh lays them out) into the bundles `text/` and `image/` of `out`.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<pjit_21k_compile::Profile>)]
        profile: pjit_21k_compile::Profile,
        /// Only this bundle, so that the checkpoint file the other one reads can be deleted
        /// once this one is written.
        #[arg(long)]
        bundle: Option<BundleKind>,
    },
    /// Generate an image into a folder holding a PNG and its raw RGBA.
    Generate {
        /// The folder holding the bundles `text/` and `image/`.
        bundle: PathBuf,
        #[command(flatten)]
        settings: Settings,
        #[arg(long, default_value = "image")]
        out: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity {
        /// The folder holding the bundles `text/` and `image/`.
        bundle: PathBuf,
        goldens: PathBuf,
    },
    /// Time each stage at a size.
    Bench {
        bundle: PathBuf,
        #[arg(long, default_value_t = 1024)]
        width: usize,
        #[arg(long, default_value_t = 1024)]
        height: usize,
    },
}

/// What to generate (`session::Request`).
#[derive(Args)]
struct Settings {
    #[arg(long)]
    prompt: String,
    /// What the image should not show.
    #[arg(long, default_value = NEGATIVE)]
    negative: String,
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: usize,
    /// Adaptive projected guidance's scale; at 0 the caption's prediction goes unguided.
    #[arg(long, default_value_t = DEFAULT_GUIDANCE)]
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
        Command::Bench {
            bundle,
            width,
            height,
        } => bench::run(&bundle, Size { width, height })?,
    }
    Ok(())
}

/// The bundle `only` of the checkpoints, or both, into `out`.
fn compile_bundles(
    checkpoint: &Path,
    out: &Path,
    profile: pjit_21k_compile::Profile,
    only: Option<BundleKind>,
) -> Result<()> {
    if only != Some(BundleKind::Image) {
        let text = out.join(TEXT_BUNDLE);
        harness::compile(|| pjit_21k_compile::text(checkpoint, &text, profile))?;
    }
    if only != Some(BundleKind::Text) {
        let image = out.join(IMAGE_BUNDLE);
        harness::compile(|| pjit_21k_compile::image(checkpoint, &image, profile))?;
    }
    Ok(())
}

fn generate(bundle: &Path, settings: &Settings, out: &Path) -> Result<()> {
    let start = Instant::now();
    let (prompt, negative) = encode(bundle, settings)?;
    eprintln!("prompts encoded in {:.2}s", start.elapsed().as_secs_f64());
    let mut engine: ImageEngine = harness::load(&bundle.join(IMAGE_BUNDLE))?;
    let request = Request {
        width: settings.width,
        height: settings.height,
        steps: settings.steps,
        guidance: settings.guidance,
        seed: settings.seed,
    };
    let plan = engine.start(&request, &prompt, &negative)?;
    loop {
        let step_start = Instant::now();
        let progress = engine.step()?;
        eprintln!(
            "step {}/{} in {:.2}s",
            progress.step,
            progress.steps,
            step_start.elapsed().as_secs_f64()
        );
        if progress.done {
            break;
        }
    }
    write_image(out, &plan, &engine.take_image()?)?;
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

/// The states of the caption and of the negative prompt, from the text engine, which is dropped
/// before the image engine loads.
fn encode(bundle: &Path, settings: &Settings) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut engine: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    let tokens = engine.tokens(&settings.prompt)?;
    if tokens.dropped > 0 {
        eprintln!(
            "the caption's last {} tokens are past what the encoder reads",
            tokens.dropped
        );
    }
    let prompt = engine.encode_tokens(&tokens.ids)?;
    let negative = engine.encode(&settings.negative)?;
    Ok((prompt, negative))
}

/// The image's RGBA rows top to bottom as `image.rgba`, and as `image.png`.
fn write_image(out: &Path, plan: &Plan, rgba: &[u8]) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| out.display().to_string())?;
    let raw = out.join("image.rgba");
    std::fs::write(&raw, rgba).with_context(|| raw.display().to_string())?;
    let size = (plan.width, plan.height);
    harness::write_png(&out.join("image.png"), size, rgba)?;
    Ok(())
}
