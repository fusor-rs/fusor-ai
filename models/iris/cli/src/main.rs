//! `iris`: the native host of the runtimes.
//!
//! - `compile`: checkpoint → one bundle (the `iris-compile` crate): `text`, `base`, `turbo`,
//!   `depth` or `upscaler`.
//! - `generate`: a prompt → a PNG, printing the time of each step.
//! - `depth`: a photo → its depth map as a PNG.
//! - `upscale`: a photo → its four times enlarged restoration as a PNG.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use iris::model::{DEFAULT_GUIDANCE, DEFAULT_SHIFT, DEFAULT_STEPS, TEXT_BUNDLE, Variant};
use iris::session::{DepthRequest, Generation, ImageEngine, Plan, TextEngine, Upscale};

#[derive(Parser)]
#[command(name = "iris", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile one bundle of the checkpoint directory (`text/` holding Qwen3-VL-4B-Instruct's
    /// files, `iris/` Iris-3B's, as the product's scripts/fetch-checkpoint.sh downloads them)
    /// into `out`.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, value_parser = Bundle::parse)]
        bundle: Bundle,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<iris_compile::Profile>)]
        profile: iris_compile::Profile,
    },
    /// Generate an image into a PNG file and its raw RGBA.
    Generate {
        /// The folder holding the bundles `text/` and the model's.
        bundle: PathBuf,
        #[command(flatten)]
        settings: Settings,
        #[arg(long, default_value = "image.png")]
        out: PathBuf,
    },
    /// Estimate the depth of a photo into a PNG file and its raw RGBA.
    Depth {
        /// The folder holding the bundle `depth/`.
        bundle: PathBuf,
        photo: PathBuf,
        /// The longest side the model sees; 0 keeps the photo's.
        #[arg(long, default_value_t = 1024)]
        max_side: usize,
        #[arg(long, default_value = "depth.png")]
        out: PathBuf,
    },
    /// Enlarge a photo four times into a PNG file and its raw RGBA.
    Upscale {
        /// The folder holding the bundle `upscaler/`.
        bundle: PathBuf,
        photo: PathBuf,
        /// The photo is shrunk to these sides first: the short one and the long one.
        #[arg(long, default_value_t = 512)]
        short_side: usize,
        #[arg(long, default_value_t = 512)]
        long_side: usize,
        #[arg(long, default_value = "upscaled.png")]
        out: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity {
        /// The folder holding the bundles.
        bundle: PathBuf,
        goldens: PathBuf,
        /// Check this bundle's stages only.
        #[arg(long, value_parser = Bundle::parse)]
        only: Option<Bundle>,
    },
    /// Time each stage.
    Bench { bundle: PathBuf },
}

/// A bundle of a checkpoint: Qwen3-VL's or one of the transformers'.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bundle {
    Text,
    Transformer(Variant),
}

impl Bundle {
    const ALL: [Self; 5] = [
        Self::Text,
        Self::Transformer(Variant::Base),
        Self::Transformer(Variant::Turbo),
        Self::Transformer(Variant::Depth),
        Self::Transformer(Variant::Upscaler),
    ];

    /// Its directory under a bundle root.
    fn name(self) -> &'static str {
        match self {
            Self::Text => TEXT_BUNDLE,
            Self::Transformer(variant) => variant.name(),
        }
    }

    fn parse(name: &str) -> Result<Self, String> {
        let names: Vec<&str> = Self::ALL.iter().map(|bundle| bundle.name()).collect();
        Self::ALL
            .into_iter()
            .find(|bundle| bundle.name() == name)
            .ok_or_else(|| format!("no bundle {name:?} ({})", names.join(", ")))
    }
}

/// What to generate (`session::Generation`).
#[derive(Args)]
struct Settings {
    #[arg(long)]
    prompt: String,
    /// What the image should not show (base, guided).
    #[arg(long, default_value = "")]
    negative: String,
    /// The model: `turbo` (four steps) or `base`.
    #[arg(long, default_value = "turbo", value_parser = prompt_variant)]
    variant: Variant,
    /// One of the sizes the model was trained at (`model::SIZES`).
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    /// Denoising steps (base).
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: usize,
    /// Classifier-free guidance (base); 1 runs the prompt alone.
    #[arg(long, default_value_t = DEFAULT_GUIDANCE)]
    guidance: f32,
    #[arg(long, default_value_t = DEFAULT_SHIFT)]
    shift: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn prompt_variant(name: &str) -> Result<Variant, String> {
    match Bundle::parse(name)? {
        Bundle::Transformer(variant @ (Variant::Base | Variant::Turbo)) => Ok(variant),
        _ => Err(format!("{name:?} does not read a prompt: base or turbo")),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            bundle,
            profile,
        } => compile_bundle(&checkpoint, &out, (bundle, profile))?,
        Command::Generate {
            bundle,
            settings,
            out,
        } => generate(&bundle, &settings, &out)?,
        Command::Depth {
            bundle,
            photo,
            max_side,
            out,
        } => depth(&bundle, &photo, max_side, &out)?,
        Command::Upscale {
            bundle,
            photo,
            short_side,
            long_side,
            out,
        } => upscale(&bundle, &photo, (short_side, long_side), &out)?,
        Command::Parity {
            bundle,
            goldens,
            only,
        } => parity::run(&bundle, &goldens, only)?,
        Command::Bench { bundle } => bench::run(&bundle)?,
    }
    Ok(())
}

fn compile_bundle(
    checkpoint: &Path,
    out: &Path,
    (bundle, profile): (Bundle, iris_compile::Profile),
) -> Result<()> {
    let target = out.join(bundle.name());
    match bundle {
        Bundle::Text => {
            let source = checkpoint.join(TEXT_BUNDLE);
            harness::compile(|| iris_compile::text(&source, &target, profile))?;
        }
        Bundle::Transformer(variant) => {
            let source = checkpoint.join("iris").join(iris_compile::folder(variant));
            harness::compile(|| iris_compile::transformer(&source, &target, (variant, profile)))?;
        }
    }
    Ok(())
}

fn generate(bundle: &Path, settings: &Settings, out: &Path) -> Result<()> {
    let start = Instant::now();
    let request = Generation {
        width: settings.width,
        height: settings.height,
        steps: settings.steps,
        guidance: settings.guidance,
        shift: settings.shift,
        seed: settings.seed,
    };
    let (prompt, negative) = encode(bundle, settings, &request)?;
    eprintln!("prompts encoded in {:.2}s", start.elapsed().as_secs_f64());
    let mut engine: ImageEngine = harness::load(&bundle.join(settings.variant.name()))?;
    let plan = engine.start_generation(&request, &prompt, &negative)?;
    finish(&mut engine, &plan, out, start)
}

/// The states of the prompt and, for a guided base run, of the negative prompt (else empty),
/// from the text engine, which is dropped before the image engine loads.
fn encode(
    bundle: &Path,
    settings: &Settings,
    request: &Generation,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut engine: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    let prompt = engine.encode(&settings.prompt)?;
    let negative = if request.guided(settings.variant) {
        engine.encode(&settings.negative)?
    } else {
        Vec::new()
    };
    Ok((prompt, negative))
}

fn depth(bundle: &Path, photo: &Path, max_side: usize, out: &Path) -> Result<()> {
    let start = Instant::now();
    let photo = harness::image(photo)?.rgb();
    let mut engine: ImageEngine = harness::load(&bundle.join(Variant::Depth.name()))?;
    let plan = engine.start_depth(&DepthRequest { max_side }, &photo)?;
    finish(&mut engine, &plan, out, start)
}

fn upscale(
    bundle: &Path,
    photo: &Path,
    (short_side, long_side): (usize, usize),
    out: &Path,
) -> Result<()> {
    let start = Instant::now();
    let photo = harness::image(photo)?.rgb();
    let mut engine: ImageEngine = harness::load(&bundle.join(Variant::Upscaler.name()))?;
    let request = Upscale {
        short_side,
        long_side,
    };
    let plan = engine.start_upscale(&request, &photo)?;
    finish(&mut engine, &plan, out, start)
}

/// Runs the started job to its end, printing each step, and writes its image to `out` (PNG)
/// and beside it as raw RGBA.
fn finish(engine: &mut ImageEngine, plan: &Plan, out: &Path, start: Instant) -> Result<()> {
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
    let rgba = engine.take_image()?;
    harness::write_png(out, (plan.width, plan.height), &rgba)?;
    let raw = out.with_extension("rgba");
    std::fs::write(&raw, &rgba).with_context(|| raw.display().to_string())?;
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
