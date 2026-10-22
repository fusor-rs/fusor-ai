//! `paddle-ocr`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `paddle-ocr-compile` crate).
//! - `read`: PNG or JPEG images → their text (or table, formula, chart, seal), with the time
//!   each took.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one image.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use paddle_ocr::prompt::Task;
use paddle_ocr::session::{Engine, Request};
use paddle_ocr::{Pixels, Size};
use paddle_ocr_compile::Profile;

#[derive(Parser)]
#[command(name = "paddle-ocr", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (model.safetensors and tokenizer.json, as
    /// the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<Profile>)]
        profile: Profile,
    },
    /// Read PNG or JPEG images, each under a `# <file>` line.
    Read {
        bundle: PathBuf,
        #[arg(required = true)]
        images: Vec<PathBuf>,
        /// What to read: ocr, table, formula, chart or seal.
        #[arg(long, default_value = "ocr", value_parser = task)]
        task: Task,
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each stage on an image.
    Bench {
        bundle: PathBuf,
        image: PathBuf,
        #[arg(long, default_value = "ocr", value_parser = task)]
        task: Task,
    },
}

fn task(name: &str) -> Result<Task, String> {
    let names = Task::ALL.map(Task::name);
    Task::ALL
        .into_iter()
        .find(|task| task.name() == name)
        .ok_or_else(|| format!("{name}: one of {}", names.join(", ")))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| paddle_ocr_compile::run(&checkpoint, &out, profile))?,
        Command::Read {
            bundle,
            images,
            task,
            max_tokens,
        } => {
            let request = Request { task, max_tokens };
            read(&mut harness::load(&bundle)?, &images, &request)?;
        }
        Command::Parity { bundle, goldens } => parity::run(&mut harness::load(&bundle)?, &goldens)?,
        Command::Bench {
            bundle,
            image,
            task,
        } => bench::run(&mut harness::load(&bundle)?, &decode(&image)?, task)?,
    }
    Ok(())
}

/// The pixels of a PNG or JPEG file, as RGBA.
fn decode(path: &Path) -> Result<Pixels> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let image = image::load_from_memory(&bytes)
        .with_context(|| path.display().to_string())?
        .into_rgba8();
    Ok(Pixels {
        size: Size {
            width: image.width() as usize,
            height: image.height() as usize,
        },
        channels: 4,
        values: image.into_raw(),
    })
}

fn read(engine: &mut Engine, images: &[PathBuf], request: &Request) -> Result<()> {
    for path in images {
        let image = decode(path)?;
        println!("# {}", path.display());
        let start = Instant::now();
        let plan = engine.start(&image, request)?;
        let (mut first, mut progress) = (None, engine.step()?);
        while !progress.done {
            progress = engine.step()?;
            first.get_or_insert_with(|| start.elapsed().as_secs_f64());
        }
        println!("{}", progress.text);
        let seconds = start.elapsed().as_secs_f64();
        let first = first.unwrap_or(seconds);
        eprintln!(
            "{}×{} → {} image tokens: first token after {first:.2}s, {} tokens in {seconds:.2}s \
             ({:.1} tokens/s after the first)",
            plan.layout.width,
            plan.layout.height,
            plan.image_tokens,
            progress.tokens,
            progress.tokens.saturating_sub(1) as f64 / (seconds - first).max(f64::EPSILON)
        );
    }
    Ok(())
}
