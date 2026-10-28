//! `d1`: the native host of the runtime, for either variant.
//!
//! - `compile`: checkpoint → bundle (the `d1-compile` crate).
//! - `decide`: the requests of a JSON file, with their images and clips → their responses,
//!   with the time each took.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one request.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use d1::Pixels;
use d1::media::SAMPLE_RATE;
use d1::request::Request;
use d1::session::Engine;
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "d1", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a checkpoint directory (its safetensors, config.json, calibration.json and
    /// tokenizer.json) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<d1_compile::Profile>)]
        profile: d1_compile::Profile,
    },
    /// Decide every request of a JSON file (`{"<case>": request}`), each under a `# <case>`
    /// line.
    Decide {
        bundle: PathBuf,
        requests: PathBuf,
        /// The directory the requests' media paths are relative to.
        #[arg(long, default_value = ".")]
        media: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference, one file per case.
    Parity {
        bundle: PathBuf,
        #[arg(required = true)]
        goldens: Vec<PathBuf>,
    },
    /// Time each stage on one request of a JSON file.
    Bench {
        bundle: PathBuf,
        requests: PathBuf,
        case: String,
        #[arg(long, default_value = ".")]
        media: PathBuf,
    },
}

/// A request as a JSON file holds it: the request, its images' paths and its clips' (16-bit
/// mono WAV files at 16 kHz, d1-Omni-2B only).
#[derive(Deserialize)]
struct Case {
    #[serde(flatten)]
    request: Request,
    #[serde(default)]
    images: Vec<PathBuf>,
    #[serde(default)]
    clips: Vec<PathBuf>,
}

impl Case {
    fn images(&self, dir: &Path) -> Result<Vec<Pixels>> {
        let images = self
            .images
            .iter()
            .map(|path| harness::image(&dir.join(path)));
        Ok(images.collect::<Result<_, _>>()?)
    }

    fn clips(&self, dir: &Path) -> Result<Vec<Vec<f32>>> {
        self.clips
            .iter()
            .map(|path| clip(&dir.join(path)))
            .collect()
    }
}

/// A WAV file's samples, which must be at `SAMPLE_RATE`.
fn clip(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let (samples, rate) = media::wav::decode(&bytes).with_context(|| path.display().to_string())?;
    ensure!(
        rate == SAMPLE_RATE,
        "{}: {rate} Hz, resample it to {SAMPLE_RATE} Hz",
        path.display()
    );
    Ok(samples)
}

/// The cases of a requests file, in order.
fn cases(path: &Path) -> Result<Vec<(String, Case)>> {
    let json = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let cases: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&json).with_context(|| path.display().to_string())?;
    cases
        .into_iter()
        .map(|(name, case)| {
            let case = Case::deserialize(case).with_context(|| name.clone())?;
            Ok((name, case))
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| d1_compile::run(&checkpoint, &out, profile))?,
        Command::Decide {
            bundle,
            requests,
            media,
        } => decide(&mut harness::load(&bundle)?, &cases(&requests)?, &media)?,
        Command::Parity { bundle, goldens } => {
            let mut engine = harness::load(&bundle)?;
            for goldens in &goldens {
                parity::run(&mut engine, goldens)?;
            }
        }
        Command::Bench {
            bundle,
            requests,
            case,
            media,
        } => {
            let cases = cases(&requests)?;
            let (_, case) = cases
                .iter()
                .find(|(name, _)| *name == case)
                .with_context(|| format!("{}: no case {case}", requests.display()))?;
            let (images, clips) = (case.images(&media)?, case.clips(&media)?);
            bench::run(&mut harness::load(&bundle)?, &case.request, &images, &clips)?;
        }
    }
    Ok(())
}

fn decide(engine: &mut Engine, cases: &[(String, Case)], media: &Path) -> Result<()> {
    for (name, case) in cases {
        println!("# {name}");
        let start = Instant::now();
        engine.start(&case.request, &case.images(media)?, &case.clips(media)?)?;
        let mut progress = engine.step()?;
        while progress.response.is_none() {
            progress = engine.step()?;
        }
        let response = progress.response.context("the last step answers")?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{} prompt tokens over {} questions: {seconds:.2}s",
            response.usage.input_tokens,
            response.answers.len(),
        );
    }
    Ok(())
}
