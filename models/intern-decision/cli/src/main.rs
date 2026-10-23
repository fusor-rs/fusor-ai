//! `intern-decision`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `intern-decision-compile` crate).
//! - `decide`: the requests of a JSON file → their responses, with the time each took.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one request.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use intern_decision::Pixels;
use intern_decision::request::Request;
use intern_decision::session::Engine;
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "intern-decision", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (its safetensors and tokenizer.json, as the product's
    /// scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<intern_decision_compile::Profile>)]
        profile: intern_decision_compile::Profile,
    },
    /// Decide every request of a JSON file (`{"<case>": request}`), each under a `# <case>` line.
    Decide {
        bundle: PathBuf,
        requests: PathBuf,
        /// The directory the requests' image paths are relative to.
        #[arg(long, default_value = ".")]
        images: PathBuf,
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
        images: PathBuf,
    },
}

/// A request as a JSON file holds it: the request, and its images' paths.
#[derive(Deserialize)]
struct Case {
    #[serde(flatten)]
    request: Request,
    #[serde(default)]
    images: Vec<PathBuf>,
}

impl Case {
    fn pixels(&self, dir: &Path) -> Result<Vec<Pixels>> {
        let pixels = self
            .images
            .iter()
            .map(|path| harness::image(&dir.join(path)));
        Ok(pixels.collect::<Result<_, _>>()?)
    }
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
        } => harness::compile(|| intern_decision_compile::run(&checkpoint, &out, profile))?,
        Command::Decide {
            bundle,
            requests,
            images,
        } => decide(&mut harness::load(&bundle)?, &cases(&requests)?, &images)?,
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
            images,
        } => {
            let cases = cases(&requests)?;
            let (_, case) = cases
                .iter()
                .find(|(name, _)| *name == case)
                .with_context(|| format!("{}: no case {case}", requests.display()))?;
            bench::run(
                &mut harness::load(&bundle)?,
                &case.request,
                &case.pixels(&images)?,
            )?;
        }
    }
    Ok(())
}

fn decide(engine: &mut Engine, cases: &[(String, Case)], images: &Path) -> Result<()> {
    for (name, case) in cases {
        println!("# {name}");
        let start = Instant::now();
        let plan = engine.start(&case.request, &case.pixels(images)?)?;
        let mut progress = engine.step()?;
        while progress.response.is_none() {
            progress = engine.step()?;
        }
        println!("{}", serde_json::to_string_pretty(&progress.response)?);
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{} prompt tokens ({} of images), {} questions: {seconds:.2}s ({:.0} tokens/s)",
            plan.prompt_tokens,
            plan.image_tokens,
            plan.questions,
            plan.prompt_tokens as f64 / seconds
        );
    }
    Ok(())
}
