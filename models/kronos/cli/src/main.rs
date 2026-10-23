//! `kronos`: the native host of the runtime.
//!
//! - `compile`: checkpoints → bundle (the `kronos-compile` crate).
//! - `forecast`: the cases of a JSON file → their forecasts, with the time each took.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one case.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kronos::sampling::Sampling;
use kronos::session::{Engine, Progress, Request, Rows};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "kronos", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (Kronos-base in model/, Kronos-Tokenizer-base in
    /// tokenizer/, as the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<kronos_compile::Profile>)]
        profile: kronos_compile::Profile,
    },
    /// Forecast every case of a JSON file (`{"<case>": case}`), each under a `# <case>` line.
    Forecast {
        bundle: PathBuf,
        cases: PathBuf,
        /// The directory the cases' series are relative to.
        #[arg(long, default_value = ".")]
        series: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference, one file per case
    /// (`golden_f32_<case>.safetensors`).
    Parity {
        bundle: PathBuf,
        cases: PathBuf,
        series: PathBuf,
        #[arg(required = true)]
        goldens: Vec<PathBuf>,
    },
    /// Time each stage on one case of a JSON file.
    Bench {
        bundle: PathBuf,
        cases: PathBuf,
        case: String,
        #[arg(long, default_value = ".")]
        series: PathBuf,
    },
}

/// A case as a JSON file holds it: a request with its series' path in place of its CSV.
#[derive(Deserialize)]
struct Case {
    series: PathBuf,
    lookback: usize,
    horizon: usize,
    end: Option<usize>,
    #[serde(default)]
    sampling: Sampling,
    #[serde(default)]
    seed: u64,
}

impl Case {
    fn request(&self, dir: &Path) -> Result<Request> {
        let path = dir.join(&self.series);
        Ok(Request {
            csv: std::fs::read_to_string(&path).with_context(|| path.display().to_string())?,
            lookback: self.lookback,
            horizon: self.horizon,
            end: self.end,
            sampling: self.sampling,
            samples: 1,
            seed: self.seed,
        })
    }
}

/// The cases of a cases file, in order.
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

/// The case named `name` of a cases file, as a request.
fn request(cases_path: &Path, name: &str, series: &Path) -> Result<Request> {
    let cases = cases(cases_path)?;
    let (_, case) = cases
        .iter()
        .find(|(case, _)| case == name)
        .with_context(|| format!("{}: no case {name}", cases_path.display()))?;
    case.request(series)
}

/// Forecasts `request` whole.
fn forecast(engine: &mut Engine, request: &Request) -> Result<Rows> {
    engine.start(request)?;
    loop {
        if let Progress {
            forecast: Some(forecast),
            ..
        } = engine.step()?
        {
            return Ok(forecast);
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| kronos_compile::run(&checkpoint, &out, profile))?,
        Command::Forecast {
            bundle,
            cases: path,
            series,
        } => {
            let mut engine = harness::load(&bundle)?;
            for (name, case) in cases(&path)? {
                println!("# {name}");
                let start = Instant::now();
                let rows = forecast(&mut engine, &case.request(&series)?)?;
                println!("{}", serde_json::to_string_pretty(&rows)?);
                let seconds = start.elapsed().as_secs_f64();
                eprintln!(
                    "{} rows of context, {} forecast: {seconds:.2}s",
                    case.lookback, case.horizon
                );
            }
        }
        Command::Parity {
            bundle,
            cases,
            series,
            goldens,
        } => {
            let mut engine = harness::load(&bundle)?;
            for goldens in &goldens {
                parity::run(&mut engine, &cases, &series, goldens)?;
            }
        }
        Command::Bench {
            bundle,
            cases,
            case,
            series,
        } => bench::run(
            &mut harness::load(&bundle)?,
            &request(&cases, &case, &series)?,
        )?,
    }
    Ok(())
}
