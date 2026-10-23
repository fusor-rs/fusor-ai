//! `d3`: the native host of the runtime, for either size.
//!
//! - `compile`: checkpoint → bundle (the `d3-compile` crate).
//! - `decide`: the requests of a JSON file → their responses, with the time each took.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one request.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use d3::Pixels;
use d3::media::Video;
use d3::request::Request;
use d3::session::Engine;
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "d3", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a checkpoint directory (its safetensors, readout, decision configuration and
    /// tokenizer.json, as the product's scripts/fetch-checkpoint.sh downloads them) into a
    /// bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix", value_parser = compile::profile::<d3_compile::Profile>)]
        profile: d3_compile::Profile,
    },
    /// Decide every request of a JSON file (`{"<case>": request}`), each under a `# <case>` line.
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

/// A request as a JSON file holds it: the request, and its images' and videos' paths. A video
/// is read as its frames: `<video>.frames/{i:03}.png` and their times, `times.json`.
#[derive(Deserialize)]
struct Case {
    #[serde(flatten)]
    request: Request,
    #[serde(default)]
    images: Vec<PathBuf>,
    #[serde(default)]
    videos: Vec<PathBuf>,
}

impl Case {
    fn images(&self, dir: &Path) -> Result<Vec<Pixels>> {
        let images = self
            .images
            .iter()
            .map(|path| harness::image(&dir.join(path)));
        Ok(images.collect::<Result<_, _>>()?)
    }

    fn videos(&self, dir: &Path) -> Result<Vec<Video>> {
        self.videos
            .iter()
            .map(|path| {
                let mut frames_dir = dir.join(path).into_os_string();
                frames_dir.push(".frames");
                video(Path::new(&frames_dir))
            })
            .collect()
    }
}

/// The frames in `dir`, in order, and their times.
fn video(dir: &Path) -> Result<Video> {
    let times_path = dir.join("times.json");
    let json =
        std::fs::read_to_string(&times_path).with_context(|| times_path.display().to_string())?;
    let times: Vec<f64> =
        serde_json::from_str(&json).with_context(|| times_path.display().to_string())?;
    let frames = (0..times.len())
        .map(|index| harness::image(&dir.join(format!("{index:03}.png"))))
        .collect::<Result<_, _>>()?;
    Ok(Video { frames, times })
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
        } => harness::compile(|| d3_compile::run(&checkpoint, &out, profile))?,
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
            let (images, videos) = (case.images(&media)?, case.videos(&media)?);
            bench::run(
                &mut harness::load(&bundle)?,
                &case.request,
                &images,
                &videos,
            )?;
        }
    }
    Ok(())
}

fn decide(engine: &mut Engine, cases: &[(String, Case)], media: &Path) -> Result<()> {
    for (name, case) in cases {
        println!("# {name}");
        let start = Instant::now();
        engine.start(&case.request, &case.images(media)?, &case.videos(media)?)?;
        let mut progress = engine.step()?;
        while progress.response.is_none() {
            progress = engine.step()?;
        }
        let response = progress.response.context("the last step answers")?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{} prompt tokens over {} questions: {seconds:.2}s ({:.0} tokens/s)",
            response.usage.input_tokens,
            response.answers.len(),
            response.usage.input_tokens as f64 / seconds
        );
    }
    Ok(())
}
