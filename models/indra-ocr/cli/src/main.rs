//! `indra-ocr`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `indra-ocr-compile` crate).
//! - `read`: the pages of a JSON file → their JSON, as the reference writes it.
//! - `parity`: every network call and pipeline step against the reference goldens.
//! - `bench`: the milliseconds of each stage on one page.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indra_ocr::Pixels;
use indra_ocr::session::{Engine, Options};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "indra-ocr", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (its ONNX networks, language models and meta files, as
    /// the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "f32", value_parser = compile::profile::<indra_ocr_compile::Profile>)]
        profile: indra_ocr_compile::Profile,
    },
    /// Read every page of a JSON file (`{"<case>": {"image": <file>, "math": <bool>}}`), each
    /// under a `# <case>` line.
    Read {
        bundle: PathBuf,
        pages: PathBuf,
        /// The directory the pages' images are in.
        #[arg(long, default_value = ".")]
        images: PathBuf,
    },
    /// Compare every network call and pipeline step with the golden tensors of the reference,
    /// one file per case.
    Parity {
        bundle: PathBuf,
        #[arg(required = true)]
        goldens: Vec<PathBuf>,
    },
    /// Time each stage on one page of a JSON file.
    Bench {
        bundle: PathBuf,
        pages: PathBuf,
        case: String,
        #[arg(long, default_value = ".")]
        images: PathBuf,
    },
}

/// A page as a JSON file lists it: its image, and whether its math is read.
#[derive(Deserialize)]
struct Case {
    image: PathBuf,
    #[serde(default)]
    math: bool,
}

impl Case {
    fn options(&self) -> Options {
        Options { math: self.math }
    }

    fn pixels(&self, dir: &Path) -> Result<Pixels> {
        Ok(harness::image(&dir.join(&self.image))?)
    }
}

/// The cases of a pages file, in order.
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
        } => harness::compile(|| indra_ocr_compile::run(&checkpoint, &out, profile))?,
        Command::Read {
            bundle,
            pages,
            images,
        } => read(&mut harness::load(&bundle)?, &cases(&pages)?, &images)?,
        Command::Parity { bundle, goldens } => {
            let mut engine = harness::load(&bundle)?;
            for goldens in &goldens {
                parity::run(&mut engine, goldens)?;
            }
        }
        Command::Bench {
            bundle,
            pages,
            case,
            images,
        } => {
            let cases = cases(&pages)?;
            let (_, case) = cases
                .iter()
                .find(|(name, _)| *name == case)
                .with_context(|| format!("{}: no case {case}", pages.display()))?;
            bench::run(
                &mut harness::load(&bundle)?,
                &case.pixels(&images)?,
                case.options(),
            )?;
        }
    }
    Ok(())
}

fn read(engine: &mut Engine, cases: &[(String, Case)], images: &Path) -> Result<()> {
    for (name, case) in cases {
        println!("# {name}");
        let start = Instant::now();
        let page = engine.read(&case.pixels(images)?, case.options())?;
        println!("{}", serde_json::to_string_pretty(&page)?);
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{} lines, {} blocks: {seconds:.2}s",
            page.lines.len(),
            page.blocks.len()
        );
    }
    Ok(())
}
