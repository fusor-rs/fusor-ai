//! `parakeet`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `parakeet-compile` crate).
//! - `transcribe`: 16 kHz WAV files → timed sentences, printing the real-time factor.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds per second of audio of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use parakeet::SAMPLE_RATE;
use parakeet::session::Engine;

#[derive(Parser)]
#[command(name = "parakeet", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (config.json, ternary.json, tokenizer.json and
    /// model.safetensors, as the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "ternary", value_parser = compile::profile::<parakeet_compile::Profile>)]
        profile: parakeet_compile::Profile,
    },
    /// Transcribe WAV files (16 kHz; channels are averaged), each under a `# <file>` line.
    Transcribe {
        bundle: PathBuf,
        #[arg(required = true)]
        recordings: Vec<PathBuf>,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each stage on a WAV file (16 kHz, at most 30 s).
    Bench { bundle: PathBuf, recording: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| parakeet_compile::run(&checkpoint, &out, profile))?,
        Command::Transcribe { bundle, recordings } => {
            transcribe(&mut harness::load(&bundle)?, &recordings)?;
        }
        Command::Parity { bundle, goldens } => parity::run(&mut harness::load(&bundle)?, &goldens)?,
        Command::Bench { bundle, recording } => {
            bench::run(&mut harness::load(&bundle)?, &read(&recording)?)?
        }
    }
    Ok(())
}

/// The samples of a 16 kHz WAV file, channels averaged.
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let (samples, rate) = parakeet::wav::decode(&bytes)?;
    ensure!(
        rate == SAMPLE_RATE,
        "{}: {rate} Hz; resample it to {SAMPLE_RATE} Hz (ffmpeg -ar {SAMPLE_RATE})",
        path.display()
    );
    Ok(samples)
}

fn transcribe(engine: &mut Engine, recordings: &[PathBuf]) -> Result<()> {
    let (mut audio, mut compute, mut segments) = (0.0, 0.0, 0);
    for recording in recordings {
        let pcm = read(recording)?;
        println!("# {}", recording.display());
        let start = Instant::now();
        let plan = engine.start(pcm)?;
        loop {
            let progress = engine.step()?;
            for sentence in &progress.sentences {
                println!(
                    "[{:7.2} – {:7.2}] {}",
                    sentence.start, sentence.end, sentence.text
                );
            }
            if progress.done {
                break;
            }
        }
        compute += start.elapsed().as_secs_f64();
        audio += plan.seconds;
        segments += plan.segments.len();
    }
    eprintln!(
        "{audio:.2}s of audio in {} recording(s), {segments} segment(s), {compute:.2}s: RTF {:.4}",
        recordings.len(),
        compute / audio
    );
    Ok(())
}
