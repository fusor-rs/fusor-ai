//! `phonon`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `phonon-compile` crate).
//! - `synth`: text → WAV with a bundle, printing the real-time factor.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds per frame of each stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use phonon::SAMPLE_RATE;
use phonon::prompt::VOICES;
use phonon::session::{Engine, Request};

#[derive(Parser)]
#[command(name = "phonon", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (config.json, tokenizer.json, model.safetensors,
    /// default-voice.safetensors and voices/ from gradium/phonon-voices, as
    /// the product's scripts/fetch-checkpoint.sh downloads them) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<phonon_compile::Profile>)]
        profile: phonon_compile::Profile,
    },
    /// Synthesize `text` into a WAV file.
    Synth {
        bundle: PathBuf,
        text: String,
        #[arg(short, long, default_value = "out.wav")]
        out: PathBuf,
        /// A voice of the bundle (default: the first).
        #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(VOICES.map(|(id, _)| id)))]
        voice: Option<String>,
        /// Sampling temperature (default: the model's).
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each stage of the per-frame loop.
    Bench { bundle: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| phonon_compile::run(&checkpoint, &out, profile))?,
        Command::Synth {
            bundle,
            text,
            out,
            voice,
            temperature,
            seed,
        } => {
            let request = Request {
                text,
                voice: voice.unwrap_or_else(|| VOICES[0].0.into()),
                seed,
                temperature,
            };
            synth(&mut harness::load(&bundle)?, request, &out)?;
        }
        Command::Parity { bundle, goldens } => parity::run(&mut harness::load(&bundle)?, &goldens)?,
        Command::Bench { bundle } => bench::run(&mut harness::load(&bundle)?)?,
    }
    Ok(())
}

fn synth(engine: &mut Engine, request: Request, out: &Path) -> Result<()> {
    let start = Instant::now();
    let plan = engine.start(request)?;
    let mut audio = Vec::new();
    let mut frames = 0;
    let mut first_audio = None;
    loop {
        let progress = engine.step()?;
        frames += 1;
        audio.extend(engine.take_audio());
        if first_audio.is_none() && !audio.is_empty() {
            first_audio = Some(start.elapsed().as_secs_f64());
        }
        if progress.done {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);
    std::fs::write(out, phonon::wav::encode(&audio, SAMPLE_RATE))?;
    println!(
        "{} fragment(s), {frames} frames, {seconds:.2}s audio in {elapsed:.2}s: RTF {:.3}, {:.1} ms/frame, first audio {:.0} ms -> {}",
        plan.fragments.len(),
        elapsed / seconds,
        elapsed * 1000.0 / frames as f64,
        first_audio.unwrap_or(elapsed) * 1000.0,
        out.display()
    );
    Ok(())
}
