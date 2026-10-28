//! `qwen3-tts`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `qwen3-tts-compile` crate).
//! - `speak`: text → WAV in the voice of a clip, printing the real-time factor.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: milliseconds per stage.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use qwen3_tts::model::{Language, SAMPLE_RATE};
use qwen3_tts::session::{Engine, Request};
use qwen3_tts::wav;

#[derive(Parser)]
#[command(name = "qwen3-tts", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (as the product's scripts/fetch-checkpoint.sh downloads
    /// it) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<qwen3_tts_compile::Profile>)]
        profile: qwen3_tts_compile::Profile,
    },
    /// Speak `text` in the voice of a 24 kHz WAV clip.
    Speak {
        bundle: PathBuf,
        voice: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, value_parser = parse_language)]
        language: Language,
        /// What the clip says; the voice is cloned from the clip's speaker embedding alone
        /// without it.
        #[arg(long, default_value = "")]
        reference_text: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(short, long, default_value = "speech.wav")]
        out: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference.
    Parity {
        bundle: PathBuf,
        #[arg(required = true)]
        goldens: Vec<PathBuf>,
    },
    /// Time each stage of a request.
    Bench {
        bundle: PathBuf,
        voice: PathBuf,
        #[arg(long, default_value = "")]
        reference_text: String,
    },
}

fn parse_language(name: &str) -> Result<Language, String> {
    Language::ALL
        .into_iter()
        .find(|language| language.name() == name)
        .ok_or_else(|| {
            let names: Vec<_> = Language::ALL.map(Language::name).into();
            format!("one of {}", names.join(", "))
        })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| qwen3_tts_compile::run(&checkpoint, &out, profile))?,
        Command::Speak {
            bundle,
            voice,
            text,
            language,
            reference_text,
            seed,
            out,
        } => {
            let request = Request {
                text,
                language,
                reference_text,
                seed,
                max_frames: None,
            };
            speak(
                &mut harness::load(&bundle)?,
                &request,
                &read_voice(&voice)?,
                &out,
            )?;
        }
        Command::Parity { bundle, goldens } => {
            let mut engine = harness::load(&bundle)?;
            for path in goldens {
                parity::run(&mut engine, &path)?;
            }
        }
        Command::Bench {
            bundle,
            voice,
            reference_text,
        } => bench::run(
            &mut harness::load(&bundle)?,
            &read_voice(&voice)?,
            &reference_text,
        )?,
    }
    Ok(())
}

/// The samples of a 24 kHz WAV clip.
fn read_voice(path: &Path) -> Result<Vec<f32>> {
    let (samples, rate) = wav::decode(&std::fs::read(path)?)?;
    if rate != SAMPLE_RATE {
        bail!("{}: {rate} Hz, not {SAMPLE_RATE} Hz", path.display());
    }
    Ok(samples)
}

fn speak(engine: &mut Engine, request: &Request, voice: &[f32], out: &Path) -> Result<()> {
    let start = Instant::now();
    let plan = engine.start(request, voice)?;
    let mut audio = Vec::new();
    let mut first_audio = None;
    let similarity = loop {
        let progress = engine.step()?;
        audio.extend(engine.speech());
        if first_audio.is_none() && !audio.is_empty() {
            first_audio = Some(start.elapsed().as_secs_f64());
        }
        if let Some(similarity) = progress.similarity {
            break similarity;
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);
    std::fs::write(out, wav::encode(&audio, SAMPLE_RATE))?;
    println!(
        "{:?} voice, {seconds:.2}s audio in {elapsed:.2}s: RTF {:.3}, first audio {:.0} ms, \
         voice match {similarity:.3} -> {}",
        plan.mode,
        elapsed / seconds,
        first_audio.unwrap_or(elapsed) * 1000.0,
        out.display()
    );
    Ok(())
}
