//! `lfm3`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `lfm3-compile` crate).
//! - `chat`: the conversations of a JSON file, turn by turn, with their images and clips → each
//!   reply, with its speed, and its speech as a WAV file.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one conversation's first turn.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use lfm3::Pixels;
use lfm3::audio::{SAMPLE_RATE, SPEECH_RATE};
use lfm3::session::{Engine, Message};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "lfm3", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (its safetensors file, config.json and
    /// tokenizer.json, and the audio variants' audio_tokenizer/) of LFM3-1.7B, LFM3-VL-2B,
    /// LFM3-Audio-2B or LFM3-Omni-2B into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<lfm3_compile::Profile>)]
        profile: lfm3_compile::Profile,
    },
    /// Reply to every turn of every conversation of a JSON file (`{"<case>": {"turns": […]}}`),
    /// each conversation under a `# <case>` line, each spoken reply into
    /// `<speech>/<case>.<turn>.wav`.
    Chat {
        bundle: PathBuf,
        conversations: PathBuf,
        /// The directory the turns' image and clip paths are relative to.
        #[arg(long, default_value = ".")]
        media: PathBuf,
        #[arg(long, default_value = "speech")]
        speech: PathBuf,
    },
    /// Compare every stage with the golden tensors of the reference, one file per case.
    Parity {
        bundle: PathBuf,
        #[arg(required = true)]
        goldens: Vec<PathBuf>,
    },
    /// Time each stage on the first turn of one conversation of a JSON file.
    Bench {
        bundle: PathBuf,
        conversations: PathBuf,
        case: String,
        #[arg(long, default_value = ".")]
        media: PathBuf,
    },
}

/// A conversation as a JSON file holds it: its turns, in order.
#[derive(Deserialize)]
struct Conversation {
    turns: Vec<Said>,
}

/// One turn: the message, and its images' and 16 kHz WAV clips' paths.
#[derive(Deserialize)]
struct Said {
    #[serde(flatten)]
    message: Message,
    #[serde(default)]
    images: Vec<PathBuf>,
    #[serde(default)]
    audio: Vec<PathBuf>,
}

impl Said {
    fn images(&self, dir: &Path) -> Result<Vec<Pixels>> {
        let images = self
            .images
            .iter()
            .map(|path| harness::image(&dir.join(path)));
        Ok(images.collect::<Result<_, _>>()?)
    }

    fn clips(&self, dir: &Path) -> Result<Vec<Vec<f32>>> {
        self.audio
            .iter()
            .map(|path| clip(&dir.join(path)))
            .collect()
    }
}

/// A WAV file's mono samples, at the rate the model hears.
fn clip(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let (samples, rate) = media::wav::decode(&bytes).with_context(|| path.display().to_string())?;
    ensure!(
        rate == SAMPLE_RATE,
        "{}: {rate} Hz, not the {SAMPLE_RATE} Hz the model hears",
        path.display()
    );
    Ok(samples)
}

/// The conversations of a file, in order.
fn conversations(path: &Path) -> Result<Vec<(String, Conversation)>> {
    let json = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let cases: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&json).with_context(|| path.display().to_string())?;
    cases
        .into_iter()
        .map(|(name, case)| {
            let case = Conversation::deserialize(case).with_context(|| name.clone())?;
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
        } => harness::compile(|| lfm3_compile::run(&checkpoint, &out, profile))?,
        Command::Chat {
            bundle,
            conversations: path,
            media,
            speech,
        } => {
            let dirs = Dirs {
                media: &media,
                speech: &speech,
            };
            chat(&mut harness::load(&bundle)?, &conversations(&path)?, dirs)?;
        }
        Command::Parity { bundle, goldens } => {
            let mut engine = harness::load(&bundle)?;
            for goldens in &goldens {
                parity::run(&mut engine, goldens)?;
            }
        }
        Command::Bench {
            bundle,
            conversations: path,
            case,
            media,
        } => {
            let conversations = conversations(&path)?;
            let (_, conversation) = conversations
                .iter()
                .find(|(name, _)| *name == case)
                .with_context(|| format!("{}: no conversation {case}", path.display()))?;
            let said = conversation
                .turns
                .first()
                .context("a conversation has turns")?;
            let inputs = bench::Inputs {
                message: &said.message,
                images: &said.images(&media)?,
                clips: &said.clips(&media)?,
            };
            bench::run(&mut harness::load(&bundle)?, &inputs)?;
        }
    }
    Ok(())
}

/// Where `chat` reads the turns' media from and writes the replies' speech to.
#[derive(Clone, Copy)]
struct Dirs<'dirs> {
    media: &'dirs Path,
    speech: &'dirs Path,
}

fn chat(engine: &mut Engine, conversations: &[(String, Conversation)], dirs: Dirs) -> Result<()> {
    for (name, conversation) in conversations {
        println!("# {name}");
        engine.reset();
        for (turn, said) in conversation.turns.iter().enumerate() {
            let start = Instant::now();
            let (images, clips) = (said.images(dirs.media)?, said.clips(dirs.media)?);
            let plan = engine.start(&said.message, &images, &clips)?;
            let mut progress = engine.step()?;
            let mut speech = engine.speech();
            while !progress.done {
                progress = engine.step()?;
                speech.extend(engine.speech());
            }
            if let Some(reasoning) = &progress.reasoning {
                println!("<think>\n{reasoning}\n</think>");
            }
            println!("{}\n", progress.answer);
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "{} tokens read, {} written, {} frames spoken: {seconds:.2}s",
                plan.prompt_tokens, progress.tokens, progress.frames
            );
            if !speech.is_empty() {
                let path = dirs.speech.join(format!("{name}.{turn}.wav"));
                std::fs::create_dir_all(dirs.speech)?;
                std::fs::write(&path, media::wav::encode(&speech, SPEECH_RATE))?;
                eprintln!("speech: {}", path.display());
            }
        }
    }
    Ok(())
}
