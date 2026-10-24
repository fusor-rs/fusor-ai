//! `qwen3-5`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `qwen3-5-compile` crate).
//! - `chat`: the conversations of a JSON file, turn by turn → each reply, with its speed.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one conversation's first turn.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use qwen3_5::session::{Engine, Message};
use qwen3_5::{Pixels, Video};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "qwen3-5", version)]
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
        #[arg(long, default_value = "q7", value_parser = compile::profile::<qwen3_5_compile::Profile>)]
        profile: qwen3_5_compile::Profile,
    },
    /// Reply to every turn of every conversation of a JSON file (`{"<case>": {"turns": […]}}`),
    /// each conversation under a `# <case>` line.
    Chat {
        bundle: PathBuf,
        conversations: PathBuf,
        /// The directory the turns' media paths are relative to.
        #[arg(long, default_value = ".")]
        media: PathBuf,
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

/// A conversation as a JSON file holds it.
#[derive(Deserialize)]
struct Conversation {
    turns: Vec<Said>,
}

/// One turn: the message, and its images' and videos' paths. A video is read as its frames:
/// `<video>.frames/{i:03}.png` and their times, `times.json`.
#[derive(Deserialize)]
struct Said {
    #[serde(flatten)]
    message: Message,
    #[serde(default)]
    images: Vec<PathBuf>,
    #[serde(default)]
    videos: Vec<PathBuf>,
}

impl Said {
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
        } => harness::compile(|| qwen3_5_compile::run(&checkpoint, &out, profile))?,
        Command::Chat {
            bundle,
            conversations: path,
            media,
        } => chat(&mut harness::load(&bundle)?, &conversations(&path)?, &media)?,
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
            let (images, videos) = (said.images(&media)?, said.videos(&media)?);
            bench::run(
                &mut harness::load(&bundle)?,
                &said.message,
                &images,
                &videos,
            )?;
        }
    }
    Ok(())
}

fn chat(engine: &mut Engine, conversations: &[(String, Conversation)], media: &Path) -> Result<()> {
    for (name, conversation) in conversations {
        println!("# {name}");
        engine.reset();
        for said in &conversation.turns {
            let start = Instant::now();
            let plan = engine.start(&said.message, &said.images(media)?, &said.videos(media)?)?;
            let mut progress = engine.step()?;
            while !progress.done {
                progress = engine.step()?;
            }
            if let Some(reasoning) = &progress.reasoning {
                println!("<think>\n{reasoning}\n</think>");
            }
            println!("{}\n", progress.answer);
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "{} tokens read, {} written: {seconds:.2}s",
                plan.read_tokens, progress.tokens
            );
        }
    }
    Ok(())
}
