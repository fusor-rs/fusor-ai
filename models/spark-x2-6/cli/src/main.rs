//! `spark-x2-6`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `spark-x2-6-compile` crate).
//! - `chat`: the conversations of a JSON file, turn by turn → each reply, with its speed.
//! - `parity`: every stage against the reference goldens.
//! - `bench`: the milliseconds of each stage on one conversation's first turn.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use spark_x2_6::session::{Engine, Message};

#[derive(Parser)]
#[command(name = "spark-x2-6", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (its safetensors shards and tokenizer.json) into a
    /// bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<spark_x2_6_compile::Profile>)]
        profile: spark_x2_6_compile::Profile,
    },
    /// Reply to every turn of every conversation of a JSON file (`{"<case>": {"turns": […]}}`),
    /// each conversation under a `# <case>` line.
    Chat {
        bundle: PathBuf,
        conversations: PathBuf,
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
    },
}

/// A conversation as a JSON file holds it: its messages, in order.
#[derive(Deserialize)]
struct Conversation {
    turns: Vec<Message>,
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
        } => harness::compile(|| spark_x2_6_compile::run(&checkpoint, &out, profile))?,
        Command::Chat {
            bundle,
            conversations: path,
        } => chat(&mut harness::load(&bundle)?, &conversations(&path)?)?,
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
        } => {
            let conversations = conversations(&path)?;
            let (_, conversation) = conversations
                .iter()
                .find(|(name, _)| *name == case)
                .with_context(|| format!("{}: no conversation {case}", path.display()))?;
            let message = conversation
                .turns
                .first()
                .context("a conversation has turns")?;
            bench::run(&mut harness::load(&bundle)?, message)?;
        }
    }
    Ok(())
}

fn chat(engine: &mut Engine, conversations: &[(String, Conversation)]) -> Result<()> {
    for (name, conversation) in conversations {
        println!("# {name}");
        engine.reset();
        for message in &conversation.turns {
            let start = Instant::now();
            let plan = engine.start(message)?;
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
                plan.prompt_tokens, progress.tokens
            );
        }
    }
    Ok(())
}
