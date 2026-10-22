//! `__name__`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `__name__-compile` crate).

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "__name__", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "q7", value_parser = compile::profile::<__crate___compile::Profile>)]
        profile: __crate___compile::Profile,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| __crate___compile::run(&checkpoint, &out, profile))?,
    }
    Ok(())
}
