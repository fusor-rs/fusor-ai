//! `spark-x2-6 bench`: the milliseconds of each stage on a conversation's first turn (the prompt
//! read from its first row, one token of the reply), and of replying to it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use kernels::numpy::argmax;
use spark_x2_6::pipeline::Stage;
use spark_x2_6::session::{Engine, Message};

const CALLS: usize = 3;

pub fn run(engine: &mut Engine, message: &Message) -> Result<()> {
    let prompt = engine.prompt(&message.system, &[], &message.text, message.thinking)?;
    let first = argmax(&engine.prefill(&prompt).logits) as u32;
    println!(
        "threads {}, {} rows a prompt",
        engine.threads(),
        prompt.len()
    );
    for &stage in Stage::ALL {
        let milliseconds = match stage {
            Stage::Prefill => timed(|| engine.prefill(&prompt)),
            Stage::Decode => timed(|| engine.force(&prompt, &[first])),
        };
        row(stage, milliseconds, "");
    }
    whole(engine, message)
}

/// The turn replied to, whole, in a new conversation.
fn whole(engine: &mut Engine, message: &Message) -> Result<()> {
    engine.reset();
    let start = Instant::now();
    engine.start(message)?;
    let mut progress = engine.step()?;
    while !progress.done {
        progress = engine.step()?;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "  reply whole  {:9.1} ms, {} tokens ({:.1} tokens/s)",
        seconds * 1000.0,
        progress.tokens,
        progress.tokens as f64 / seconds
    );
    Ok(())
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, std::convert::Infallible>(call()));
    milliseconds
}
