//! `oxr bench`: the milliseconds of each stage on one image, and of reading it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use oxr::pipeline::Stage;
use oxr::prompt::Task;
use oxr::session::{Engine, Request};
use oxr::{Patches, Pixels};

const CALLS: usize = 3;
/// Answer tokens the decode stage is timed over.
const DECODE_TOKENS: usize = 32;

pub fn run(engine: &mut Engine, image: &Pixels, task: Task) -> Result<()> {
    let patches = Patches::of(image)?;
    let grid = patches.grid;
    let prompt = engine.prompt(task, grid)?;
    println!(
        "threads {}, {}×{} image, {}×{} patches, {} prompt tokens",
        engine.threads(),
        image.size.width,
        image.size.height,
        grid.rows,
        grid.columns,
        prompt.ids.len()
    );
    let embedded = engine.embed(&patches);
    let encoded = engine.encode(&embedded, grid);
    let tokens = engine.tokenize(&encoded, grid);
    for &stage in Stage::ALL {
        let ms = match stage {
            Stage::Preprocess => time(CALLS, || Patches::of(image))?,
            Stage::Embed => time(CALLS, || anyhow::Ok(engine.embed(&patches)))?,
            Stage::Encode => time(CALLS, || anyhow::Ok(engine.encode(&embedded, grid)))?,
            Stage::Tokenize => time(CALLS, || anyhow::Ok(engine.tokenize(&encoded, grid)))?,
            Stage::Prefill => time(CALLS, || anyhow::Ok(engine.prefill(&prompt, &tokens)))?,
            Stage::Decode => {
                let answer = |engine: &mut Engine| engine.answer(&prompt, &tokens, DECODE_TOKENS);
                let written = answer(engine).len();
                let with_prefill = time(CALLS, || anyhow::Ok(answer(engine)))?;
                let prefill = time(CALLS, || anyhow::Ok(engine.prefill(&prompt, &tokens)))?;
                (with_prefill - prefill) / written.saturating_sub(1).max(1) as f64
            }
        };
        row(stage, ms, "");
    }
    let request = Request {
        task,
        max_tokens: None,
    };
    let start = Instant::now();
    engine.start(image, &request)?;
    let mut progress = engine.step()?;
    while !progress.done {
        progress = engine.step()?;
    }
    println!(
        "  read whole   {:9.1} ms, {} tokens",
        start.elapsed().as_secs_f64() * 1000.0,
        progress.tokens
    );
    Ok(())
}
