//! `indra-ocr bench`: the milliseconds of each stage on one page (a stage done in batches
//! summed over them), and of reading it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::row;
use indra_ocr::Pixels;
use indra_ocr::pipeline::Stage;
use indra_ocr::session::{Engine, Options};

pub fn run(engine: &mut Engine, image: &Pixels, options: Options) -> Result<()> {
    engine.read(image, options)?;
    let mut milliseconds = vec![0.0; Stage::ALL.len()];
    let start = Instant::now();
    engine.start(image, options);
    loop {
        let step = Instant::now();
        let progress = engine.step()?;
        let stage = Stage::ALL
            .iter()
            .position(|stage| stage.name() == progress.stage);
        let stage = stage.expect("a step reports one of the stages");
        milliseconds[stage] += step.elapsed().as_secs_f64() * 1000.0;
        if let Some(page) = progress.page {
            println!(
                "threads {}, {} lines, {} blocks, {} batches",
                engine.threads(),
                page.lines.len(),
                page.blocks.len(),
                progress.planned
            );
            break;
        }
    }
    for (&stage, milliseconds) in Stage::ALL.iter().zip(milliseconds) {
        row(stage, milliseconds, "");
    }
    println!(
        "  read whole   {:9.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
