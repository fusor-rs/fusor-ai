//! `kronos bench`: the milliseconds of each stage on one case, and of its forecast whole.

use std::convert::Infallible;
use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use kronos::model::MAX_CONTEXT;
use kronos::pipeline::Stage;
use kronos::predictor::Window;
use kronos::series::{Normalized, Series};
use kronos::session::{Engine, Request};
use kronos::tokenizer::{self, Token};

const CALLS: usize = 3;

pub fn run(engine: &mut Engine, request: &Request) -> Result<()> {
    let series = Series::parse(&request.csv)?;
    let context = request.context(&series)?;
    let rows = &series.rows[context.clone()];
    let normalized = Normalized::new(rows);
    let tokens = tokenizer::tokens(&engine.codes(&normalized.values));
    let stamps: Vec<_> = series.times[context]
        .iter()
        .map(|time| time.stamp())
        .collect();
    let window = Window {
        tokens: &tokens,
        stamps: &stamps,
    };
    println!(
        "threads {}, {} rows of context, {} forecast",
        engine.threads(),
        request.lookback,
        request.horizon
    );
    let full_window: Vec<Token> = tokens.iter().copied().cycle().take(MAX_CONTEXT).collect();
    for &stage in Stage::ALL {
        let (milliseconds, unit) = match stage {
            Stage::Normalize => (timed(|| Normalized::new(rows)), ""),
            Stage::Encode => (timed(|| engine.codes(&normalized.values)), ""),
            Stage::Read => (timed(|| engine.logits(window, 0)), ""),
            Stage::Sample => (sample(engine, request)?, " per row"),
            Stage::Decode => (timed(|| engine.decode(&full_window)), ""),
        };
        row(stage, milliseconds, unit);
    }
    let start = Instant::now();
    crate::forecast(engine, request)?;
    println!(
        "  forecast whole {:9.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, Infallible>(call()));
    milliseconds
}

/// The milliseconds of each step after the first (which reads the context), the last's decoding
/// aside.
fn sample(engine: &mut Engine, request: &Request) -> Result<f64> {
    engine.start(request)?;
    engine.step()?;
    let start = Instant::now();
    for _ in 2..request.horizon {
        engine.step()?;
    }
    let steps = request.horizon.saturating_sub(2).max(1);
    let milliseconds = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    engine.step()?;
    Ok(milliseconds)
}
