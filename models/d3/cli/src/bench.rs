//! `d3 bench`: the milliseconds of each stage on one request (the vision stages over its first
//! image or video group, the language stages over its first answerable question), and of
//! deciding it whole.

use std::time::Instant;

use anyhow::{Context, Result};
use bundle::Engine as _;
use bundle::Stages as _;
use d3::Pixels;
use d3::media::{Media, Video};
use d3::pipeline::Stage;
use d3::request::Request;
use d3::session::Engine;
use harness::bench::{row, time};

const CALLS: usize = 3;

pub fn run(
    engine: &mut Engine,
    request: &Request,
    images: &[Pixels],
    videos: &[Video],
) -> Result<()> {
    let media = Media::new(images, videos)?;
    let questions = request.questions()?;
    let question = questions
        .iter()
        // A question that failed its check is not read: it is answered with the error.
        .find_map(|asked| asked.question.as_ref().ok())
        .context("the request has no answerable question")?;
    let prompt = engine.prompt(&request.state, question, &media)?;
    let item = media.items().next();
    let embedded = item.map(|patches| engine.embed(patches));
    let encoded = item
        .zip(embedded.as_deref())
        .map(|(patches, embedded)| engine.encode(embedded, patches.grid));
    let tokens: Vec<f32> = media
        .items()
        .flat_map(|patches| engine_see(engine, patches))
        .collect();
    let normed = engine.read(&prompt, &tokens);
    println!(
        "threads {}, {} images, {} videos, {} questions, {} rows a prompt",
        engine.threads(),
        images.len(),
        videos.len(),
        questions.len(),
        prompt.ids.len()
    );
    for &stage in Stage::ALL {
        let milliseconds = match (stage, item, &embedded, &encoded) {
            (Stage::Preprocess, Some(_), _, _) => time(CALLS, || Media::new(images, videos))?,
            (Stage::Embed, Some(patches), _, _) => timed(|| engine.embed(patches)),
            (Stage::Encode, Some(patches), Some(embedded), _) => {
                timed(|| engine.encode(embedded, patches.grid))
            }
            (Stage::Merge, Some(patches), _, Some(encoded)) => {
                timed(|| engine.merge(encoded, patches.grid))
            }
            (Stage::Read, ..) => timed(|| engine.read(&prompt, &tokens)),
            (Stage::Answer, ..) => timed(|| engine.probabilities(&normed, question.keys.len())),
            _ => continue,
        };
        row(stage, milliseconds, "");
    }
    let start = Instant::now();
    engine.start(request, images, videos)?;
    while engine.step()?.response.is_none() {}
    println!(
        "  decide whole {:9.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// An item's tokens through the tower and the merger, stage by stage.
fn engine_see(engine: &mut Engine, patches: &d3::Patches) -> Vec<f32> {
    let embedded = engine.embed(patches);
    let encoded = engine.encode(&embedded, patches.grid);
    engine.merge(&encoded, patches.grid)
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, std::convert::Infallible>(call()));
    milliseconds
}
