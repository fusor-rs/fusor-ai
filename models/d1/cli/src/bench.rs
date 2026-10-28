//! `d1 bench`: the milliseconds of each stage on one request (seeing its first image, hearing
//! its first clip, reading its first pack and answering it), and of deciding it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use d1::Pixels;
use d1::model::HIDDEN;
use d1::pipeline::Stage;
use d1::request::Request;
use d1::session::Engine;
use harness::bench::{row, time};

const CALLS: usize = 3;

pub fn run(
    engine: &mut Engine,
    request: &Request,
    images: &[Pixels],
    clips: &[Vec<f32>],
) -> Result<()> {
    let patches = images
        .iter()
        .map(|image| engine.patches(image))
        .collect::<Result<Vec<_>, _>>()?;
    let features = clips
        .iter()
        .map(|clip| engine.features(clip))
        .collect::<Result<Vec<_>, _>>()?;
    let seen: Vec<Vec<f32>> = patches.iter().map(|patches| engine.see(patches)).collect();
    let heard = features
        .iter()
        .map(|features| engine.hear(features))
        .collect::<Result<Vec<_>, _>>()?;
    let counts = |items: &[Vec<f32>]| -> Vec<usize> {
        items.iter().map(|tokens| tokens.len() / HIDDEN).collect()
    };
    let prepared = engine.prepare(request, &counts(&seen), &counts(&heard))?;
    let (image_tokens, audio_tokens) = (seen.concat(), heard.concat());
    let rows: Vec<usize> = prepared
        .packs
        .iter()
        .map(|pack| pack.packed.ids.len())
        .collect();
    println!(
        "threads {}, {} images, {} clips, {} questions, packs of {rows:?} rows",
        engine.threads(),
        images.len(),
        clips.len(),
        prepared.questions.len(),
    );
    for &stage in Stage::ALL {
        let milliseconds = match stage {
            Stage::See => match patches.first() {
                Some(patches) => timed(|| engine.see(patches)),
                None => continue,
            },
            Stage::Hear => match features.first() {
                Some(features) => time(CALLS, || engine.hear(features))?,
                None => continue,
            },
            Stage::Read => {
                let packed = &prepared.packs[0].packed;
                time(CALLS, || {
                    let normed = engine.read(packed, &image_tokens, &audio_tokens);
                    engine.answers(&prepared, 0, &normed)
                })?
            }
        };
        row(stage, milliseconds, "");
    }
    let start = Instant::now();
    engine.start(request, images, clips)?;
    while engine.step()?.response.is_none() {}
    println!(
        "  decide whole {:9.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, std::convert::Infallible>(call()));
    milliseconds
}
