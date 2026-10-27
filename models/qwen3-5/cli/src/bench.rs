//! `qwen3-5 bench`: the milliseconds of each stage on a conversation's first turn (the vision
//! stages over its first image or video group, the prompt read from its first row, one token of
//! the reply), and of replying to it whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use kernels::numpy::argmax;
use qwen3_5::media::Media;
use qwen3_5::pipeline::Stage;
use qwen3_5::prompt::Turn;
use qwen3_5::session::{Engine, Message};
use qwen3_5::{Patches, Pixels, Video};

const CALLS: usize = 3;

pub fn run(
    engine: &mut Engine,
    message: &Message,
    images: &[Pixels],
    videos: &[Video],
) -> Result<()> {
    let media = Media::new(images, videos)?;
    let turn = Turn {
        text: message.text.clone(),
        media: media.kept(),
    };
    let prompt = engine.prompt(&message.system, &[], &turn, message.thinking)?;
    let item = media.items().next();
    let embedded = item.map(|patches| engine.embed(patches));
    let encoded = item
        .zip(embedded.as_deref())
        .map(|(patches, embedded)| engine.encode(embedded, patches.grid));
    let tokens: Vec<f32> = media
        .items()
        .flat_map(|patches| see(engine, patches))
        .collect();
    let first = argmax(&engine.prefill(&prompt, &tokens).logits) as u32;
    println!(
        "threads {}, {} images, {} videos, {} rows a prompt",
        engine.threads(),
        images.len(),
        videos.len(),
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
            (Stage::Prefill, ..) => timed(|| {
                engine.reset();
                engine.prefill(&prompt, &tokens)
            }),
            (Stage::Decode, ..) => timed(|| engine.force(&prompt, &[first])),
            _ => continue,
        };
        row(stage, milliseconds, "");
    }
    whole(engine, message, images, videos)
}

/// The turn replied to, whole, in a new conversation.
fn whole(
    engine: &mut Engine,
    message: &Message,
    images: &[Pixels],
    videos: &[Video],
) -> Result<()> {
    engine.reset();
    let start = Instant::now();
    engine.start(message, images, videos)?;
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

/// An item's tokens through the tower and the merger, stage by stage.
fn see(engine: &mut Engine, patches: &Patches) -> Vec<f32> {
    let embedded = engine.embed(patches);
    let encoded = engine.encode(&embedded, patches.grid);
    engine.merge(&encoded, patches.grid)
}

/// The milliseconds of a call, the mean of `CALLS` after a first.
fn timed<T>(mut call: impl FnMut() -> T) -> f64 {
    let Ok(milliseconds) = time(CALLS, || Ok::<_, std::convert::Infallible>(call()));
    milliseconds
}
