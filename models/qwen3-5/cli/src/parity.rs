//! `qwen3-5 parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its images and video frames, its patches, its
//! encoder output, its media tokens, its replies), so errors do not compound from one stage to
//! the next. Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); each turn's prompt ids
//! must be the reference's, one for one, its reply's first token the reference's, and most of
//! the reply's next tokens, read after the reference's, the reference's too. The turns are read
//! as a conversation is: each from the last reply's opener, on the state saved there.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use kernels::numpy::argmax;
use qwen3_5::media::Media;
use qwen3_5::model::HIDDEN;
use qwen3_5::pipeline::Stage;
use qwen3_5::prompt::{ENDS, Prompt, Replied, Turn};
use qwen3_5::session::Engine;
use qwen3_5::{Grid, Patches, Pixels, Video};

use crate::Conversation;

/// The share of a reply's teacher-forced tokens whose likeliest next token must be the
/// reference's: the quantized weights move near ties.
const AGREEMENT: f64 = 0.9;

/// One image's or video group's tensors (`reference/dump.py` documents them).
struct Item {
    patches: Patches,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    merged: Vec<f32>,
}

/// One turn's tensors.
struct Answered {
    ids: Vec<i64>,
    reply: Vec<u32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

/// The reference's tensors for one conversation.
struct Goldens {
    conversation: Conversation,
    images: Vec<Pixels>,
    videos: Vec<Video>,
    items: Vec<Item>,
    turns: Vec<Answered>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let count = |prefix: &str, last: &str| {
            (0..)
                .take_while(|index| file.contains(&format!("{prefix}.{index}{last}")))
                .count()
        };
        let images = (0..count("image", ""))
            .map(|index| pixels(&file, &format!("image.{index}"), 0))
            .collect::<Result<_>>()?;
        let videos = (0..count("video", ".frames"))
            .map(|index| video(&file, index))
            .collect::<Result<_>>()?;
        let items = (0..count("item", ".grid")).map(|index| item(&file, index));
        let turns = (0..count("turn", ".ids")).map(|index| {
            let name = |part: &str| format!("turn.{index}.{part}");
            let reply = file.i64(&name("reply"))?.into_iter().map(u32::try_from);
            Ok(Answered {
                ids: file.i64(&name("ids"))?,
                reply: reply.collect::<Result<_, _>>()?,
                hidden: file.f32(&name("hidden"))?,
                logits: file.f32(&name("logits"))?,
            })
        });
        Ok(Self {
            conversation: serde_json::from_slice(&file.u8("conversation")?)?,
            images,
            videos,
            items: items.collect::<Result<_>>()?,
            turns: turns.collect::<Result<_>>()?,
        })
    }
}

/// Frame `frame` of the RGB frames under `name`, `[frames, height, width, 3]` or `[height,
/// width, 3]`.
fn pixels(file: &Checkpoint, name: &str, frame: usize) -> Result<Pixels> {
    let shape = file.shape(name)?;
    let &[height, width, channels] = &shape[shape.len().saturating_sub(3)..] else {
        bail!("{name}: shape {shape:?}, not frames of height, width and channels");
    };
    let size = height * width * channels;
    let values = file.u8(name)?;
    Ok(Pixels {
        size: qwen3_5::Size { width, height },
        channels,
        values: values[frame * size..][..size].to_vec(),
    })
}

fn video(file: &Checkpoint, index: usize) -> Result<Video> {
    let name = format!("video.{index}.frames");
    let frames = file.shape(&name)?[0];
    Ok(Video {
        frames: (0..frames)
            .map(|frame| pixels(file, &name, frame))
            .collect::<Result<_>>()?,
        times: file.f64(&format!("video.{index}.times"))?,
    })
}

fn item(file: &Checkpoint, index: usize) -> Result<Item> {
    let name = |part: &str| format!("item.{index}.{part}");
    let grid = file.i64(&name("grid"))?;
    Ok(Item {
        patches: Patches {
            grid: Grid {
                rows: usize::try_from(grid[0])?,
                columns: usize::try_from(grid[1])?,
            },
            values: file.f32(&name("patches"))?,
        },
        embedded: file.f32(&name("embedded"))?,
        encoded: file.f32(&name("encoded"))?,
        merged: file.f32(&name("merged"))?,
    })
}

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Goldens::read(path)?;
    println!("# {}", path.display());
    let media = turn_media(&goldens)?;
    for &stage in Stage::ALL {
        match stage {
            Stage::Preprocess => preprocess(&media, &goldens)?,
            Stage::Embed | Stage::Encode | Stage::Merge => vision(engine, stage, &goldens)?,
            Stage::Prefill => conversation(engine, &media, &goldens)?,
            // Each turn's reply is read right after its prompt, on the state the prompt leaves.
            Stage::Decode => {}
        }
    }
    Ok(())
}

/// Each turn's images and videos, from the reference's, in order.
fn turn_media(goldens: &Goldens) -> Result<Vec<Media>> {
    let (mut images, mut videos) = (goldens.images.as_slice(), goldens.videos.as_slice());
    let media = goldens.conversation.turns.iter().map(|said| {
        let (turn_images, rest) = images.split_at(said.images.len());
        images = rest;
        let (turn_videos, rest) = videos.split_at(said.videos.len());
        videos = rest;
        Media::new(turn_images, turn_videos)
    });
    Ok(media.collect::<Result<_, _>>()?)
}

fn preprocess(media: &[Media], goldens: &Goldens) -> Result<()> {
    let items = media.iter().flat_map(Media::items);
    for (index, (ours, golden)) in items.zip(&goldens.items).enumerate() {
        ensure!(
            ours.grid == golden.patches.grid,
            "preprocess {index}: grid {:?}, reference {:?}",
            ours.grid,
            golden.patches.grid
        );
        let name = format!("preprocess {index}");
        println!("{}", compare(&name, &ours.values, &golden.patches.values)?);
    }
    Ok(())
}

fn vision(engine: &mut Engine, stage: Stage, goldens: &Goldens) -> Result<()> {
    for (index, item) in goldens.items.iter().enumerate() {
        let grid = item.patches.grid;
        let (ours, golden) = match stage {
            Stage::Embed => (engine.embed(&item.patches), &item.embedded),
            Stage::Encode => (engine.encode(&item.embedded, grid), &item.encoded),
            _ => (engine.merge(&item.encoded, grid), &item.merged),
        };
        println!(
            "{}",
            compare(&format!("{} {index}", stage.name()), &ours, golden)?
        );
    }
    Ok(())
}

/// Every turn, in order: its prompt after the reference's replies, its last row's state and
/// logits on the reference's media tokens, then its reply's, teacher-forced.
fn conversation(engine: &mut Engine, media: &[Media], goldens: &Goldens) -> Result<()> {
    engine.reset();
    let mut history = Vec::new();
    let mut items = goldens.items.iter();
    let turns = goldens.conversation.turns.iter().zip(media);
    for (index, ((said, media), golden)) in turns.zip(&goldens.turns).enumerate() {
        let turn = Turn {
            text: said.message.text.clone(),
            media: media.kept(),
        };
        let prompt = engine.prompt(&said.message.system, &history, &turn, said.message.thinking)?;
        let ids = prompt.ids.iter().map(|&id| i64::from(id));
        ensure!(
            ids.eq(golden.ids.iter().copied()),
            "turn {index}: the prompt's ids are not the reference's"
        );
        let tokens: Vec<f32> = (&mut items)
            .take(media.items().count())
            .flat_map(|item| item.merged.iter().copied())
            .collect();
        let prefill = engine.prefill(&prompt, &tokens);
        ensure!(
            argmax(&prefill.logits) == golden.reply[0] as usize,
            "turn {index}: the reply's first token is not the reference's"
        );
        let first = &golden.hidden[..HIDDEN];
        println!(
            "prefill {index}: {} ids identical\n{}\n{}",
            prompt.ids.len(),
            compare(&format!("prefill {index} hidden"), &prefill.hidden, first)?,
            compare(
                &format!("prefill {index} logits"),
                &prefill.logits,
                &golden.logits
            )?
        );
        decode(engine, &prompt, golden, index)?;
        let written = match golden.reply.split_last() {
            Some((last, written)) if ENDS.contains(last) => written,
            _ => &golden.reply,
        };
        history.push(Replied {
            turn,
            reply: engine.text(written)?,
        });
    }
    Ok(())
}

/// The reply's tokens read after the reference's, but its last: each one's state against the
/// reference's, and how many of the likeliest tokens after them are the reference's.
fn decode(engine: &mut Engine, prompt: &Prompt, golden: &Answered, index: usize) -> Result<()> {
    let reply = &golden.reply;
    let forced = engine.force(prompt, &reply[..reply.len() - 1]);
    ensure!(!forced.is_empty(), "turn {index}: a reply of one token");
    let hidden: Vec<f32> = forced
        .iter()
        .flat_map(|prediction| prediction.hidden.iter().copied())
        .collect();
    let agreed = forced
        .iter()
        .zip(&reply[1..])
        .filter(|(prediction, token)| argmax(&prediction.logits) == **token as usize)
        .count();
    let share = agreed as f64 / forced.len() as f64;
    ensure!(
        share >= AGREEMENT,
        "decode {index}: {agreed} of {} next tokens the reference's",
        forced.len()
    );
    println!(
        "{} ({agreed} of {} next tokens the reference's)",
        compare(
            &format!("decode {index} hidden"),
            &hidden,
            &golden.hidden[HIDDEN..]
        )?,
        forced.len()
    );
    Ok(())
}
