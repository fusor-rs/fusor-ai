//! `lfm3 parity`: each image's and each clip's stages and each turn of a conversation,
//! teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its images, patches, embeddings, tower output and
//! image tokens; its clips, features, encoder output and clip tokens), and each turn reads the
//! reference's prompt after the reference's earlier replies, so errors do not compound from
//! stage to stage or turn to turn. Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`);
//! each turn's prompt ids must be the reference's, one for one. A text reply's first token must
//! be the reference's, and most of its next tokens, read after the reference's, the reference's
//! too. A spoken reply's first frame is compared by the depth decoder's logits over it, and its
//! frames by the codec's samples of them.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use compile::Checkpoint;
use harness::parity::compare;
use kernels::numpy::argmax;
use lfm3::audio::{CODEBOOKS, Frame};
use lfm3::media::{self, IMAGE};
use lfm3::model::HIDDEN;
use lfm3::prompt::{Opening, Replied, Turn, written};
use lfm3::session::Engine;
use lfm3::{Grid, Patches, Pixels, Size};

use crate::Conversation;

/// The share of a reply's teacher-forced tokens whose likeliest next token must be the
/// reference's: the quantized weights move near ties.
const AGREEMENT: f64 = 0.9;

/// One turn's tensors (`reference/dump.py` documents them).
struct Answered {
    ids: Vec<i64>,
    reply: Vec<u32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    spoken: Option<Spoken>,
}

/// A spoken reply's tensors.
struct Spoken {
    depth: Vec<f32>,
    frames: Vec<Frame>,
    speech: Vec<f32>,
}

/// One clip's tensors.
struct Clip {
    samples: Vec<f32>,
    features: Vec<f32>,
    encoded: Vec<f32>,
    tokens: Vec<f32>,
}

/// One image's tensors.
struct Item {
    image: Pixels,
    patches: Patches,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    tokens: Vec<f32>,
}

/// The reference's tensors for one conversation.
struct Goldens {
    conversation: Conversation,
    items: Vec<Item>,
    clips: Vec<Clip>,
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
        let items = (0..count("item", ".grid")).map(|index| item(&file, index));
        let clips = (0..count("clip", ".tokens")).map(|index| {
            let name = |part: &str| format!("clip.{index}.{part}");
            Ok(Clip {
                samples: file.f32(&format!("clip.{index}"))?,
                features: file.f32(&name("features"))?,
                encoded: file.f32(&name("encoded"))?,
                tokens: file.f32(&name("tokens"))?,
            })
        });
        let turns = (0..count("turn", ".ids")).map(|index| answered(&file, index));
        Ok(Self {
            conversation: serde_json::from_slice(&file.u8("conversation")?)?,
            items: items.collect::<Result<_>>()?,
            clips: clips.collect::<Result<_>>()?,
            turns: turns.collect::<Result<_>>()?,
        })
    }
}

fn answered(file: &Checkpoint, index: usize) -> Result<Answered> {
    let name = |part: &str| format!("turn.{index}.{part}");
    let reply = file.i64(&name("reply"))?.into_iter().map(u32::try_from);
    let spoken = if file.contains(&name("depth")) {
        let codes = file.i64(&name("codes"))?.into_iter().map(u32::try_from);
        let codes = codes.collect::<Result<Vec<_>, _>>()?;
        let frames = codes.as_chunks::<CODEBOOKS>().0.to_vec();
        Some(Spoken {
            depth: file.f32(&name("depth"))?,
            frames,
            speech: file.f32(&name("speech"))?,
        })
    } else {
        None
    };
    Ok(Answered {
        ids: file.i64(&name("ids"))?,
        reply: reply.collect::<Result<_, _>>()?,
        hidden: file.f32(&name("hidden"))?,
        logits: file.f32(&name("logits"))?,
        spoken,
    })
}

fn item(file: &Checkpoint, index: usize) -> Result<Item> {
    let name = |part: &str| format!("item.{index}.{part}");
    let image = format!("image.{index}");
    let &[height, width, channels] = file.shape(&image)? else {
        bail!("{image}: not of height, width and channels");
    };
    let grid = file.i64(&name("grid"))?;
    Ok(Item {
        image: Pixels {
            size: Size { width, height },
            channels,
            values: file.u8(&image)?,
        },
        patches: Patches {
            grid: Grid {
                rows: usize::try_from(grid[0])?,
                columns: usize::try_from(grid[1])?,
            },
            values: file.f32(&name("patches"))?,
        },
        embedded: file.f32(&name("embedded"))?,
        encoded: file.f32(&name("encoded"))?,
        tokens: file.f32(&name("tokens"))?,
    })
}

/// Each image's stages on the reference's input.
fn see(engine: &mut Engine, items: &[Item]) -> Result<()> {
    for (index, item) in items.iter().enumerate() {
        let grid = item.patches.grid;
        let patches = media::patches(&item.image)?;
        ensure!(
            patches.grid == grid,
            "image {index}: a {:?} grid, the reference's {grid:?}",
            patches.grid
        );
        let embedded = engine.embed(&item.patches)?;
        let encoded = engine.encode(&item.embedded, grid)?;
        let tokens = engine.project(&item.encoded, grid)?;
        let name = |stage: &str| format!("image {index} {stage}");
        println!(
            "{}\n{}\n{}\n{}",
            compare(&name("patches"), &patches.values, &item.patches.values)?,
            compare(&name("embedded"), &embedded, &item.embedded)?,
            compare(&name("encoded"), &encoded, &item.encoded)?,
            compare(&name("tokens"), &tokens, &item.tokens)?
        );
    }
    Ok(())
}

/// Each clip's stages on the reference's input.
fn hear(engine: &mut Engine, clips: &[Clip]) -> Result<()> {
    for (index, clip) in clips.iter().enumerate() {
        let features = engine.audio_features(&clip.samples)?;
        let encoded = engine.encode_audio(&clip.features)?;
        let tokens = engine.project_audio(&clip.encoded)?;
        let name = |stage: &str| format!("clip {index} {stage}");
        println!(
            "{}\n{}\n{}",
            compare(&name("features"), &features, &clip.features)?,
            compare(&name("encoded"), &encoded, &clip.encoded)?,
            compare(&name("tokens"), &tokens, &clip.tokens)?
        );
    }
    Ok(())
}

/// The next `count` of `goldens`, as many as a turn has.
fn take<'goldens, T>(
    goldens: &mut impl Iterator<Item = &'goldens T>,
    count: usize,
    index: usize,
) -> Result<Vec<&'goldens T>> {
    let taken: Vec<&T> = goldens.take(count).collect();
    ensure!(
        taken.len() == count,
        "turn {index}: fewer images or clips in the goldens than the turn's"
    );
    Ok(taken)
}

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Goldens::read(path)?;
    println!("# {}", path.display());
    see(engine, &goldens.items)?;
    hear(engine, &goldens.clips)?;
    let mut history = Vec::new();
    let (mut items, mut clips) = (goldens.items.iter(), goldens.clips.iter());
    let mut media = Vec::new();
    let turns = goldens.conversation.turns.iter().zip(&goldens.turns);
    for (index, (said, golden)) in turns.enumerate() {
        let message = &said.message;
        let images = take(&mut items, said.images.len(), index)?;
        let heard = take(&mut clips, said.audio.len(), index)?;
        for tokens in images.iter().map(|image| &image.tokens) {
            media.extend_from_slice(tokens);
        }
        for tokens in heard.iter().map(|clip| &clip.tokens) {
            media.extend_from_slice(tokens);
        }
        let turn = Turn {
            images: images
                .iter()
                .map(|image| IMAGE.tokens(image.patches.grid.patches()))
                .collect(),
            audio: heard
                .iter()
                .map(|clip| clip.tokens.len() / HIDDEN)
                .collect(),
            text: message.text.clone(),
        };
        let opening = Opening {
            mode: message.mode,
            thinking: message.thinking,
        };
        let prompt = engine.prompt(&message.system, &history, &turn, opening)?;
        let ids = prompt.iter().map(|&id| i64::from(id));
        ensure!(
            ids.eq(golden.ids.iter().copied()),
            "turn {index}: the prompt's ids are not the reference's"
        );
        let prefill = engine.prefill(&prompt, &media);
        println!(
            "prefill {index}: {} ids identical\n{}",
            prompt.len(),
            compare(
                &format!("prefill {index} hidden"),
                &prefill.hidden,
                &golden.hidden[..HIDDEN]
            )?
        );
        match &golden.spoken {
            Some(spoken) => speak(engine, &prefill.hidden, spoken, index)?,
            None => write(engine, (&prompt, &prefill.logits), golden, index)?,
        }
        history.push(Replied {
            turn,
            reply: engine.text(&written(&golden.reply))?,
        });
    }
    Ok(())
}

/// A text reply: its first token and the prompt's logits, then its next tokens.
fn write(
    engine: &mut Engine,
    (prompt, logits): (&[u32], &[f32]),
    golden: &Answered,
    index: usize,
) -> Result<()> {
    ensure!(
        argmax(logits) == golden.reply[0] as usize,
        "turn {index}: the reply's first token is not the reference's"
    );
    println!(
        "{}",
        compare(&format!("prefill {index} logits"), logits, &golden.logits)?
    );
    decode(engine, prompt, golden, index)
}

/// A spoken reply: the depth decoder's first pass from the prompt's last state, and the codec's
/// samples of the reference's frames.
fn speak(engine: &mut Engine, hidden: &[f32], spoken: &Spoken, index: usize) -> Result<()> {
    let depth = engine.depth_logits(hidden)?;
    let speech = engine.speak(&spoken.frames)?;
    println!(
        "{}\n{}",
        compare(&format!("speak {index} depth"), &depth, &spoken.depth)?,
        compare(&format!("speak {index} samples"), &speech, &spoken.speech)?
    );
    Ok(())
}

/// The reply's tokens read after the reference's, but its last: each one's state against the
/// reference's, and how many of the likeliest tokens after them are the reference's.
fn decode(engine: &mut Engine, prompt: &[u32], golden: &Answered, index: usize) -> Result<()> {
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
