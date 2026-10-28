//! `d1 parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its images and clips, its patches, its encoder
//! output, its media tokens, its final states), so errors do not compound from one stage to the
//! next. Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); each pack's ids must be
//! the reference's, one for one, and the answers read from the reference's final states the
//! reference's but for their numbers, whose largest difference is printed.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use d1::model::HIDDEN;
use d1::pipeline::Stage;
use d1::request::Request;
use d1::session::Engine;
use d1::{Grid, Patches, Pixels, Size};
use harness::parity::compare;
use serde_json::Value;

/// One image's tensors (`reference/dump.py` documents them).
struct Item {
    patches: Patches,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    tokens: Vec<f32>,
}

/// One clip's tensors.
struct Clip {
    samples: Vec<f32>,
    features: Vec<f32>,
    tokens: Vec<f32>,
}

/// One pack's tensors.
struct Pack {
    ids: Vec<i64>,
    hidden: Vec<f32>,
}

/// The reference's tensors for one case.
struct Goldens {
    request: Request,
    answers: Value,
    images: Vec<Pixels>,
    items: Vec<Item>,
    clips: Vec<Clip>,
    packs: Vec<Pack>,
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
            .map(|index| pixels(&file, &format!("image.{index}")))
            .collect::<Result<_>>()?;
        let items = (0..count("item", ".grid")).map(|index| item(&file, index));
        let clips = (0..count("clip", ".samples")).map(|index| {
            let name = |part: &str| format!("clip.{index}.{part}");
            Ok(Clip {
                samples: file.f32(&name("samples"))?,
                features: file.f32(&name("features"))?,
                tokens: file.f32(&name("tokens"))?,
            })
        });
        let packs = (0..count("pack", ".ids")).map(|index| {
            let name = |part: &str| format!("pack.{index}.{part}");
            Ok(Pack {
                ids: file.i64(&name("ids"))?,
                hidden: file.f32(&name("hidden"))?,
            })
        });
        Ok(Self {
            request: serde_json::from_slice(&file.u8("request")?)?,
            answers: serde_json::from_slice(&file.u8("answers")?)?,
            images,
            items: items.collect::<Result<_>>()?,
            clips: clips.collect::<Result<_>>()?,
            packs: packs.collect::<Result<_>>()?,
        })
    }
}

/// The RGB image under `name`, `[height, width, 3]`.
fn pixels(file: &Checkpoint, name: &str) -> Result<Pixels> {
    let &[height, width, channels] = file.shape(name)? else {
        bail!("{name}: not an image of height, width and channels");
    };
    Ok(Pixels {
        size: Size { width, height },
        channels,
        values: file.u8(name)?,
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
        tokens: file.f32(&name("tokens"))?,
    })
}

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Goldens::read(path)?;
    println!("# {} ({})", path.display(), engine.variant().name());
    for &stage in Stage::ALL {
        match stage {
            Stage::See => see(engine, &goldens)?,
            Stage::Hear => hear(engine, &goldens)?,
            Stage::Read => read(engine, &goldens)?,
        }
    }
    Ok(())
}

/// Each image's patches, then its embedding, encoding and tokens, stage by stage.
fn see(engine: &mut Engine, goldens: &Goldens) -> Result<()> {
    for (index, (image, item)) in goldens.images.iter().zip(&goldens.items).enumerate() {
        let grid = item.patches.grid;
        let ours = engine.patches(image)?;
        ensure!(
            ours.grid == grid,
            "patches {index}: grid {:?}, reference {grid:?}",
            ours.grid
        );
        let stages = [
            ("patches", ours.values, &item.patches.values),
            ("embed", engine.embed(&item.patches), &item.embedded),
            ("encode", engine.encode(&item.embedded, grid), &item.encoded),
            ("project", engine.project(&item.encoded, grid), &item.tokens),
        ];
        for (name, ours, golden) in stages {
            println!(
                "{}",
                compare(&format!("see {name} {index}"), &ours, golden)?
            );
        }
    }
    Ok(())
}

/// Each clip's features, then its tokens.
fn hear(engine: &mut Engine, goldens: &Goldens) -> Result<()> {
    for (index, clip) in goldens.clips.iter().enumerate() {
        let features = engine.features(&clip.samples)?;
        let name = format!("hear features {index}");
        println!("{}", compare(&name, &features, &clip.features)?);
        let tokens = engine.hear(&clip.features)?;
        println!(
            "{}",
            compare(&format!("hear {index}"), &tokens, &clip.tokens)?
        );
    }
    Ok(())
}

/// Each pack's ids, its rows' final states on the reference's media tokens, and its answers
/// from the reference's final states.
fn read(engine: &mut Engine, goldens: &Goldens) -> Result<()> {
    let image_counts: Vec<usize> = goldens
        .items
        .iter()
        .map(|item| item.tokens.len() / HIDDEN)
        .collect();
    let clip_counts: Vec<usize> = goldens
        .clips
        .iter()
        .map(|clip| clip.tokens.len() / HIDDEN)
        .collect();
    let prepared = engine.prepare(&goldens.request, &image_counts, &clip_counts)?;
    ensure!(
        prepared.packs.len() == goldens.packs.len(),
        "{} packs, reference {}",
        prepared.packs.len(),
        goldens.packs.len()
    );
    let image_tokens: Vec<f32> = goldens
        .items
        .iter()
        .flat_map(|item| item.tokens.clone())
        .collect();
    let audio_tokens: Vec<f32> = goldens
        .clips
        .iter()
        .flat_map(|clip| clip.tokens.clone())
        .collect();
    let mut answers = serde_json::Map::new();
    for (index, (pack, golden)) in prepared.packs.iter().zip(&goldens.packs).enumerate() {
        let ids = pack.packed.ids.iter().map(|&id| i64::from(id));
        ensure!(
            ids.eq(golden.ids.iter().copied()),
            "read {index}: the pack's ids are not the reference's"
        );
        let normed = engine.read(&pack.packed, &image_tokens, &audio_tokens);
        let name = format!("read {index}");
        println!(
            "{} ({} ids identical)",
            compare(&name, &normed, &golden.hidden)?,
            golden.ids.len()
        );
        answers.extend(engine.answers(&prepared, index, &golden.hidden)?);
    }
    let mut largest = 0.0_f64;
    differences(
        &Value::Object(answers),
        &goldens.answers,
        "answers",
        &mut largest,
    )?;
    println!("answers identical but for numbers, max |Δ| {largest:.2e}");
    Ok(())
}

/// Whether `ours` is `reference` but for its numbers, the largest difference of which goes into
/// `largest`.
fn differences(ours: &Value, reference: &Value, at: &str, largest: &mut f64) -> Result<()> {
    match (ours, reference) {
        (Value::Number(ours), Value::Number(reference)) => {
            let value = |number: &serde_json::Number| number.as_f64().expect("a JSON number");
            *largest = largest.max((value(ours) - value(reference)).abs());
        }
        (Value::Array(ours), Value::Array(reference)) if ours.len() == reference.len() => {
            for (index, (ours, reference)) in ours.iter().zip(reference).enumerate() {
                differences(ours, reference, &format!("{at}[{index}]"), largest)?;
            }
        }
        (Value::Object(ours), Value::Object(reference)) if ours.keys().eq(reference.keys()) => {
            for ((key, ours), reference) in ours.iter().zip(reference.values()) {
                differences(ours, reference, &format!("{at}.{key}"), largest)?;
            }
        }
        (ours, reference) => ensure!(ours == reference, "{at}: {ours}, reference {reference}"),
    }
    Ok(())
}
