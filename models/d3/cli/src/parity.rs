//! `d3 parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its images and video frames, its patches, its
//! encoder output, its media tokens), so errors do not compound from one stage to the next.
//! Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); each question's prompt ids must
//! be the reference's, one for one, and its likeliest option the reference's.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use d3::media::{Media, Video};
use d3::pipeline::Stage;
use d3::request::Request;
use d3::session::Engine;
use d3::{Grid, Patches, Pixels};
use harness::parity::compare;
use kernels::numpy::argmax;

/// One image's or video group's tensors (`reference/dump.py` documents them).
struct Item {
    patches: Patches,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    merged: Vec<f32>,
}

/// One answered question's tensors.
struct Answered {
    ids: Vec<i64>,
    hidden: Vec<f32>,
    probabilities: Vec<f32>,
}

/// The reference's tensors for one case.
struct Goldens {
    request: Request,
    images: Vec<Pixels>,
    videos: Vec<Video>,
    items: Vec<Item>,
    questions: Vec<Answered>,
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
        let questions = (0..count("question", ".ids")).map(|index| {
            let name = |part: &str| format!("question.{index}.{part}");
            Ok(Answered {
                ids: file.i64(&name("ids"))?,
                hidden: file.f32(&name("hidden"))?,
                probabilities: file.f32(&name("probabilities"))?,
            })
        });
        Ok(Self {
            request: serde_json::from_slice(&file.u8("request")?)?,
            images,
            videos,
            items: items.collect::<Result<_>>()?,
            questions: questions.collect::<Result<_>>()?,
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
        size: d3::Size { width, height },
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
    for &stage in Stage::ALL {
        match stage {
            Stage::Preprocess => preprocess(&goldens)?,
            Stage::Embed | Stage::Encode | Stage::Merge => vision(engine, stage, &goldens)?,
            Stage::Read | Stage::Answer => questions(engine, stage, &goldens)?,
        }
    }
    Ok(())
}

fn preprocess(goldens: &Goldens) -> Result<()> {
    let media = Media::new(&goldens.images, &goldens.videos)?;
    for (index, (ours, golden)) in media.items().zip(&goldens.items).enumerate() {
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

/// Each answered question's prompt, its last row's state on the reference's media tokens, and
/// its options' probabilities from the reference's state.
fn questions(engine: &mut Engine, stage: Stage, goldens: &Goldens) -> Result<()> {
    let media = Media::new(&goldens.images, &goldens.videos)?;
    let tokens: Vec<f32> = goldens
        .items
        .iter()
        .flat_map(|item| item.merged.iter().copied())
        .collect();
    let questions = goldens.request.questions()?;
    let answered = questions
        .iter()
        // A question that failed its check has no golden: the reference answers it with the error.
        .filter_map(|asked| asked.question.as_ref().ok());
    for (index, (question, golden)) in answered.zip(&goldens.questions).enumerate() {
        let name = format!("{} {index}", stage.name());
        if stage == Stage::Answer {
            let ours = engine.probabilities(&golden.hidden, question.keys.len());
            let same = argmax(&ours) == argmax(&golden.probabilities);
            ensure!(
                same,
                "{name}: likeliest option {}, reference {}",
                argmax(&ours),
                argmax(&golden.probabilities)
            );
            println!(
                "{}, likeliest identical",
                compare(&name, &ours, &golden.probabilities)?
            );
            continue;
        }
        let prompt = engine.prompt(&goldens.request.state, question, &media)?;
        let ids = prompt.ids.iter().map(|&id| i64::from(id));
        ensure!(
            ids.eq(golden.ids.iter().copied()),
            "{name}: the prompt's ids are not the reference's"
        );
        let hidden = engine.read(&prompt, &tokens);
        println!(
            "{} ({} ids identical)",
            compare(&name, &hidden, &golden.hidden)?,
            golden.ids.len()
        );
    }
    Ok(())
}
