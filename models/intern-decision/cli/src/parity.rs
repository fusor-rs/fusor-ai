//! `intern-decision parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own input (its patches, its encoder output, its image
//! tokens, its logits), so errors do not compound from one stage to the next. Tensors are
//! compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); the prompt's ids must be the reference's,
//! one for one, and our logits must decide every question as the reference's do.

use std::path::Path;

use anyhow::{Result, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use intern_decision::decision;
use intern_decision::pipeline::Stage;
use intern_decision::prompt::Prompt;
use intern_decision::request::Question;
use intern_decision::session::Engine;
use intern_decision::{Grid, Patches, Pixels, Size};
use serde_json::Value;

use crate::Case;

/// The reference's tensors of a case's image (`reference/dump.py` documents them).
struct Image {
    pixels: Pixels,
    patches: Vec<f32>,
    grid: Grid,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    merged: Vec<f32>,
}

/// The reference's tensors for one case.
struct Goldens {
    case: Case,
    image: Option<Image>,
    ids: Vec<i64>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let case = serde_json::from_slice(&file.u8("request")?)?;
        let image = if file.contains("image") {
            Some(image(&file)?)
        } else {
            None
        };
        Ok(Self {
            case,
            image,
            ids: file.i64("ids")?,
            hidden: file.f32("hidden")?,
            logits: file.f32("logits")?,
            probabilities: file.f32("probabilities")?,
        })
    }
}

fn image(file: &Checkpoint) -> Result<Image> {
    let shape = file.shape("image")?;
    let grid = file.i64("grid")?;
    Ok(Image {
        pixels: Pixels {
            size: Size {
                width: shape[1],
                height: shape[0],
            },
            channels: shape[2],
            values: file.u8("image")?,
        },
        patches: file.f32("patches")?,
        grid: Grid {
            rows: usize::try_from(grid[1])?,
            columns: usize::try_from(grid[2])?,
        },
        embedded: file.f32("embedded")?,
        encoded: file.f32("encoded")?,
        merged: file.f32("merged")?,
    })
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let grids: Vec<Grid> = goldens.image.iter().map(|image| image.grid).collect();
    let (questions, prompt) = engine.prompt(&goldens.case.request, &grids)?;
    let ids = prompt.ids.iter().map(|&id| i64::from(id));
    ensure!(
        ids.eq(goldens.ids.iter().copied()),
        "prompt: the ids are not the reference's"
    );
    println!(
        "{} questions, prompt: {} ids identical",
        questions.len(),
        prompt.ids.len()
    );
    for &stage in Stage::ALL {
        let report = match (stage, &goldens.image) {
            (Stage::Preprocess, Some(image)) => preprocess(image)?,
            (Stage::Embed, Some(image)) => {
                let patches = Patches {
                    grid: image.grid,
                    values: image.patches.clone(),
                };
                compare("embed", &engine.embed(&patches), &image.embedded)?
            }
            (Stage::Encode, Some(image)) => compare(
                "encode",
                &engine.encode(&image.embedded, image.grid),
                &image.encoded,
            )?,
            (Stage::Merge, Some(image)) => compare(
                "merge",
                &engine.merge(&image.encoded, image.grid),
                &image.merged,
            )?,
            (Stage::Read, _) => read(engine, &prompt, &questions, &goldens)?,
            (Stage::Decide, _) => decide(&questions, &goldens)?,
            (_, None) => continue,
        };
        println!("{report}");
    }
    Ok(())
}

fn preprocess(image: &Image) -> Result<String> {
    let patches = intern_decision::patches(&image.pixels)?;
    ensure!(
        patches.grid == image.grid,
        "preprocess: grid {:?}, reference {:?}",
        patches.grid,
        image.grid
    );
    Ok(compare("preprocess", &patches.values, &image.patches)?)
}

/// The deciding rows' states on the reference's image tokens, and the options' logits from
/// them, which must decide as the reference's.
fn read(
    engine: &mut Engine,
    prompt: &Prompt,
    questions: &[Question],
    goldens: &Goldens,
) -> Result<String> {
    let tokens = goldens
        .image
        .as_ref()
        .map_or(&[][..], |image| &image.merged);
    let hidden = engine.read(prompt, tokens);
    let logits = engine.logits(questions, &hidden);
    let ours = decisions(&decision::respond(questions, &logits, prompt.ids.len()));
    let reference = decisions(&respond(questions, &goldens.logits, prompt.ids.len()));
    ensure!(
        ours == reference,
        "read: decisions {ours:?}, reference {reference:?}"
    );
    Ok(format!(
        "{}\n{}\nread: decisions identical: {ours:?}",
        compare("read hidden", &hidden, &goldens.hidden)?,
        compare("read logits", &logits.concat(), &goldens.logits)?
    ))
}

/// The calibrated probabilities from the reference's logits.
fn decide(questions: &[Question], goldens: &Goldens) -> Result<String> {
    let response = respond(questions, &goldens.logits, 0);
    let probabilities: Vec<f32> = response
        .answers
        .values()
        .flat_map(|answer| answer["probabilities"].as_object().into_iter().flatten())
        .filter_map(|(_, probability)| probability.as_f64())
        .map(|probability| probability as f32)
        .collect();
    Ok(compare("decide", &probabilities, &goldens.probabilities)?)
}

/// The response from `logits`, the questions' concatenated.
fn respond(questions: &[Question], logits: &[f32], rows: usize) -> decision::Response {
    let mut rest = logits;
    let split = questions.iter().map(|question| {
        let (own, after) = rest.split_at(question.candidates.len());
        rest = after;
        own.to_vec()
    });
    decision::respond(questions, &split.collect::<Vec<_>>(), rows)
}

fn decisions(response: &decision::Response) -> Vec<Value> {
    let answers = response.answers.values();
    answers.map(|answer| answer["decision"].clone()).collect()
}
