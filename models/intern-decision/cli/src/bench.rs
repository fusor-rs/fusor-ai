//! `intern-decision bench`: the milliseconds of each stage on one request, and of deciding it
//! whole.

use std::time::Instant;

use anyhow::Result;
use bundle::Engine as _;
use bundle::Stages as _;
use harness::bench::{row, time};
use intern_decision::decision;
use intern_decision::pipeline::Stage;
use intern_decision::prompt::Prompt;
use intern_decision::request::{Question, Request};
use intern_decision::session::Engine;
use intern_decision::{Grid, MAX_ASPECT, PREPROCESSING, Patches, Pixels};

const CALLS: usize = 3;

/// Each stage's input: the images' patches, then what each stage made of them.
struct Inputs {
    patches: Vec<Patches>,
    embedded: Vec<Vec<f32>>,
    encoded: Vec<Vec<f32>>,
    tokens: Vec<f32>,
    questions: Vec<Question>,
    prompt: Prompt,
    logits: Vec<Vec<f32>>,
}

pub fn run(engine: &mut Engine, request: &Request, images: &[Pixels]) -> Result<()> {
    let inputs = inputs(engine, request, images)?;
    println!(
        "threads {}, {} images, {} prompt tokens, {} questions",
        engine.threads(),
        images.len(),
        inputs.prompt.ids.len(),
        inputs.questions.len()
    );
    for &stage in Stage::ALL {
        row(stage, milliseconds(engine, stage, images, &inputs)?, "");
    }
    let start = Instant::now();
    engine.start(request, images)?;
    while engine.step()?.response.is_none() {}
    println!(
        "  decide whole {:9.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn inputs(engine: &mut Engine, request: &Request, images: &[Pixels]) -> Result<Inputs> {
    let patches = images
        .iter()
        .map(|image| PREPROCESSING.centered_patches(image, MAX_ASPECT))
        .collect::<Result<Vec<_>, _>>()?;
    let grids: Vec<Grid> = patches.iter().map(|patches| patches.grid).collect();
    let (questions, prompt) = engine.prompt(request, &grids)?;
    let embedded: Vec<Vec<f32>> = patches
        .iter()
        .map(|patches| engine.embed(patches))
        .collect();
    let encoded: Vec<Vec<f32>> = (patches.iter().zip(&embedded))
        .map(|(patches, embedded)| engine.encode(embedded, patches.grid))
        .collect();
    let tokens: Vec<f32> = (patches.iter().zip(&encoded))
        .flat_map(|(patches, encoded)| engine.merge(encoded, patches.grid))
        .collect();
    let hidden = engine.read(&prompt, &tokens);
    let logits = engine.logits(&questions, &hidden);
    Ok(Inputs {
        patches,
        embedded,
        encoded,
        tokens,
        questions,
        prompt,
        logits,
    })
}

/// The milliseconds per call of `stage`, over every image or over the prompt.
fn milliseconds(
    engine: &mut Engine,
    stage: Stage,
    images: &[Pixels],
    inputs: &Inputs,
) -> Result<f64> {
    let patches = &inputs.patches;
    let ms = match stage {
        Stage::Preprocess => time(CALLS, || {
            images
                .iter()
                .map(|image| PREPROCESSING.centered_patches(image, MAX_ASPECT))
                .collect::<Result<Vec<_>, _>>()
        })?,
        Stage::Embed => time(CALLS, || {
            for patches in patches {
                engine.embed(patches);
            }
            anyhow::Ok(())
        })?,
        Stage::Encode => time(CALLS, || {
            for (patches, embedded) in patches.iter().zip(&inputs.embedded) {
                engine.encode(embedded, patches.grid);
            }
            anyhow::Ok(())
        })?,
        Stage::Merge => time(CALLS, || {
            for (patches, encoded) in patches.iter().zip(&inputs.encoded) {
                engine.merge(encoded, patches.grid);
            }
            anyhow::Ok(())
        })?,
        Stage::Read => time(CALLS, || {
            anyhow::Ok(engine.read(&inputs.prompt, &inputs.tokens))
        })?,
        Stage::Decide => time(CALLS, || {
            let rows = inputs.prompt.ids.len();
            anyhow::Ok(decision::respond(&inputs.questions, &inputs.logits, rows))
        })?,
    };
    Ok(ms)
}
