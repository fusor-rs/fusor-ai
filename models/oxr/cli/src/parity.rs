//! `oxr parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own input (its patches, its encoder output, its image
//! tokens), so errors do not compound from one stage to the next. Tensors are compared by
//! relative L2 (`‖ours − ref‖ / ‖ref‖`); the prompt's text ids, and the greedy answer's tokens,
//! must be the reference's, one for one.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use oxr::pipeline::Stage;
use oxr::prompt::{Prompt, Task};
use oxr::session::Engine;
use oxr::{Grid, Patches, Pixels, Size};

/// The reference's tensors for one image (`reference/dump.py` documents them).
struct Goldens {
    image: Pixels,
    patches: Vec<f32>,
    grid: Grid,
    embedded: Vec<f32>,
    encoded: Vec<f32>,
    visual: Vec<f32>,
    ids: Vec<i64>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    answer: Vec<u32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let shape = file.shape("image")?;
        let image = Pixels {
            size: Size {
                width: shape[1],
                height: shape[0],
            },
            channels: shape[2],
            values: file.u8("image")?,
        };
        let grid = file.i64("grid")?;
        let answer = file.i64("answer")?;
        Ok(Self {
            image,
            patches: file.f32("patches")?,
            grid: Grid {
                rows: usize::try_from(grid[1])?,
                columns: usize::try_from(grid[2])?,
            },
            embedded: file.f32("embedded")?,
            encoded: file.f32("encoded")?,
            visual: file.f32("visual")?,
            ids: file.i64("ids")?,
            hidden: file.f32("hidden")?,
            logits: file.f32("logits")?,
            answer: answer
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?,
        })
    }
}

/// Whether the prompt's ids are the reference's: its text ids one for one, and the image's
/// rows, its indicators included, where the reference holds its (negative) image placeholders.
fn same_ids(prompt: &Prompt, reference: &[i64]) -> bool {
    let image = prompt.image.start - 1..prompt.image.end + 1;
    prompt.ids.len() == reference.len()
        && prompt
            .ids
            .iter()
            .zip(reference)
            .enumerate()
            .all(|(row, (&id, &want))| {
                if image.contains(&row) {
                    want < 0
                } else {
                    i64::from(id) == want
                }
            })
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let Grid { rows, columns } = goldens.grid;
    println!(
        "{}×{} image, {rows}×{columns} patches",
        goldens.image.size.width, goldens.image.size.height
    );
    let prompts = Task::ALL.map(|task| engine.prompt(task, goldens.grid));
    let prompt = prompts
        .into_iter()
        .filter_map(Result::ok)
        .find(|prompt| same_ids(prompt, &goldens.ids))
        .ok_or_else(|| anyhow!("prompt: no task's ids are the reference's"))?;
    for &stage in Stage::ALL {
        let report = match stage {
            Stage::Preprocess => preprocess(&goldens)?,
            Stage::Embed => {
                let patches = Patches {
                    grid: goldens.grid,
                    values: goldens.patches.clone(),
                };
                compare("embed", &engine.embed(&patches), &goldens.embedded)?
            }
            Stage::Encode => compare(
                "encode",
                &engine.encode(&goldens.embedded, goldens.grid),
                &goldens.encoded,
            )?,
            Stage::Tokenize => compare(
                "tokenize",
                &engine.tokens(&goldens.encoded, goldens.grid),
                &goldens.visual,
            )?,
            Stage::Prefill => prefill(engine, &prompt, &goldens)?,
            Stage::Decode => decode(engine, &prompt, &goldens)?,
        };
        println!("{report}");
    }
    Ok(())
}

fn preprocess(goldens: &Goldens) -> Result<String> {
    let patches = oxr::patches(&goldens.image)?;
    ensure!(
        patches.grid == goldens.grid,
        "preprocess: grid {:?}, reference {:?}",
        patches.grid,
        goldens.grid
    );
    Ok(compare("preprocess", &patches.values, &goldens.patches)?)
}

/// The last prompt row's hidden state and logits.
fn prefill(engine: &mut Engine, prompt: &Prompt, goldens: &Goldens) -> Result<String> {
    let prediction = engine.prefill(prompt, &goldens.visual);
    let best = |logits: &[f32]| {
        let best = logits.iter().enumerate();
        best.fold((0, f32::NEG_INFINITY), |best, (token, &logit)| {
            if logit > best.1 { (token, logit) } else { best }
        })
        .0
    };
    ensure!(
        best(&prediction.logits) == best(&goldens.logits),
        "prefill: the most likely token differs from the reference's"
    );
    Ok(format!(
        "prompt: {} ids identical\n{}\n{}",
        prompt.ids.len(),
        compare("prefill hidden", &prediction.hidden, &goldens.hidden)?,
        compare("prefill logits", &prediction.logits, &goldens.logits)?
    ))
}

/// The greedy answer on the reference's image tokens.
fn decode(engine: &mut Engine, prompt: &Prompt, goldens: &Goldens) -> Result<String> {
    let ours = engine.answer(prompt, &goldens.visual, goldens.answer.len() + 1);
    let agreed = ours
        .iter()
        .zip(&goldens.answer)
        .take_while(|(ours, reference)| ours == reference)
        .count();
    ensure!(
        agreed == ours.len() && ours.len() == goldens.answer.len(),
        "decode: {} tokens, reference {}; they part at {agreed}: ours {:?}, reference {:?}",
        ours.len(),
        goldens.answer.len(),
        ours.get(agreed),
        goldens.answer.get(agreed)
    );
    Ok(format!(
        "decode: {} tokens identical: {:?}",
        ours.len(),
        engine.text(&ours)?
    ))
}
