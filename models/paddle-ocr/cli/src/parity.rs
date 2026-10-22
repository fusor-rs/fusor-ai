//! `paddle-ocr parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own input (its patches, its encoder output, its image
//! tokens), so errors do not compound from one stage to the next. Tensors are compared by
//! relative L2 (`‖ours − ref‖ / ‖ref‖`); the prompt's ids and rotary positions, and the greedy
//! answer's tokens, must be the reference's, one for one.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use paddle_ocr::pipeline::Stage;
use paddle_ocr::prompt::{Prompt, Task};
use paddle_ocr::session::Engine;
use paddle_ocr::{Grid, Patches, Pixels, Size};

/// The reference's tensors for one image (`reference/dump.py` documents them).
struct Goldens {
    image: Pixels,
    patches: Vec<f32>,
    grid: Grid,
    embedded: Vec<f32>,
    vision: Vec<f32>,
    projected: Vec<f32>,
    ids: Vec<u32>,
    positions: Vec<i64>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
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
        let ids = |name: &str| -> Result<Vec<u32>> {
            let values = file.i64(name)?;
            Ok(values
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?)
        };
        Ok(Self {
            image,
            patches: file.f32("patches")?,
            grid: Grid {
                rows: usize::try_from(grid[1])?,
                columns: usize::try_from(grid[2])?,
            },
            embedded: file.f32("embedded")?,
            vision: file.f32("vision")?,
            projected: file.f32("projected")?,
            ids: ids("ids")?,
            positions: file.i64("positions")?,
            hidden: file.f32("hidden")?,
            logits: file.f32("logits")?,
            tokens: ids("tokens")?,
        })
    }
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
        .find(|prompt| prompt.ids == goldens.ids)
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
                &goldens.vision,
            )?,
            Stage::Project => compare(
                "project",
                &engine.project(&goldens.vision, goldens.grid),
                &goldens.projected,
            )?,
            Stage::Prefill => prefill(engine, &prompt, &goldens)?,
            Stage::Decode => decode(engine, &prompt, &goldens)?,
        };
        println!("{report}");
    }
    Ok(())
}

fn preprocess(goldens: &Goldens) -> Result<String> {
    let patches = Patches::of(&goldens.image)?;
    ensure!(
        patches.grid == goldens.grid,
        "preprocess: grid {:?}, reference {:?}",
        patches.grid,
        goldens.grid
    );
    Ok(compare("preprocess", &patches.values, &goldens.patches)?)
}

/// The prompt's ids and positions, then its last row's hidden state and logits.
fn prefill(engine: &mut Engine, prompt: &Prompt, goldens: &Goldens) -> Result<String> {
    let len = prompt.ids.len();
    let positions = prompt.positions();
    let agree = positions.iter().enumerate().all(|(index, position)| {
        let axes = [position.temporal, position.height, position.width];
        let golden = |axis: usize| goldens.positions[axis * len + index];
        axes.iter()
            .enumerate()
            .all(|(axis, &value)| golden(axis) == value as i64)
    });
    ensure!(
        agree,
        "prompt: rotary positions differ from the reference's"
    );
    let prediction = engine.prefill(prompt, &goldens.projected);
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
        "prompt: {len} ids and positions identical\n{}\n{}",
        compare("prefill hidden", &prediction.hidden, &goldens.hidden)?,
        compare("prefill logits", &prediction.logits, &goldens.logits)?
    ))
}

/// The greedy answer on the reference's image tokens.
fn decode(engine: &mut Engine, prompt: &Prompt, goldens: &Goldens) -> Result<String> {
    let ours = engine.answer(prompt, &goldens.projected, goldens.tokens.len() + 1);
    let agreed = ours
        .iter()
        .zip(&goldens.tokens)
        .take_while(|(ours, reference)| ours == reference)
        .count();
    ensure!(
        agreed == ours.len() && ours.len() == goldens.tokens.len(),
        "decode: {} tokens, reference {}; they part at {agreed}: ours {:?}, reference {:?}",
        ours.len(),
        goldens.tokens.len(),
        ours.get(agreed),
        goldens.tokens.get(agreed)
    );
    Ok(format!(
        "decode: {} tokens identical: {:?}",
        ours.len(),
        engine.text(&ours)?
    ))
}
