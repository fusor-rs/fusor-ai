//! `iris parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/dump.py`, one `golden_f32_<case>.safetensors` per bundle).
//!
//! Each stage gets the reference's own input (its tokens, its text states, its noise, its tile
//! fusion), so errors do not compound from one stage to the next, but for the samplers, which run
//! every step from the reference's starting noise. Tensors are compared by relative L2
//! (`‖ours − ref‖ / ‖ref‖`); the tokens and the budgeted photo must be the reference's, byte for
//! byte; the colours and the restored photo may differ in a few pixels. Each engine is dropped
//! before the next bundle loads: the bundles do not fit memory together.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use compile::Checkpoint;
use harness::parity::{compare, pixel_major};
use iris::depth;
use iris::model::{
    DEFAULT_SHIFT, DEPTH_TIME, IMAGE_CHANNELS, TEXT_BUNDLE, TRAINING_TIMESTEPS, TURBO_LEVELS,
    Variant,
};
use iris::picture::colorize;
use iris::restore;
use iris::session::{Branch, Generation, ImageEngine, TextEngine};
use iris::{Pixels, Size};

use crate::Bundle;

/// The dump's photo budgets and depth side (`reference/dump.py`'s `SHORT`, `LONG`, `MAX_SIDE`).
const SHORT: usize = 192;
const LONG: usize = 512;
const MAX_SIDE: usize = 1024;
/// The share of the colours or bytes that may differ from the reference's, per mille.
const DIFFERENT_PER_MILLE: usize = 1;

/// The golden file of the case `name`.
fn goldens_of(goldens: &Path, name: &str) -> Result<Checkpoint> {
    Ok(Checkpoint::read(
        &goldens.join(format!("golden_f32_{name}.safetensors")),
    )?)
}

pub fn run(bundle: &Path, goldens: &Path, only: Option<Bundle>) -> Result<()> {
    for case in Bundle::ALL {
        if only.is_none_or(|only| only == case) {
            println!("{}:", case.name());
            match case {
                Bundle::Text => text(bundle, goldens)?,
                Bundle::Transformer(variant) => image(bundle, goldens, variant)?,
            }
        }
    }
    Ok(())
}

fn text(bundle: &Path, goldens: &Path) -> Result<()> {
    let file = goldens_of(goldens, "text")?;
    let mut engine: TextEngine = harness::load(&bundle.join(TEXT_BUNDLE))?;
    let prompt = String::from_utf8(file.u8("prompt")?)?;
    let ids = |name: &str| -> Result<Vec<u32>> {
        Ok(file
            .i64(name)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<_, _>>()?)
    };
    let (prompt_ids, negative_ids) = (ids("prompt_tokens")?, ids("negative_tokens")?);
    for (text, reference) in [(prompt.as_str(), &prompt_ids), ("", &negative_ids)] {
        let ours = engine.tokens(text)?;
        ensure!(
            &ours == reference,
            "prompt {text:?}: tokens {ours:?}, reference {reference:?}"
        );
    }
    println!("prompt: both prompts' tokens identical");
    let ours = [
        engine.encode_tokens(&prompt_ids)?,
        engine.encode_tokens(&negative_ids)?,
    ]
    .concat();
    let reference = [file.f32("prompt_states")?, file.f32("negative_states")?].concat();
    println!("{}", compare("text", &ours, &reference)?);
    Ok(())
}

fn image(bundle: &Path, goldens: &Path, variant: Variant) -> Result<()> {
    let file = goldens_of(goldens, variant.name())?;
    let mut engine: ImageEngine = harness::load(&bundle.join(variant.name()))?;
    match variant {
        Variant::Base | Variant::Turbo => generation(&mut engine, goldens, &file),
        Variant::Depth => depth_map(&mut engine, &file),
        Variant::Upscaler => restoration(&mut engine, &file),
    }
}

/// The size of a channel-major `[channels, height, width]` tensor.
fn plane(file: &Checkpoint, name: &str) -> Result<Size> {
    let &[_, height, width] = file.shape(name)? else {
        anyhow::bail!("{name}: [channels, height, width] expected");
    };
    Ok(Size { width, height })
}

/// The photo of a `[height, width, 3]` byte tensor.
fn photo(file: &Checkpoint, name: &str) -> Result<Pixels> {
    let &[height, width, channels] = file.shape(name)? else {
        anyhow::bail!("{name}: [height, width, channels] expected");
    };
    Ok(Pixels {
        size: Size { width, height },
        channels,
        values: file.u8(name)?,
    })
}

/// The states of the prompt (and, for base, of the negative prompt) the text case wrote, the
/// engine conditioned on them.
fn condition(engine: &mut ImageEngine, goldens: &Path) -> Result<()> {
    let text = goldens_of(goldens, "text")?;
    engine.condition(Branch::Prompt, &text.f32("prompt_states")?)?;
    if engine.variant() == Variant::Base {
        engine.condition(Branch::Negative, &text.f32("negative_states")?)?;
    }
    Ok(())
}

/// The base model's or Turbo's adapter, transformer and sampler, from the reference's noise.
fn generation(engine: &mut ImageEngine, goldens: &Path, file: &Checkpoint) -> Result<()> {
    condition(engine, goldens)?;
    let size = plane(file, "noise")?;
    let noise = pixel_major(
        &file.f32("noise")?,
        (IMAGE_CHANNELS, size.width * size.height),
    );
    if engine.variant() == Variant::Base {
        println!("{}", adapter(engine, file)?);
    }
    println!("{}", transformer(engine, file, (&noise, size))?);
    sampler(engine, file, (&noise, size))
}

/// The adapted text of the prompt, then of the negative prompt.
fn adapter(engine: &ImageEngine, file: &Checkpoint) -> Result<String> {
    let prompt = engine.adapted(Branch::Prompt)?;
    let negative = engine.adapted(Branch::Negative)?;
    Ok(compare(
        "adapter",
        &[prompt, negative].concat(),
        &file.f32("adapted")?,
    )?)
}

/// The transformer's velocity for the reference's noise at its first timestep under the prompt
/// (and, for base, the negative prompt).
fn transformer(
    engine: &mut ImageEngine,
    file: &Checkpoint,
    (noise, size): (&[f32], Size),
) -> Result<String> {
    let (timestep, branches) = match engine.variant() {
        Variant::Base => (
            file.f32("timesteps")?[0],
            vec![Branch::Prompt, Branch::Negative],
        ),
        _ => (
            TURBO_LEVELS[0] * TRAINING_TIMESTEPS as f32,
            vec![Branch::Prompt],
        ),
    };
    let mut velocity = Vec::new();
    for branch in branches {
        velocity.extend(engine.predict((noise, size), timestep, branch)?);
    }
    let reference = pixel_major(
        &file.f32("velocity")?,
        (IMAGE_CHANNELS, size.width * size.height),
    );
    Ok(compare("dit", &velocity, &reference)?)
}

/// The sampler's schedule (base) and every step from the reference's noise to its image.
fn sampler(
    engine: &mut ImageEngine,
    file: &Checkpoint,
    (noise, size): (&[f32], Size),
) -> Result<()> {
    let mut request = Generation {
        width: size.width,
        height: size.height,
        steps: 1,
        guidance: 1.0,
        shift: DEFAULT_SHIFT,
        seed: 0,
    };
    if engine.variant() == Variant::Base {
        let timesteps = file.f32("timesteps")?;
        request.steps = timesteps.len();
        request.guidance = file.f32("guidance")?[0];
        request.shift = file.f32("shift")?[0];
        let schedule = engine.schedule(request.steps, request.shift);
        let ours: Vec<f32> = (0..request.steps)
            .map(|step| schedule.timestep(step))
            .collect();
        println!("{}", compare("timesteps", &ours, &timesteps)?);
        println!(
            "{}",
            compare("levels", schedule.levels(), &file.f32("sigmas")?)?
        );
    }
    let image: Vec<f32> = engine
        .sample(noise, &request)?
        .into_iter()
        .map(|value| value.clamp(-1.0, 1.0))
        .collect();
    let reference = pixel_major(
        &file.f32("image")?,
        (IMAGE_CHANNELS, size.width * size.height),
    );
    println!("{}", compare("sampler", &image, &reference)?);
    Ok(())
}

/// How many of `ours` differ from `reference` by more than `tolerance`.
fn differing(ours: &[u8], reference: &[u8], tolerance: u8) -> usize {
    ours.iter()
        .zip(reference)
        .filter(|(left, right)| left.abs_diff(**right) > tolerance)
        .count()
}

/// The depth model: the resize, the prediction, the upsample and the colours.
fn depth_map(engine: &mut ImageEngine, file: &Checkpoint) -> Result<()> {
    let photo = photo(file, "input")?;
    let model = plane(file, "resized")?;
    ensure!(
        depth::model_size(photo.size, MAX_SIDE) == model,
        "the model sees {:?}, reference {model:?}",
        depth::model_size(photo.size, MAX_SIDE)
    );
    let pixels = model.width * model.height;
    let reference = pixel_major(&file.f32("resized")?, (IMAGE_CHANNELS, pixels));
    println!(
        "{}",
        compare("resize", &depth::input(&photo, model), &reference)?
    );
    let prediction = engine.predict((&reference, model), DEPTH_TIME, Branch::Prompt)?;
    println!(
        "{}",
        compare("depth", &prediction, &file.f32("prediction")?)?
    );
    let golden = file.f32("prediction")?;
    let upsampled = depth::upsample(&golden, model, photo.size);
    println!("{}", compare("upsample", &upsampled, &file.f32("depth")?)?);
    let ours = colorize(&file.f32("depth")?);
    let different = differing(&ours, &file.u8("colorized")?, 0) / IMAGE_CHANNELS;
    let total = ours.len() / IMAGE_CHANNELS;
    println!("colorize: {different} of {total} pixels differ");
    ensure!(
        different * 1000 < total * DIFFERENT_PER_MILLE,
        "more than {DIFFERENT_PER_MILLE} per mille of the colours differ"
    );
    Ok(())
}

/// The restorer: the budget, the enlargement, the work image, the tiles, the finish and the
/// colour fix.
fn restoration(engine: &mut ImageEngine, file: &Checkpoint) -> Result<()> {
    let photo = photo(file, "input")?;
    let budgeted = restore::budget(&photo, (SHORT, LONG));
    ensure!(
        budgeted.values == file.u8("budgeted")?,
        "the budgeted photo differs from the reference's"
    );
    println!("budget: identical");
    let output = restore::enlarged(budgeted.size);
    let enlarged = restore::upsample(&budgeted, output);
    let golden_reference = pixel_major(
        &file.f32("reference")?,
        (IMAGE_CHANNELS, output.width * output.height),
    );
    println!("{}", compare("bicubic", &enlarged, &golden_reference)?);
    let layout = restore::layout(output).context("the enlargement has a work image")?;
    let work = restore::work(&golden_reference, output, &layout);
    let padded = layout.padded;
    let golden_work = pixel_major(
        &file.f32("work")?,
        (IMAGE_CHANNELS, padded.width * padded.height),
    );
    println!("{}", compare("work", &work, &golden_work)?);
    let fused = engine.restore_tiles((&golden_work, padded))?;
    let golden_fused = pixel_major(
        &file.f32("fused")?,
        (IMAGE_CHANNELS, padded.width * padded.height),
    );
    println!("{}", compare("tiles", &fused, &golden_fused)?);
    let restored = restore::finish(&golden_fused, &layout, output);
    let golden_restored = pixel_major(
        &file.f32("restored")?,
        (IMAGE_CHANNELS, output.width * output.height),
    );
    println!("{}", compare("finish", &restored, &golden_restored)?);
    let fixed = restore::color_fix(&golden_restored, &golden_reference, output);
    let ours = restore::bytes(&fixed);
    let reference = file.u8("image")?;
    let different = differing(&ours, &reference, 0);
    println!("image: {different} of {} bytes differ", ours.len());
    ensure!(
        differing(&ours, &reference, 1) == 0 && different * 1000 < ours.len() * DIFFERENT_PER_MILLE,
        "the restored photo's bytes differ by more than 1, or more than {DIFFERENT_PER_MILLE} per \
         mille differ"
    );
    Ok(())
}
