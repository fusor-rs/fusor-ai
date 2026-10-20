//! `ming parity`: each component, teacher-forced, against the reference's goldens (plan §8).
//!
//! Each stage gets the reference's own inputs (prompt ids, `z`, history, noise, latents), so
//! errors do not compound through the autoregressive loop. Reported as relative L2
//! (`‖ours − ref‖ / ‖ref‖`), worst case over frames.

use std::path::Path;

use anyhow::{Result, ensure};
use ming::dit::LATENT;
use ming::model::HIDDEN;
use ming::prompt::Voice;
use ming::session::{Engine, Preset};

use compile::Checkpoint;

const SENTENCE_A: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";

fn relative_l2(ours: &[f32], reference: &[f32]) -> f64 {
    let diff: f64 = ours
        .iter()
        .zip(reference)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let norm: f64 = reference.iter().map(|b| f64::from(*b).powi(2)).sum();
    (diff / norm).sqrt()
}

fn rows(values: &[f32], width: usize) -> std::slice::ChunksExact<'_, f32> {
    values.chunks_exact(width)
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let g = Checkpoint::read(goldens)?;
    let ids: Vec<u32> = g.i32("ids")?.into_iter().map(|i| i as u32).collect();
    let prompt = engine
        .prompter()
        .build(SENTENCE_A, &Voice::Builtin("US_James".into()), None)?;
    ensure!(
        prompt.ids == ids,
        "prompt ids differ:\n ours {:?}\n ref  {:?}",
        prompt.ids,
        ids
    );
    println!("prompt ids: {} tokens, identical", ids.len());

    let (z, agg, hist, noise, latent, stop) = (
        g.f32("z")?,
        g.f32("agg")?,
        g.f32("hist")?,
        g.f32("noise")?,
        g.f32("latent")?,
        g.f32("stop")?,
    );
    let frames = latent.len() / LATENT;

    let z0 = engine.prefill(&prompt);
    println!(
        "z after prefill: rel {:.2e}",
        relative_l2(&z0, &z[..HIDDEN])
    );
    let mut worst = 0f64;
    for (k, embedding) in rows(&agg, HIDDEN).enumerate() {
        let ours = engine.decode(embedding, ids.len() + k);
        worst = worst.max(relative_l2(&ours, &z[(k + 1) * HIDDEN..(k + 2) * HIDDEN]));
    }
    println!(
        "z per frame (LLM decode, teacher-forced): worst rel {worst:.2e} over {} frames",
        frames - 1
    );

    let mut worst = 0f64;
    let mut decisions = 0;
    for k in 0..frames {
        let (zk, hk, nk) = (
            &z[k * HIDDEN..(k + 1) * HIDDEN],
            &hist[k * LATENT..(k + 1) * LATENT],
            &noise[k * LATENT..(k + 1) * LATENT],
        );
        let ours = engine.sample(zk, hk, nk, Preset::Quality);
        worst = worst.max(relative_l2(&ours, &latent[k * LATENT..(k + 1) * LATENT]));
        let [keep, end] = engine.stop_logits(zk);
        let reference = &stop[k * 2..k * 2 + 2];
        decisions += usize::from((end > keep) == (reference[1] > reference[0]));
    }
    println!(
        "flow head latent: worst rel {worst:.2e}; stop decisions agree on {decisions}/{frames} frames"
    );

    let mut worst = 0f64;
    for (k, embedding) in rows(&agg, HIDDEN).enumerate() {
        worst = worst.max(relative_l2(
            &engine.aggregate(&latent[k * LATENT..(k + 1) * LATENT]),
            embedding,
        ));
    }
    println!("aggregator: worst rel {worst:.2e}");

    let reference = g.f32("wav")?;
    let ours = engine.vocode(&latent);
    ensure!(
        ours.len() == reference.len(),
        "vocoder: {} samples, reference {}",
        ours.len(),
        reference.len()
    );
    let max_abs = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "vocoder: {} samples, rel {:.2e}, max abs {max_abs:.2e}",
        ours.len(),
        relative_l2(&ours, &reference)
    );
    Ok(())
}
