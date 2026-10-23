//! `indra-ocr parity`: every network call of the reference fed its own inputs (the goldens'
//! `<net>.<i>.<name>` tensors, `reference/dump.py`) against its outputs, by relative L2 and,
//! for decisions, by how many differ; the pipeline's steps from the reference's inputs (the
//! page turned upright, the detector's inputs, the detected lines); and the whole page read
//! from the image against the reference's JSON.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use compile::Checkpoint;
use harness::parity::compare;
use indra_ocr::lines::{Detection, adaptive_side, decode, detector_input, to_page};
use indra_ocr::math::decoder::{Reading, TOKENS};
use indra_ocr::math::encoder::Memory;
use indra_ocr::model::BASE;
use indra_ocr::page::{orientation_input, preview, skew, straightened};
use indra_ocr::recognizer::{BEST, Batch};
use indra_ocr::session::{Engine, Networks, Options, Upright};
use indra_ocr::{Pixels, Size};
use kernels::conv::Grid;
use serde_json::Value;

/// The networks whose calls are compared one by one, beside the detector's and the decoder's.
const RECOGNIZERS: [&str; 4] = ["rec", "rec_cjk", "rec_rtl", "rec_world"];

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Checkpoint::read(path)?;
    println!("# {}", path.display());
    let gray = image(&goldens, "image")?;
    let math = goldens.contains("mathcls.0.x");
    {
        let networks = engine.networks();
        upright(networks, &goldens, &gray)?;
        detector(networks, &goldens)?;
        crops(networks, &goldens, math)?;
        classifiers(networks, &goldens)?;
        for name in RECOGNIZERS {
            for call in calls(&goldens, name, "x") {
                recognizer(networks, &goldens, name, &call)?;
            }
        }
        order(networks, &goldens)?;
        math_encoder(networks, &goldens)?;
        math_decoder(networks, &goldens)?;
    }
    let page = engine.read(&gray, Options { math })?;
    let ours = serde_json::to_value(&page)?;
    let reference: Value = serde_json::from_slice(&goldens.u8("page")?)?;
    print_page(&ours, &reference);
    Ok(())
}

/// The prefixes `<net>.<i>` of a network's calls, in order: each holds its tensor `field`.
fn calls(goldens: &Checkpoint, net: &str, field: &str) -> Vec<String> {
    (0..)
        .map(|index| format!("{net}.{index}"))
        .take_while(|call| goldens.contains(&format!("{call}.{field}")))
        .collect()
}

fn image(goldens: &Checkpoint, name: &str) -> Result<Pixels> {
    let shape = goldens.shape(name)?;
    Ok(Pixels {
        size: Size {
            width: shape[1],
            height: shape[0],
        },
        channels: 1,
        values: goldens.u8(name)?,
    })
}

fn show(name: &str, ours: &[f32], reference: &[f32]) -> Result<()> {
    println!("{}", compare(name, ours, reference)?);
    Ok(())
}

/// How many of `ours` differ from `reference`.
fn differing<T: PartialEq>(name: &str, ours: &[T], reference: &[T]) {
    let count = ours
        .iter()
        .zip(reference)
        .filter(|(ours, reference)| ours != reference)
        .count();
    println!("{name}: {count} of {} differ", reference.len());
}

fn integers(goldens: &Checkpoint, name: &str) -> Result<Vec<usize>> {
    let values = goldens.i64(name)?;
    Ok(values.into_iter().map(|value| value as usize).collect())
}

/// The skew, the turn and the upright page; the orientation network's input and output.
fn upright(networks: Networks, goldens: &Checkpoint, gray: &Pixels) -> Result<()> {
    let upright = networks.upright(gray);
    let reference = image(goldens, "upright")?;
    println!(
        "turn {} (reference {}), skew {} (reference {})",
        upright.turn,
        goldens.i64("turn")?[0],
        upright.skew,
        goldens.f32("skew")?[0]
    );
    differing("upright pixels", &upright.page.values, &reference.values);
    let small = preview(gray);
    let input = orientation_input(&straightened(&small, skew(&small)));
    show("orient input", &input, &goldens.f32("orient.0.x")?)?;
    let probabilities = networks.model.orientation.forward(
        networks.team,
        networks.scratch,
        &goldens.f32("orient.0.x")?,
    );
    show("orient", &probabilities, &goldens.f32("orient.0.p")?)
}

/// The detector's inputs from the upright page, its maps from its inputs, and the lines
/// decoded from its maps.
fn detector(networks: Networks, goldens: &Checkpoint) -> Result<()> {
    let page = image(goldens, "upright")?;
    let long_side = networks.model.long_side;
    let (input, _, mut scale) = detector_input(&page, long_side);
    show("det input", &input, &goldens.f32("det.0.x")?)?;
    let mut detections = Vec::new();
    for call in calls(goldens, "det", "x") {
        let shape = goldens.shape(&format!("{call}.x"))?;
        let grid = Grid {
            images: 1,
            height: shape[2],
            width: shape[3],
        };
        let maps = networks.model.detector.forward(
            networks.team,
            networks.scratch,
            &goldens.f32(&format!("{call}.x"))?,
            grid,
        );
        let reference = pixel_major(&goldens.f32(&format!("{call}.out"))?, maps.len() / 3);
        show(&format!("{call} maps"), &maps, &reference)?;
        let half = Grid {
            height: shape[2] / 2,
            width: shape[3] / 2,
            ..grid
        };
        let found = decode(&reference, half);
        if let Some(side) = adaptive_side(&found, long_side) {
            scale = detector_input(&page, side).2;
        }
        detections = found;
    }
    to_page(&mut detections, scale, page.size);
    let ours: Vec<f32> = detections
        .iter()
        .flat_map(|detection| detection.line.into_iter().chain([detection.score]))
        .collect();
    show("boxes", &ours, &goldens.f32("boxes")?)
}

/// The recognizers' and the math encoder's inputs, cropped from the upright page at the
/// reference's lines, against theirs.
fn crops(networks: Networks, goldens: &Checkpoint, math: bool) -> Result<()> {
    let upright = Upright {
        page: image(goldens, "upright")?,
        turn: integers(goldens, "turn")?[0],
        skew: f64::from(goldens.f32("skew")?[0]),
    };
    let boxes = goldens.f32("boxes")?;
    let detections: Vec<Detection> = boxes
        .chunks(5)
        .map(|row| Detection {
            line: [row[0], row[1], row[2], row[3]],
            score: row[4],
        })
        .collect();
    let lines = networks.classify(upright, &detections, Options { math });
    let mut counts = std::collections::HashMap::new();
    for work in lines.work() {
        let name = &networks.model.groups[work.group].name;
        let net = if name == BASE {
            "rec".to_string()
        } else {
            format!("rec_{name}")
        };
        let index = counts.entry(net.clone()).or_insert(0);
        let call = format!("{net}.{index}");
        *index += 1;
        let padded = networks.recognizer_input(&lines, work);
        show(
            &format!("{call} crops"),
            &padded.pixels,
            &goldens.f32(&format!("{call}.x"))?,
        )?;
    }
    for (index, batch) in lines.math_batches().enumerate() {
        let call = format!("math_enc.{index}");
        show(
            &format!("{call} crops"),
            &lines.math_input(batch).pixels,
            &goldens.f32(&format!("{call}.x"))?,
        )?;
    }
    Ok(())
}

/// Channel-major maps (`channels × pixels`) as pixel-major ones.
fn pixel_major(maps: &[f32], pixels: usize) -> Vec<f32> {
    let channels = maps.len() / pixels;
    (0..pixels * channels)
        .map(|at| maps[(at % channels) * pixels + at / channels])
        .collect()
}

/// A call's line crops (`x`, `lines × 1 × height × width`) and widths.
fn batch_of(goldens: &Checkpoint, call: &str) -> Result<(Vec<f32>, Vec<usize>, [usize; 2])> {
    let shape = goldens.shape(&format!("{call}.x"))?;
    Ok((
        goldens.f32(&format!("{call}.x"))?,
        integers(goldens, &format!("{call}.widths"))?,
        [shape[2], shape[3]],
    ))
}

/// The script classifier's probabilities and the math classifier's.
fn classifiers(networks: Networks, goldens: &Checkpoint) -> Result<()> {
    for (net, classifier) in [
        ("script", &networks.model.script),
        ("mathcls", &networks.model.math.classifier),
    ] {
        for call in calls(goldens, net, "x") {
            let (pixels, widths, [height, width]) = batch_of(goldens, &call)?;
            let batch = Batch {
                pixels: &pixels,
                widths: &widths,
                height,
                width,
            };
            let logits = classifier.forward(networks.team, networks.scratch, batch);
            let probabilities: Vec<f32> = if net == "script" {
                logits
                    .chunks(logits.len() / widths.len())
                    .flat_map(|row| {
                        let mut row = row.to_vec();
                        kernels::ops::softmax(&mut row);
                        row
                    })
                    .collect()
            } else {
                logits.into_iter().map(kernels::ops::sigmoid).collect()
            };
            show(&call, &probabilities, &goldens.f32(&format!("{call}.p"))?)?;
        }
    }
    Ok(())
}

/// A recognizer's outputs over its call's crops.
fn recognizer(networks: Networks, goldens: &Checkpoint, net: &str, call: &str) -> Result<()> {
    let name = net.strip_prefix("rec_").unwrap_or(indra_ocr::model::BASE);
    let group = networks
        .model
        .groups
        .iter()
        .find(|group| group.name == name)
        .with_context(|| format!("no recognizer {name}"))?;
    let (pixels, widths, [height, width]) = batch_of(goldens, call)?;
    let outputs = group.recognizer.forward(
        networks.team,
        networks.scratch,
        Batch {
            pixels: &pixels,
            widths: &widths,
            height,
            width,
        },
    );
    let field = |part: &str| format!("{call}.{part}");
    differing(
        &field("lengths"),
        &outputs.lengths,
        &integers(goldens, &field("lengths"))?,
    );
    let ids: Vec<usize> = outputs.ids.iter().map(|&id| id as usize).collect();
    differing(&field("ids"), &ids, &integers(goldens, &field("ids"))?);
    show(
        &field("conf"),
        &outputs.confidences,
        &goldens.f32(&field("conf"))?,
    )?;
    show(
        &field("blank"),
        &outputs.blanks,
        &goldens.f32(&field("blank"))?,
    )?;
    show(
        &field("topv"),
        &outputs.top_values,
        &goldens.f32(&field("topv"))?,
    )?;
    let top: Vec<usize> = outputs.top_ids.iter().map(|&id| id as usize).collect();
    ensure!(
        top.len() == outputs.ids.len() * BEST,
        "{call}: {BEST} classes a step"
    );
    differing(&field("topi"), &top, &integers(goldens, &field("topi"))?);
    Ok(())
}

/// The order network's heads over the reference's features.
fn order(networks: Networks, goldens: &Checkpoint) -> Result<()> {
    for call in calls(goldens, "order", "feats") {
        let relations = networks
            .model
            .order
            .forward(networks.team, &goldens.f32(&format!("{call}.feats"))?);
        for (part, ours) in [
            ("a", &relations.order),
            ("cls", &relations.classes),
            ("same", &relations.same),
        ] {
            show(
                &format!("{call}.{part}"),
                ours,
                &goldens.f32(&format!("{call}.{part}"))?,
            )?;
        }
    }
    Ok(())
}

/// The math encoder over its calls' crops.
fn math_encoder(networks: Networks, goldens: &Checkpoint) -> Result<()> {
    let encoder = &networks.model.math.encoder;
    for call in calls(goldens, "math_enc", "x") {
        let (pixels, widths, [height, width]) = batch_of(goldens, &call)?;
        let batch = Batch {
            pixels: &pixels,
            widths: &widths,
            height,
            width,
        };
        let memory = encoder.forward(networks.team, networks.scratch, batch);
        show(
            &format!("{call}.memory"),
            &memory.values,
            &goldens.f32(&format!("{call}.memory"))?,
        )?;
        let mask: Vec<f32> = memory
            .lengths
            .iter()
            .flat_map(|&length| {
                (0..memory.steps).map(move |step| f32::from(u8::from(step < length)))
            })
            .collect();
        show(
            &format!("{call}.mask"),
            &mask,
            &goldens.f32(&format!("{call}.mask"))?,
        )?;
    }
    Ok(())
}

/// The decoder's logits at each step of the reference's readings, from the reference's memory
/// and tokens: a batch's steps run from position 0 to the next batch's.
fn math_decoder(networks: Networks, goldens: &Checkpoint) -> Result<()> {
    let decoder = &networks.model.math.decoder;
    let steps = calls(goldens, "math_dec_step", "token");
    let mut next = 0;
    for batch in calls(goldens, "math_dec_init", "memory") {
        let shape = goldens.shape(&format!("{batch}.memory"))?;
        let (lines, length) = (shape[0], shape[1]);
        let mask = goldens.f32(&format!("{}.mask", steps[next]))?;
        let lengths = mask
            .chunks(length)
            .map(|row| row.iter().filter(|&&valid| valid > 0.0).count())
            .collect();
        let memory = Memory {
            steps: length,
            lengths,
            values: goldens.f32(&format!("{batch}.memory"))?,
        };
        let mut state = decoder.start(networks.team, networks.scratch, &memory);
        let (mut ours, mut reference) = (Vec::new(), Vec::new());
        while next < steps.len()
            && (ours.is_empty() || goldens.i64(&format!("{}.pos", steps[next]))?[0] != 0)
        {
            let tokens = goldens.i64(&format!("{}.token", steps[next]))?;
            let readings: Vec<Reading> = (0..lines)
                .map(|line| Reading {
                    line,
                    token: tokens[line] as u32,
                })
                .collect();
            ours.extend(decoder.step(networks.team, networks.scratch, &mut state, &readings));
            reference.extend(goldens.f32(&format!("{}.logits", steps[next]))?);
            next += 1;
        }
        show(&format!("{batch} decoder logits"), &ours, &reference)?;
        differing(
            &format!("{batch} decoder tokens"),
            &best_tokens(&ours),
            &best_tokens(&reference),
        );
    }
    Ok(())
}

/// Each row's likeliest token.
fn best_tokens(logits: &[f32]) -> Vec<usize> {
    logits
        .chunks(TOKENS)
        .map(|row| {
            (0..TOKENS).fold(
                0,
                |best, token| if row[token] > row[best] { token } else { best },
            )
        })
        .collect()
}

/// The page's lines, blocks and Markdown against the reference's, and the lines whose texts
/// differ.
fn print_page(ours: &Value, reference: &Value) {
    let lines = |page: &Value| page["lines"].as_array().cloned().unwrap_or_default();
    let (our_lines, reference_lines) = (lines(ours), lines(reference));
    let field = |lines: &[Value], name: &str| {
        lines
            .iter()
            .map(|line| line[name].clone())
            .collect::<Vec<_>>()
    };
    println!(
        "page: {} lines (reference {})",
        our_lines.len(),
        reference_lines.len()
    );
    differing(
        "line texts",
        &field(&our_lines, "text"),
        &field(&reference_lines, "text"),
    );
    differing(
        "line boxes",
        &field(&our_lines, "box"),
        &field(&reference_lines, "box"),
    );
    for (ours, reference) in our_lines.iter().zip(&reference_lines) {
        if ours["text"] != reference["text"] {
            println!(
                "  ours {} {}\n  ref  {} {}",
                ours["conf"], ours["text"], reference["conf"], reference["text"]
            );
        }
    }
    for name in ["rotation", "skew", "vertical", "blocks", "markdown"] {
        println!(
            "{name}: {}",
            if ours[name] == reference[name] {
                "same"
            } else {
                "differs"
            }
        );
    }
}
