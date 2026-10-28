//! `qwen3-tts parity`: each case's stages against the reference's goldens (`reference/dump.py`
//! documents them).
//!
//! Each stage gets the reference's own input — the speaker encoder its features, the prompt its
//! embedding and frames, the talker the reference's frames read in order, the code predictor its
//! talker state, the decoder its frames — so errors do not compound from stage to stage. Tensors
//! are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); the ids must be the reference's, one for
//! one. The speech tokenizer's frames and the likeliest codes after the reference's own are
//! compared by agreement, not exactly.

use std::path::Path;

use anyhow::{Result, ensure};
use compile::Checkpoint;
use harness::parity::compare;
use kernels::numpy::argmax;
use qwen3_tts::model::{CODE_GROUPS, CODEC_EOS, CODES, Frame, HIDDEN, Language};
use qwen3_tts::prompt::{Conditioning, Reference};
use qwen3_tts::sampling;
use qwen3_tts::session::Engine;
use serde::Deserialize;

/// The share of codes that must be the reference's: float order moves near ties, the quantizer's
/// nearest centroid, the talker's likeliest code, the code predictor's.
const AGREEMENT: f64 = 0.9;

#[derive(Deserialize)]
struct Case {
    text: String,
    language: Language,
    #[serde(default)]
    reference_text: String,
}

/// The reference's tensors for one case.
struct Goldens {
    case: Case,
    clip: Vec<f32>,
    features: Vec<f32>,
    embedding: Vec<f32>,
    /// The clip's latent and frames, with a transcript.
    clip_frames: Option<(Vec<f32>, Vec<Frame>)>,
    text_ids: Vec<i64>,
    reference_ids: Vec<i64>,
    prompt: Vec<f32>,
    trailing: Vec<f32>,
    pad: Vec<f32>,
    logits: Vec<f32>,
    hidden: Vec<f32>,
    frames: Vec<Frame>,
    ended: bool,
    predictor: Vec<f32>,
    speech: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let case: Case = serde_json::from_slice(&file.u8("case")?)?;
        let frames = |name: &str| -> Result<Vec<Frame>> {
            let codes = file.i64(name)?.into_iter().map(u32::try_from);
            let codes = codes.collect::<Result<Vec<_>, _>>()?;
            Ok(codes.as_chunks::<CODE_GROUPS>().0.to_vec())
        };
        let icl = !case.reference_text.is_empty();
        Ok(Self {
            clip: file.f32("reference")?,
            features: file.f32("speaker.features")?,
            embedding: file.f32("speaker.embedding")?,
            clip_frames: if icl {
                Some((file.f32("codec.latent")?, frames("codec.codes")?))
            } else {
                None
            },
            text_ids: file.i64("text.ids")?,
            reference_ids: if icl {
                file.i64("reference.ids")?
            } else {
                Vec::new()
            },
            prompt: file.f32("prompt")?,
            trailing: file.f32("trailing")?,
            pad: file.f32("pad")?,
            logits: file.f32("talker.logits")?,
            hidden: file.f32("talker.hidden")?,
            frames: frames("codes")?,
            ended: file.u8("ended")?[0] == 1,
            predictor: file.f32("predictor.logits")?,
            speech: file.f32("speech")?,
            case,
        })
    }
}

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Goldens::read(path)?;
    println!("# {}", path.display());
    speaker(engine, &goldens)?;
    let ids = ids(engine, &goldens)?;
    let (speech, reference) = ids;
    let clip_frames = encode(engine, &goldens)?;
    let reference = reference.as_deref().zip(clip_frames.as_deref());
    let conditioning = Conditioning {
        language: goldens.case.language,
        speech: &speech,
        speaker: &goldens.embedding,
        reference: reference.map(|(ids, frames)| Reference { ids, frames }),
    };
    let prompt = engine.prompt(&conditioning);
    println!("{}", compare("prompt", &prompt.rows, &goldens.prompt)?);
    if prompt.trailing.is_empty() && goldens.trailing.is_empty() {
        println!("trailing: none, the prompt holds all the text");
    } else {
        println!(
            "{}",
            compare("trailing", &prompt.trailing, &goldens.trailing)?
        );
    }
    println!("{}", compare("pad", &prompt.pad, &goldens.pad)?);
    talker(engine, &goldens, &prompt)?;
    predictor(engine, &goldens)?;
    // An x-vector case has no reference frames.
    let reference_frames = clip_frames.as_deref().unwrap_or_default();
    let speech = engine.decode(reference_frames, &goldens.frames);
    println!("{}", compare("speech", &speech, &goldens.speech)?);
    Ok(())
}

/// The clip's features and embedding.
fn speaker(engine: &Engine, goldens: &Goldens) -> Result<()> {
    let features = engine.speaker_features(&goldens.clip);
    let embedding = engine.speaker_embedding(&goldens.features);
    println!(
        "{}\n{}",
        compare("speaker features", &features, &goldens.features)?,
        compare("speaker embedding", &embedding, &goldens.embedding)?
    );
    Ok(())
}

/// The case's text and transcript ids, which must be the reference's.
fn ids(
    engine: &Engine,
    goldens: &Goldens,
) -> Result<(qwen3_tts::prompt::Speech, Option<Vec<u32>>)> {
    let speech = engine.speech_ids(&goldens.case.text)?;
    ensure!(
        speech
            .text
            .iter()
            .map(|&id| i64::from(id))
            .eq(goldens.text_ids.iter().copied()),
        "the text's ids are not the reference's"
    );
    let reference = match goldens.clip_frames {
        Some(_) => {
            let ids = engine.reference_ids(&goldens.case.reference_text)?;
            ensure!(
                ids.iter()
                    .map(|&id| i64::from(id))
                    .eq(goldens.reference_ids.iter().copied()),
                "the transcript's ids are not the reference's"
            );
            Some(ids)
        }
        None => None,
    };
    println!("ids: {} text tokens identical", speech.text.len());
    Ok((speech, reference))
}

/// With a transcript, the clip's frames: the latent compared, the codes by agreement. The
/// reference's frames are what the rest reads.
fn encode(engine: &mut Engine, goldens: &Goldens) -> Result<Option<Vec<Frame>>> {
    let Some((latent, frames)) = &goldens.clip_frames else {
        return Ok(None);
    };
    let encoded = engine.encode(&goldens.clip);
    println!("{}", compare("codec latent", &encoded.latent, latent)?);
    let agreed = encoded
        .codes
        .iter()
        .zip(frames)
        .map(|(ours, theirs)| {
            ours.iter()
                .zip(theirs)
                .filter(|(first, second)| first == second)
                .count()
        })
        .sum::<usize>();
    let total = frames.len() * CODE_GROUPS;
    ensure!(
        agreed as f64 / total as f64 >= AGREEMENT,
        "codec codes: {agreed} of {total} the reference's"
    );
    println!("codec codes: {agreed} of {total} the reference's");
    Ok(Some(frames.clone()))
}

/// The prompt's state and logits, then the reference's frames read in order: every state
/// compared, the likeliest first code after each the reference's (the end of speech after the
/// last of a speech that ended).
fn talker(
    engine: &mut Engine,
    goldens: &Goldens,
    prompt: &qwen3_tts::prompt::Prompt,
) -> Result<()> {
    let frames = &goldens.frames;
    let prefill = engine.prefill(prompt);
    println!(
        "{}\n{}",
        compare("prefill hidden", &prefill.hidden, &goldens.hidden[..HIDDEN])?,
        compare("prefill logits", &prefill.logits, &goldens.logits)?
    );
    let read = frames.len() - usize::from(!goldens.ended);
    let forced = engine.force(prompt, &frames[..read]);
    let hidden: Vec<f32> = forced
        .iter()
        .take(frames.len() - 1)
        .flat_map(|prediction| prediction.hidden.iter().copied())
        .collect();
    println!(
        "{}",
        compare("talker hidden", &hidden, &goldens.hidden[HIDDEN..])?
    );
    let drawn: Vec<u32> = frames.iter().map(|frame| frame[0]).collect();
    let logits = std::iter::once(&prefill)
        .chain(&forced)
        .map(|prediction| &prediction.logits);
    let wanted = drawn
        .iter()
        .copied()
        .chain(goldens.ended.then_some(CODEC_EOS));
    let agreed = logits
        .zip(wanted)
        .enumerate()
        .filter(|(index, (logits, wanted))| sampling::greedy(logits, &drawn[..*index]) == *wanted)
        .count();
    let total = read + 1;
    ensure!(
        agreed as f64 / total as f64 >= AGREEMENT,
        "talker: {agreed} of {total} first codes the reference's"
    );
    println!("talker: {agreed} of {total} first codes the reference's");
    Ok(())
}

/// The code predictor's logits over the first frame from its talker state, and how many of its
/// 15 likeliest codes are the reference's.
fn predictor(engine: &mut Engine, goldens: &Goldens) -> Result<()> {
    let frame = &goldens.frames[0];
    let logits = engine.predictor_logits(&goldens.hidden[..HIDDEN], frame);
    println!(
        "{}",
        compare("predictor logits", &logits, &goldens.predictor)?
    );
    let agreed = logits
        .as_chunks::<CODES>()
        .0
        .iter()
        .zip(&frame[1..])
        .filter(|(logits, code)| argmax(*logits) == **code as usize)
        .count();
    ensure!(
        agreed as f64 / (CODE_GROUPS - 1) as f64 >= AGREEMENT,
        "predictor: {agreed} of {} codes the reference's",
        CODE_GROUPS - 1
    );
    println!(
        "predictor: {agreed} of {} codes the reference's",
        CODE_GROUPS - 1
    );
    Ok(())
}
