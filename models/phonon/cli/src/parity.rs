//! `phonon parity`: each stage, teacher-forced, against the reference's goldens
//! (`reference/`).
//!
//! Each stage gets the reference's own inputs (tokens, previous latent, noise, latents), so errors
//! do not compound through the autoregressive loop. Reported as relative L2
//! (`‖ours − ref‖ / ‖ref‖`), worst case over frames.

use std::path::Path;

use anyhow::{Result, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::{max_abs, relative_l2};
use phonon::FRAME_SAMPLES;
use phonon::model::LATENT;
use phonon::pipeline::Stage;
use phonon::prompt::VOICES;
use phonon::session::{EOS_THRESHOLD, Engine};

pub(crate) const SENTENCE_A: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";

// The reference's tensors for sentence A with the default voice: the tokens, the static
// embeddings, per frame the noise it drew, the latent it sampled and the end-of-speech logit, and
// the whole waveform.
struct Goldens {
    tokens: Vec<u32>,
    voice_emb: Vec<f32>,
    conditions: Vec<f32>,
    text_emb: Vec<f32>,
    noise: Vec<f32>,
    latent: Vec<f32>,
    eos_logit: Vec<f32>,
    pcm: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let tokens = file.i64("tokens")?;
        Ok(Self {
            tokens: tokens
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?,
            voice_emb: file.f32("voice_emb")?,
            conditions: file.f32("conditions")?,
            text_emb: file.f32("text_emb")?,
            noise: file.f32("noise")?,
            latent: file.f32("latent")?,
            eos_logit: file.f32("eos_logit")?,
            pcm: file.f32("pcm")?,
        })
    }

    fn frames(&self) -> usize {
        self.eos_logit.len()
    }

    fn latent(&self, frame: usize) -> &[f32] {
        &self.latent[frame * LATENT..(frame + 1) * LATENT]
    }

    fn noise(&self, frame: usize) -> &[f32] {
        &self.noise[frame * LATENT..(frame + 1) * LATENT]
    }
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let chunks = engine.chunks(SENTENCE_A)?;
    ensure!(
        chunks.len() == 1 && chunks[0].tokens == goldens.tokens,
        "tokens differ:\n ours {:?}\n ref  {:?}",
        chunks.iter().map(|chunk| &chunk.tokens).collect::<Vec<_>>(),
        goldens.tokens
    );
    println!(
        "tokens: {} identical; frame budget {}, {} frame(s) after the end of speech",
        goldens.tokens.len(),
        chunks[0].frame_budget,
        chunks[0].frames_after_eos
    );
    for &stage in Stage::ALL {
        let report = match stage {
            Stage::Voice => voice(engine, &goldens)?,
            Stage::Text => text(engine, &goldens),
            Stage::Flow => flow(engine, &goldens)?,
            Stage::Decode => decode(engine, &goldens)?,
        };
        println!("{report}");
    }
    Ok(())
}

fn voice(engine: &Engine, goldens: &Goldens) -> Result<String> {
    let ours = engine.voice_embedding(VOICES[0].0)?;
    Ok(format!(
        "voice embedding: rel {:.2e}",
        relative_l2(&ours, &goldens.voice_emb)
    ))
}

fn text(engine: &Engine, goldens: &Goldens) -> String {
    let ours = engine.text_embeddings(&goldens.tokens);
    format!(
        "text embeddings: rel {:.2e}",
        relative_l2(&ours, &goldens.text_emb)
    )
}

// The flow LM after the voice and text prompt, fed the reference's previous latent (none for
// the first frame) and noise.
fn flow(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let conditions = relative_l2(&engine.conditions(), &goldens.conditions);
    engine.prompt(VOICES[0].0, &goldens.tokens)?;
    let frames = goldens.frames();
    let mut worst = 0f64;
    let mut worst_eos = 0f32;
    let mut decisions = 0;
    for frame in 0..frames {
        let previous = frame
            .checked_sub(1)
            .map(|previous| goldens.latent(previous));
        let step = engine.flow_step(previous, goldens.noise(frame));
        worst = worst.max(relative_l2(&step.latent, goldens.latent(frame)));
        let reference = goldens.eos_logit[frame];
        worst_eos = worst_eos.max((step.eos_logit - reference).abs());
        decisions += usize::from((step.eos_logit > EOS_THRESHOLD) == (reference > EOS_THRESHOLD));
    }
    Ok(format!(
        "conditions: rel {conditions:.2e}\nflow latent: worst rel {worst:.2e} over {frames} frames; \
         end-of-speech logit max abs {worst_eos:.2e}, decisions agree on {decisions}/{frames} frames"
    ))
}

// The Mimi decoder on the reference's whole latent stream.
fn decode(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let ours = engine.decode(&goldens.latent);
    ensure!(
        ours.len() == goldens.pcm.len() && ours.len() == goldens.frames() * FRAME_SAMPLES,
        "decoder: {} samples, reference {}",
        ours.len(),
        goldens.pcm.len()
    );
    Ok(format!(
        "decoder: {} samples, rel {:.2e}, max abs {:.2e}",
        ours.len(),
        relative_l2(&ours, &goldens.pcm),
        max_abs(&ours, &goldens.pcm)
    ))
}
