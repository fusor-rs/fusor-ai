//! The synthesis loop: one `step` per 80 ms frame.
//!
//! Per text fragment: prefill the prompt; then per frame `z → flow head → latent`, the stop
//! head on `z` (stop when `logit₁ > logit₀` after frame 3, keeping that frame), the latent into
//! the streaming vocoder, and unless stopped `latent → aggregator → LLM → next z`. Trailing
//! silence is trimmed per fragment (`silence`) by holding back silent chunks while streaming.

use serde::{Deserialize, Serialize};

use blocks::transformer::{Buffers, KvCache, Mask, Span};
use bundle::{Store, Tokenizer};
use kernels::rng::Rng;
use team::Team;

use crate::Error;
use crate::flow_head::{self, LATENT};
use crate::model::{HIDDEN, Model, QWEN2_SHAPE, STOP_LOGITS};
use crate::preset::{Preset, Schedules};
use crate::prompt::{Language, Prompt, Prompter, Voice};
use crate::silence::SilenceGate;
use crate::text::{self, Segmentation};
use crate::vocoder::VocoderState;

/// LLM positions (prompt + frames) one fragment may use.
const LLM_CAPACITY: usize = 1024;
/// Prompt rows per prefill call.
const PREFILL_ROWS: usize = 64;
const DEFAULT_MAX_FRAMES: usize = 1000;
/// The stop head is ignored on frames `0..=MIN_STOP_FRAME` (the reference's `step > 3`).
const MIN_STOP_FRAME: usize = 3;
const DEFAULT_SEED: u64 = 1895;

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub text: String,
    pub voice: Voice,
    #[serde(default)]
    pub preset: Preset,
    pub seed: Option<u64>,
    /// Target language line ("English", "German", …); omitted for the built-in voices.
    pub language: Option<String>,
    pub max_frames: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub fragments: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Progress {
    pub fragment: usize,
    pub fragments: usize,
    pub frame: usize,
    pub done: bool,
}

pub struct Engine {
    model: Model,
    prompter: Prompter,
    team: Team,
    llm_cache: KvCache,
    llm: Buffers,
    flow: flow_head::Buffers,
    vocoder: VocoderState,
    schedules: Schedules,
    job: Option<Job>,
    audio: Vec<f32>,
}

struct Job {
    request: Request,
    language: Option<Language>,
    fragments: Vec<String>,
    fragment: usize,
    frame: Option<Frame>,
    rng: Rng,
    silence: SilenceGate,
}

/// The state between two frames of a fragment.
struct Frame {
    index: usize,
    position: usize,
    budget: usize,
    z: Vec<f32>,
    history: Vec<f32>,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let prompter = Prompter::new(Tokenizer::take(store)?);
        let team = Team::global();
        let threads = team.threads();
        let schedules = Schedules::new(&model.flow);
        let vocoder = VocoderState::new(&model.vocoder, threads);
        Ok(Self {
            model,
            prompter,
            llm_cache: KvCache::new(QWEN2_SHAPE, LLM_CAPACITY),
            llm: QWEN2_SHAPE.buffers(PREFILL_ROWS, threads),
            flow: flow_head::Buffers::new(threads),
            vocoder,
            team,
            schedules,
            job: None,
            audio: Vec::new(),
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {
        self.job = None;
    }
}

impl Engine {
    pub fn start(&mut self, request: Request) -> Result<Plan, Error> {
        let language = request.language.as_deref().and_then(Language::parse);
        let segmentation = match (&request.voice, &language) {
            (_, Some(language)) if language.is_multilingual() => Segmentation::Verbatim,
            (Voice::Builtin(_), _) => Segmentation::Sentences,
            (Voice::Design(text) | Voice::Freeform(text), _) if text.trim().is_empty() => {
                return Err(Error::Request("the voice description is empty".into()));
            }
            (Voice::Design(_) | Voice::Freeform(_), _) => Segmentation::Whole,
        };
        let fragments = text::fragments(&request.text, segmentation);
        if fragments.is_empty() {
            return Err(Error::Request("the text has nothing to say".into()));
        }
        let rng = Rng::new(request.seed.unwrap_or(DEFAULT_SEED));
        self.audio.clear();
        self.job = Some(Job {
            request,
            language,
            fragments: fragments.clone(),
            fragment: 0,
            frame: None,
            rng,
            silence: SilenceGate::default(),
        });
        Ok(Plan { fragments })
    }

    /// Samples produced since the last call.
    pub fn take_audio(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.audio)
    }

    /// Generates one frame (prefilling first when a fragment starts).
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no synthesis in progress".into()))?;
        let frame = match job.frame.take() {
            Some(frame) => frame,
            None => self.begin_fragment(&job)?,
        };
        let mut noise = vec![0.0; LATENT];
        job.rng.fill_normal(&mut noise);
        let latent = self.sample(&frame.z, &frame.history, &noise, job.request.preset);
        let [keep, end] = self.stop_logits(&frame.z);
        let stop = end > keep && frame.index > MIN_STOP_FRAME;
        let mut samples = Vec::new();
        self.model
            .vocoder
            .push(&self.team, &mut self.vocoder, &latent, &mut samples);
        let last = stop || frame.index + 1 >= frame.budget;
        if last {
            self.model
                .vocoder
                .finish(&self.team, &mut self.vocoder, &mut samples);
        } else {
            let embedding = self.aggregate(&latent);
            let z = self.decode(&embedding, frame.position);
            job.frame = Some(Frame {
                index: frame.index + 1,
                position: frame.position + 1,
                budget: frame.budget,
                z,
                history: latent,
            });
        }
        job.silence.push(&samples, &mut self.audio);
        let progress = Progress {
            fragment: job.fragment,
            fragments: job.fragments.len(),
            frame: frame.index,
            done: false,
        };
        if !last {
            self.job = Some(job);
            return Ok(progress);
        }
        job.silence.finish(&mut self.audio);
        job.fragment += 1;
        let done = job.fragment == job.fragments.len();
        if !done {
            self.job = Some(job);
        }
        Ok(Progress { done, ..progress })
    }

    fn begin_fragment(&mut self, job: &Job) -> Result<Frame, Error> {
        let prompt = self.prompter.build(
            &job.fragments[job.fragment],
            &job.request.voice,
            job.language.as_ref(),
        )?;
        if prompt.ids.len() >= LLM_CAPACITY {
            return Err(Error::Request("the text fragment is too long".into()));
        }
        self.vocoder.reset();
        let z = self.prefill(&prompt);
        let budget = job
            .request
            .max_frames
            .unwrap_or(DEFAULT_MAX_FRAMES)
            .min(LLM_CAPACITY - prompt.ids.len());
        Ok(Frame {
            index: 0,
            position: prompt.ids.len(),
            budget,
            z,
            history: vec![0.0; LATENT],
        })
    }

    pub fn prompter(&self) -> &Prompter {
        &self.prompter
    }

    /// Runs the prompt through the LLM from position 0; returns `z` of its last position.
    pub fn prefill(&mut self, prompt: &Prompt) -> Vec<f32> {
        let embeddings = self.model.llm.embeddings();
        for first in (0..prompt.ids.len()).step_by(PREFILL_ROWS) {
            let rows = (prompt.ids.len() - first).min(PREFILL_ROWS);
            let hidden = self.llm.hidden.get_mut();
            for (offset, row) in hidden[..rows * HIDDEN]
                .as_chunks_mut::<HIDDEN>()
                .0
                .iter_mut()
                .enumerate()
            {
                let position = first + offset;
                if position == prompt.speaker_row {
                    row.fill(0.0);
                } else {
                    embeddings.dequantize_row(prompt.ids[position] as usize, row);
                }
            }
            self.llm_forward(Span {
                rows,
                first,
                mask: Mask::Causal,
            });
        }
        self.llm_output((prompt.ids.len() - 1) % PREFILL_ROWS)
    }

    /// One LLM step: `embedding` at `position` (after a prefill of the positions before it).
    pub fn decode(&mut self, embedding: &[f32], position: usize) -> Vec<f32> {
        self.llm.hidden.get_mut()[..HIDDEN].copy_from_slice(embedding);
        self.llm_forward(Span {
            rows: 1,
            first: position,
            mask: Mask::Causal,
        });
        self.llm_output(0)
    }

    fn llm_forward(&self, span: Span) {
        let body = &self.model.llm.qwen2.body;
        self.team
            .run(|ctx| body.forward(ctx, &self.llm_cache, &self.llm, span));
    }

    /// `z`: the final norm of a hidden row.
    fn llm_output(&self, row: usize) -> Vec<f32> {
        let mut z = vec![0.0; HIDDEN];
        self.model.llm.qwen2.final_norm(&self.llm, row, &mut z);
        z
    }

    /// The flow head: one frame latent from `z`, the previous latent and the initial noise.
    pub fn sample(&self, z: &[f32], history: &[f32], noise: &[f32], preset: Preset) -> Vec<f32> {
        let schedule = self.schedules.get(preset);
        let input = flow_head::FrameInput { z, history, noise };
        self.team
            .run(|ctx| self.model.flow.sample(ctx, &self.flow, &input, schedule));
        self.flow.output(LATENT).to_vec()
    }

    /// The stop head's `[continue, stop]` logits.
    pub fn stop_logits(&self, z: &[f32]) -> [f32; STOP_LOGITS] {
        let mut logits = [0.0; STOP_LOGITS];
        self.model
            .stop
            .apply(z, &mut Default::default(), &mut logits);
        logits
    }

    /// The aggregator: a frame latent → the LLM's next input embedding.
    pub fn aggregate(&self, latent: &[f32]) -> Vec<f32> {
        self.team
            .run(|ctx| self.model.aggregator.embed(ctx, &self.flow, latent));
        self.flow.output(HIDDEN).to_vec()
    }

    /// Decodes a whole latent sequence (`frames × 64`) to audio, without silence trimming.
    pub fn vocode(&mut self, latents: &[f32]) -> Vec<f32> {
        self.vocoder.reset();
        let mut samples = Vec::new();
        for latent in latents.as_chunks::<LATENT>().0 {
            self.model
                .vocoder
                .push(&self.team, &mut self.vocoder, latent, &mut samples);
        }
        self.model
            .vocoder
            .finish(&self.team, &mut self.vocoder, &mut samples);
        samples
    }
}
