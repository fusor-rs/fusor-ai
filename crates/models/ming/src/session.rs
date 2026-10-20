//! The synthesis loop (plan §6.3 "Generator"): one `step` per 80 ms frame.
//!
//! Per text fragment: prefill the prompt; then per frame `z → flow head → latent`, the stop
//! head on `z` (stop when `logit₁ > logit₀` after frame 3, keeping that frame), the latent into
//! the streaming vocoder, and unless stopped `latent → aggregator → LLM → next z`. Trailing
//! silence is trimmed per fragment as the reference does (0.1 s chunks, mean |x| ≤ 1e-3 is
//! silent, keep 0.3 s after the last loud chunk) by holding back silent chunks while streaming.

use serde::{Deserialize, Serialize};

use blocks::qwen2::{self, KvCache, Mask, Span};
use bundle::Store;
use kernels::rng::Rng;
use team::Team;

use crate::Error;
use crate::flow_head::{self, LATENT, Steps};
use crate::model::{HIDDEN, Model, QWEN2_SHAPE, TOKENIZER_PART};
use crate::prompt::{self, Prompt, Prompter, Voice};
use crate::text::{self, Segmentation};
use crate::vocoder::VocoderState;

pub const SAMPLE_RATE: u32 = 44_100;
/// LLM positions (prompt + frames) one fragment may use.
const LLM_CAPACITY: usize = 1024;
/// Prompt rows per prefill call.
const PREFILL_ROWS: usize = 64;
const DEFAULT_MAX_FRAMES: usize = 1000;
/// The stop head is ignored on frames `0..=MIN_STOP_FRAME` (the reference's `step > 3`).
const MIN_STOP_FRAME: usize = 3;
const CFG_SCALE: f32 = 2.0;
const DEFAULT_SEED: u64 = 1895;

/// EPSS knots in units of 1/32 (`fm/CFM.py`).
const EPSS_10: [u32; 11] = [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
const EPSS_7: [u32; 8] = [0, 2, 4, 6, 8, 16, 24, 32];
/// Guidance interval of the faster presets (plan §3: CFG only for `t ∈ [0.05, 0.8)`).
const GUIDED: std::ops::Range<f32> = 0.05..0.8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    /// 10 steps, CFG on every step: the reference sampler.
    #[default]
    Quality,
    /// 10 steps, CFG on 5 of them.
    Balanced,
    /// 7 steps, CFG on 3 of them (may slur numbers).
    Fast,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Quality, Preset::Balanced, Preset::Fast];
}

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub text: String,
    pub voice: Voice,
    #[serde(default)]
    pub preset: Preset,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Target language line ("English", "German", …); omitted for the built-in voices.
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
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
    llm: qwen2::Buffers,
    flow: flow_head::Buffers,
    vocoder: VocoderState,
    schedules: Vec<(Preset, Steps)>,
    job: Option<Job>,
    audio: Vec<f32>,
}

struct Job {
    request: Request,
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

impl Engine {
    /// Builds the engine from a complete store.
    pub fn new(store: &mut Store) -> Result<Self, Error> {
        if !store.complete() {
            return Err(Error::Format("the bundle is not fully loaded".into()));
        }
        let model = Model::load(store)?;
        let prompter = Prompter::new(&store.take_blob(TOKENIZER_PART)?)?;
        let team = Team::global();
        let threads = team.threads();
        let schedules = Preset::ALL
            .iter()
            .map(|&preset| (preset, schedule(&model, preset)))
            .collect();
        let vocoder = VocoderState::new(&model.vocoder, threads);
        Ok(Self {
            model,
            prompter,
            llm_cache: KvCache::new(QWEN2_SHAPE, LLM_CAPACITY),
            llm: qwen2::Buffers::new(QWEN2_SHAPE, PREFILL_ROWS, threads),
            flow: flow_head::Buffers::new(threads),
            vocoder,
            team,
            schedules,
            job: None,
            audio: Vec::new(),
        })
    }

    pub fn threads(&self) -> usize {
        self.team.threads()
    }

    pub fn start(&mut self, mut request: Request) -> Result<Plan, Error> {
        request.language = request.language.filter(|name| !name.trim().is_empty());
        let segmentation = match (&request.voice, &request.language) {
            (_, Some(language)) if prompt::is_multilingual(language) => Segmentation::Verbatim,
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
            fragments: fragments.clone(),
            fragment: 0,
            frame: None,
            rng,
            silence: SilenceGate::default(),
        });
        Ok(Plan { fragments })
    }

    pub fn cancel(&mut self) {
        self.job = None;
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
            job.request.language.as_deref(),
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
        self.team.run(|ctx| {
            self.model
                .llm
                .body
                .forward(ctx, &self.llm_cache, &self.llm, span)
        });
    }

    /// `z`: the final norm of a hidden row.
    fn llm_output(&self, row: usize) -> Vec<f32> {
        let mut z = vec![0.0; HIDDEN];
        self.model.llm.body.final_norm(&self.llm, row, &mut z);
        z
    }

    /// The flow head: one frame latent from `z`, the previous latent and the initial noise.
    pub fn sample(&self, z: &[f32], history: &[f32], noise: &[f32], preset: Preset) -> Vec<f32> {
        let schedule = &self
            .schedules
            .iter()
            .find(|(candidate, _)| *candidate == preset)
            .expect("every preset has a schedule")
            .1;
        let input = flow_head::FrameInput { z, history, noise };
        self.team
            .run(|ctx| self.model.flow.sample(ctx, &self.flow, &input, schedule));
        self.flow.output(LATENT).to_vec()
    }

    /// The stop head's `[continue, stop]` logits.
    pub fn stop_logits(&self, z: &[f32]) -> [f32; 2] {
        let mut logits = [0.0; 2];
        self.model
            .stop
            .apply(z, 1, &mut Default::default(), &mut logits);
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

fn schedule(model: &Model, preset: Preset) -> Steps {
    match preset {
        Preset::Quality => model.flow.steps(&EPSS_10, |_, _| true, CFG_SCALE),
        Preset::Balanced => model
            .flow
            .steps(&EPSS_10, |_, time| GUIDED.contains(&time), CFG_SCALE),
        Preset::Fast => model
            .flow
            .steps(&EPSS_7, |_, time| GUIDED.contains(&time), CFG_SCALE),
    }
}

const CHUNK: usize = SAMPLE_RATE as usize / 10;
const SILENCE: f32 = 1e-3;
const TAIL: usize = 3 * CHUNK;

/// Streams a fragment's audio while trimming its trailing silence like `trim_trailing_silence`:
/// silent chunks are held until a loud one follows; at the end at most `TAIL` of them remain
/// and the partial last chunk is dropped.
#[derive(Default)]
struct SilenceGate {
    chunk: Vec<f32>,
    held: Vec<f32>,
}

impl SilenceGate {
    fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        for &sample in samples {
            self.chunk.push(sample);
            if self.chunk.len() < CHUNK {
                continue;
            }
            let loud =
                self.chunk.iter().map(|sample| sample.abs()).sum::<f32>() / CHUNK as f32 > SILENCE;
            if loud {
                out.append(&mut self.held);
                out.append(&mut self.chunk);
            } else {
                self.held.append(&mut self.chunk);
            }
        }
    }

    fn finish(&mut self, out: &mut Vec<f32>) {
        let keep = self.held.len().min(TAIL);
        out.extend_from_slice(&self.held[..keep]);
        *self = Self::default();
    }
}
