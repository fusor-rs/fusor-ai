//! The synthesis loop: one `step` per 80 ms frame.
//!
//! A request's text is cut into chunks (`prompt`). The voice is run through the flow LM once and
//! stays at the start of its KV cache; each chunk's text follows it, then each frame samples a
//! latent from the previous one and decodes it to audio at once. A chunk ends `frames_after_eos`
//! frames after the first frame whose end-of-speech logit passes `EOS_THRESHOLD` (that frame and
//! those after it are kept), or at its frame budget.

use serde::{Deserialize, Serialize};

use bundle::{Store, Tokenizer};
use kernels::QuantizedRows;
use kernels::rng::Rng;
use team::Team;

use crate::Error;
pub use crate::lm::FlowStep;
use crate::lm::LmState;
use crate::mimi::MimiState;
use crate::model::{FRAME_SAMPLES, HIDDEN, LATENT, Model, VOICE_FRAMES};
use crate::prompt::{self, Chunk, MAX_CHUNK_TOKENS, Prompter, VOICES};

/// The noise variance of sampling (the reference runtime's default; the config's `temp` is not
/// used).
const DEFAULT_TEMPERATURE: f32 = 0.3;
pub const EOS_THRESHOLD: f32 = 4.0;
const DEFAULT_SEED: u64 = 4_242_424_242_424_242;

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub text: String,
    /// An id from `prompt::VOICES`.
    pub voice: String,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
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
    lm: LmState,
    mimi: MimiState,
    /// The voice whose embeddings fill the first `VOICE_FRAMES` positions of the LM cache.
    primed: Option<usize>,
    job: Option<Job>,
    audio: Vec<f32>,
}

struct Job {
    chunks: Vec<Chunk>,
    chunk: usize,
    voice: usize,
    frame: Option<Frame>,
    rng: Rng,
    noise_scale: f32,
}

/// The state between two frames of a chunk.
struct Frame {
    index: usize,
    budget: usize,
    previous: Option<Vec<f32>>,
    /// Frames still to emit once the end of speech was seen.
    remaining: Option<usize>,
}

impl bundle::Engine for Engine {
    /// Builds the engine from a complete store, with the first voice already run through the LM
    /// so that a request starts speaking after its text alone.
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let prompter = Prompter::new(Tokenizer::take(store)?);
        let team = Team::global();
        let threads = team.threads();
        let capacity = VOICE_FRAMES + MAX_CHUNK_TOKENS + prompt::frame_budget(MAX_CHUNK_TOKENS);
        let mut engine = Self {
            lm: LmState::new(&model.lm, capacity, threads),
            mimi: MimiState::new(&model.mimi, threads),
            model,
            prompter,
            team,
            primed: None,
            job: None,
            audio: Vec::new(),
        };
        engine.prime(0);
        Ok(engine)
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
        let Request {
            text,
            voice,
            seed,
            temperature,
        } = request;
        let voice = voice_index(&voice)?;
        let temperature = temperature.unwrap_or(DEFAULT_TEMPERATURE);
        if !(temperature.is_finite() && temperature >= 0.0) {
            return Err(Error::Request(format!("temperature {temperature}")));
        }
        let chunks = self.prompter.chunks(&text)?;
        if chunks.is_empty() {
            return Err(Error::Request("the text has nothing to say".into()));
        }
        let fragments = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        self.audio.clear();
        self.job = Some(Job {
            chunks,
            chunk: 0,
            voice,
            frame: None,
            rng: Rng::new(seed.unwrap_or(DEFAULT_SEED)),
            noise_scale: temperature.sqrt(),
        });
        Ok(Plan { fragments })
    }

    /// Samples produced since the last call.
    pub fn take_audio(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.audio)
    }

    /// Generates one frame (prompting first when a chunk starts).
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no synthesis in progress".into()))?;
        let mut frame = match job.frame.take() {
            Some(frame) => frame,
            None => self.begin_chunk(&job),
        };
        let mut noise = vec![0.0; LATENT];
        job.rng.fill_normal(&mut noise);
        noise.iter_mut().for_each(|value| *value *= job.noise_scale);
        let FlowStep { latent, eos_logit } = self.flow_step(frame.previous.as_deref(), &noise);
        self.model
            .mimi
            .decode(&self.team, &mut self.mimi, &latent, &mut self.audio);
        let progress = Progress {
            fragment: job.chunk,
            fragments: job.chunks.len(),
            frame: frame.index,
            done: false,
        };
        let chunk = &job.chunks[job.chunk];
        let ended = frame.end_of_speech(eos_logit, chunk.frames_after_eos);
        if !ended && frame.index + 1 < frame.budget {
            frame.index += 1;
            frame.previous = Some(latent);
            job.frame = Some(frame);
            self.job = Some(job);
            return Ok(progress);
        }
        job.chunk += 1;
        let done = job.chunk == job.chunks.len();
        if !done {
            self.job = Some(job);
        }
        Ok(Progress { done, ..progress })
    }

    fn begin_chunk(&mut self, job: &Job) -> Frame {
        let chunk = &job.chunks[job.chunk];
        self.prime(job.voice);
        self.prompt_text(&chunk.tokens);
        self.mimi.reset();
        Frame {
            index: 0,
            budget: chunk.frame_budget,
            previous: None,
            remaining: None,
        }
    }

    /// Runs the voice through the LM into the first `VOICE_FRAMES` cache positions, unless it is
    /// there already.
    fn prime(&mut self, voice: usize) {
        if self.primed == Some(voice) {
            return;
        }
        let embedded = self.model.voice(voice).matrix();
        self.model
            .lm
            .prompt(&self.team, &mut self.lm, 0..VOICE_FRAMES, |frame, row| {
                embedded.dequantize_row(frame, row);
            });
        self.primed = Some(voice);
    }

    /// Runs `tokens` through the LM after the voice; the next `flow_step` follows them.
    fn prompt_text(&mut self, tokens: &[u32]) {
        let embeddings = self.model.lm.embeddings();
        let positions = VOICE_FRAMES..VOICE_FRAMES + tokens.len();
        self.model
            .lm
            .prompt(&self.team, &mut self.lm, positions, |index, row| {
                embeddings.dequantize_row(tokens[index] as usize, row);
            });
    }

    /// The chunks a request's text becomes.
    pub fn chunks(&self, text: &str) -> Result<Vec<Chunk>, Error> {
        self.prompter.chunks(text)
    }

    /// Voice `voice`'s embedded latents, `[VOICE_FRAMES, HIDDEN]`.
    pub fn voice_embedding(&self, voice: &str) -> Result<Vec<f32>, Error> {
        Ok(self.model.voice(voice_index(voice)?).dequantize())
    }

    /// The summed conditions added to every latent's input projection, `[HIDDEN]`: the
    /// projection of a zero latent, as the compiler folds them into its bias.
    pub fn conditions(&self) -> Vec<f32> {
        let mut conditions = vec![0.0; HIDDEN];
        let input = &self.model.lm.input;
        input.apply(
            &[0.0; LATENT],
            &mut QuantizedRows::default(),
            &mut conditions,
        );
        conditions
    }

    /// The text embeddings of `tokens`, `[tokens, HIDDEN]`.
    pub fn text_embeddings(&self, tokens: &[u32]) -> Vec<f32> {
        let embeddings = self.model.lm.embeddings();
        let mut out = vec![0.0; tokens.len() * HIDDEN];
        for (token, row) in tokens.iter().zip(out.as_chunks_mut::<HIDDEN>().0) {
            embeddings.dequantize_row(*token as usize, row);
        }
        out
    }

    /// Runs voice `voice` and `tokens` into the LM state as a chunk does (cancels any request).
    pub fn prompt(&mut self, voice: &str, tokens: &[u32]) -> Result<(), Error> {
        let voice = voice_index(voice)?;
        if tokens.len() > MAX_CHUNK_TOKENS {
            return Err(Error::Request(format!(
                "{} tokens, more than a chunk's {MAX_CHUNK_TOKENS}",
                tokens.len()
            )));
        }
        self.job = None;
        self.prime(voice);
        self.prompt_text(tokens);
        Ok(())
    }

    /// One frame from `previous` (BOS for `None`, the first frame) and `noise` (`LATENT`
    /// values, already scaled by √temperature), at the position after the last.
    pub fn flow_step(&mut self, previous: Option<&[f32]>, noise: &[f32]) -> FlowStep {
        let input = previous.unwrap_or(self.model.lm.bos.values());
        self.model.lm.step(&self.team, &mut self.lm, input, noise)
    }

    /// Decodes a latent sequence (`frames × LATENT`) from a fresh decoder state.
    pub fn decode(&mut self, latents: &[f32]) -> Vec<f32> {
        self.mimi.reset();
        let mut samples = Vec::with_capacity(latents.len() / LATENT * FRAME_SAMPLES);
        self.model
            .mimi
            .decode(&self.team, &mut self.mimi, latents, &mut samples);
        samples
    }
}

impl Frame {
    /// Whether the chunk ends after this frame, given its end-of-speech logit.
    fn end_of_speech(&mut self, eos_logit: f32, frames_after_eos: usize) -> bool {
        if eos_logit > EOS_THRESHOLD && self.remaining.is_none() {
            self.remaining = Some(frames_after_eos);
        }
        match self.remaining {
            Some(0) => true,
            Some(left) => {
                self.remaining = Some(left - 1);
                false
            }
            None => false,
        }
    }
}

fn voice_index(voice: &str) -> Result<usize, Error> {
    VOICES
        .iter()
        .position(|(id, _)| *id == voice)
        .ok_or_else(|| Error::Request(format!("no voice {voice:?}")))
}
