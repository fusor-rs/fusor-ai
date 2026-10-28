//! A request: `start` takes the text to speak, its language and the voice clip (24 kHz samples)
//! to speak it in, with the clip's transcript when it is given; then each `step` does one stage
//! — the clip through the speaker encoder, with a transcript through the speech tokenizer's
//! encoder, the prompt through the talker, one frame of speech, the speech's likeness to the
//! voice — so that a host streams the samples (`speech`) and can stop between steps. The stages
//! also run one by one, on given inputs, for checks against the reference.
//!
//! The last voice's embedding and frames are kept, so the next request with the same clip skips
//! those stages.

use blocks::lm::{Positions, Prediction as Logits, Reader, Rows};
use bundle::{Error, Stages as _, Store};
use serde::{Deserialize, Serialize};
use team::Team;

use crate::codec::decoder::{Decoder, Stream};
use crate::codec::encoder::{Encoded, Encoder};
use crate::model::{
    CODEC_EOS, FRAME_SAMPLES, Frame, HIDDEN, Language, MAX_REFERENCE_SECONDS,
    MIN_REFERENCE_SECONDS, SAMPLE_RATE, TALKER, Talker,
};
use crate::pipeline::Stage;
use crate::predictor::Predictor;
use crate::prompt::{self, Conditioning, Prompt, Reference, Speech};
use crate::sampling::{self, Sampler};
use crate::speaker::Speaker;

/// The most positions the talker holds: the prompt and every frame.
pub const MAX_ROWS: usize = 2048;
/// The most frames a request speaks, 96 s.
pub const MAX_FRAMES: usize = 1200;
/// Prompt rows per call of the talker's stack.
const PREFILL_ROWS: usize = 256;
/// Prompt rows before the text's first and the voice's frames: the role, the language tags,
/// the speaker and the padding.
const PREFIX_ROWS: usize = 9;

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub text: String,
    pub language: Language,
    /// What the clip says: the voice is cloned from the clip's frames and this text, or from its
    /// speaker embedding alone when empty.
    #[serde(default)]
    pub reference_text: String,
    pub seed: u64,
    /// At most `MAX_FRAMES`, its default, and the positions the prompt leaves.
    pub max_frames: Option<usize>,
}

/// How the voice is cloned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// From the clip's frames and transcript.
    Icl,
    /// From the clip's speaker embedding alone.
    Xvector,
}

/// How a request is read: the mode, the clip's frames in it, the prompt's rows and the most
/// frames it speaks.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub mode: Mode,
    pub reference_frames: usize,
    pub prompt_rows: usize,
    pub max_frames: usize,
}

/// One step done: the frames spoken so far (their samples wait in `Engine::speech`), and, when
/// the speech is done, its likeness to the voice (cosine of their speaker embeddings).
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub frames: usize,
    pub done: bool,
    pub similarity: Option<f32>,
}

/// The last row's state: its final-normed hidden state and its logits.
pub struct Prediction {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

/// A voice clip and what the model made of it so far.
#[derive(Default)]
struct Voice {
    samples: Vec<f32>,
    embedding: Option<Vec<f32>>,
    codes: Option<Vec<Frame>>,
}

pub struct Engine {
    talker: Talker,
    predictor: Predictor,
    speaker: Speaker,
    encoder: Encoder,
    decoder: Decoder,
    team: Team,
    reader: Reader,
    stream: Stream,
    voice: Voice,
    /// The samples spoken and not yet taken.
    speech: Vec<f32>,
    job: Option<Job>,
}

struct Job {
    stage: Stage,
    speech: Speech,
    language: Language,
    reference: Option<Vec<u32>>,
    prompt_rows: usize,
    max_frames: usize,
    talker_sampler: Sampler,
    predictor_sampler: Sampler,
    prompt: Option<Prompt>,
    /// The first code of every frame spoken.
    drawn: Vec<u32>,
    frames: Vec<Frame>,
    /// Every sample spoken.
    samples: Vec<f32>,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let team = Team::global();
        let threads = team.threads();
        let decoder = Decoder::load(store)?;
        Ok(Self {
            stream: Stream::new(&decoder, threads),
            reader: Reader::new(&TALKER, MAX_ROWS, PREFILL_ROWS, threads),
            predictor: Predictor::load(store, threads)?,
            speaker: Speaker::load(store)?,
            encoder: Encoder::load(store, threads)?,
            talker: Talker::load(store)?,
            decoder,
            team,
            voice: Voice::default(),
            speech: Vec::new(),
            job: None,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    /// Stops the request; the voice's embedding and frames stay.
    fn cancel(&mut self) {
        self.job = None;
    }
}

impl Engine {
    /// The ids of `text` as the speaking turn holds them.
    ///
    /// # Errors
    /// When the tokenizer fails.
    pub fn speech_ids(&self, text: &str) -> Result<Speech, Error> {
        prompt::speech_ids(&self.talker.tokenizer, text)
    }

    /// The ids of a transcript as the reference turn holds them.
    ///
    /// # Errors
    /// When the tokenizer fails.
    pub fn reference_ids(&self, transcript: &str) -> Result<Vec<u32>, Error> {
        prompt::reference_ids(&self.talker.tokenizer, transcript)
    }

    /// The speaker encoder's log-mel features of a clip, `frames × mels`.
    pub fn speaker_features(&self, samples: &[f32]) -> Vec<f32> {
        self.speaker.features(samples)
    }

    /// The speaker encoder's embedding of `features` (`speaker_features`' output).
    pub fn speaker_embedding(&self, features: &[f32]) -> Vec<f32> {
        self.speaker.embed(&self.team, features)
    }

    /// The frames of a clip of whole frames.
    ///
    /// # Panics
    /// When the clip is not whole frames, or longer than `MAX_REFERENCE_SECONDS`.
    pub fn encode(&mut self, samples: &[f32]) -> Encoded {
        self.encoder.encode(&self.team, samples)
    }

    pub fn prompt(&self, conditioning: &Conditioning) -> Prompt {
        Prompt::new(&self.talker, &self.predictor, conditioning)
    }

    /// The prompt through the talker from position 0: the last row's prediction.
    pub fn prefill(&mut self, prompt: &Prompt) -> Prediction {
        let rows = Rows {
            embeddings: &prompt.rows,
            first: 0,
            positions: Positions::Sequential,
        };
        self.reader.read(&self.team, &self.talker.lm, &rows);
        self.prediction()
    }

    /// After `prefill`, the rows `frames` give the talker read in order: the prediction after
    /// each.
    pub fn force(&mut self, prompt: &Prompt, frames: &[Frame]) -> Vec<Prediction> {
        let first = prompt.rows.len() / HIDDEN;
        let mut predictions = Vec::with_capacity(frames.len());
        for (index, frame) in frames.iter().enumerate() {
            self.read_frame(prompt, first + index, (index, frame));
            predictions.push(self.prediction());
        }
        predictions
    }

    /// The logits of the code predictor's 15 passes over `frame` from the talker's final-normed
    /// state `hidden`, `15 × 2048`.
    pub fn predictor_logits(&mut self, hidden: &[f32], frame: &Frame) -> Vec<f32> {
        let model = (&self.team, &self.talker.lm);
        self.predictor.forced(model, hidden, frame)
    }

    /// The samples of `frames` decoded after the reference clip's `reference` frames, one frame
    /// at a time as a request decodes them.
    pub fn decode(&mut self, reference: &[Frame], frames: &[Frame]) -> Vec<f32> {
        self.stream.reset();
        self.decoder.warm(&self.team, &mut self.stream, reference);
        let mut samples = Vec::new();
        for frame in frames {
            let call = std::slice::from_ref(frame);
            self.decoder
                .decode(&self.team, &mut self.stream, call, &mut samples);
        }
        samples
    }

    /// The samples spoken since the last call, 24 kHz.
    pub fn speech(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.speech)
    }

    fn prediction(&self) -> Prediction {
        let Logits { last, logits, .. } = &self.reader.prediction;
        Prediction {
            hidden: last.read().to_vec(),
            logits: logits.read().to_vec(),
        }
    }

    /// The row the talker reads after `frame`, the `index`th spoken: the frame's embeddings
    /// summed with the next text row, or the padding when the text is out.
    fn frame_input(&self, prompt: &Prompt, (index, frame): (usize, &Frame)) -> Vec<f32> {
        let mut row = self.predictor.frame_row(&self.talker.lm, frame);
        let text = prompt
            .trailing
            .get(index * HIDDEN..(index + 1) * HIDDEN)
            .unwrap_or(&prompt.pad);
        for (value, added) in row.iter_mut().zip(text) {
            *value += added;
        }
        row
    }

    /// Reads the row after `frame` at `position`.
    fn read_frame(&mut self, prompt: &Prompt, position: usize, frame: (usize, &Frame)) {
        let embeddings = self.frame_input(prompt, frame);
        let rows = Rows {
            embeddings: &embeddings,
            first: position,
            positions: Positions::Sequential,
        };
        self.reader.read(&self.team, &self.talker.lm, &rows);
    }

    /// Starts speaking `request.text` in the voice of `reference`, 24 kHz samples; frames
    /// beyond the last whole one are left out.
    ///
    /// # Errors
    /// When the text is empty, or the clip is shorter than `MIN_REFERENCE_SECONDS` or longer than
    /// `MAX_REFERENCE_SECONDS`.
    pub fn start(&mut self, request: &Request, reference: &[f32]) -> Result<Plan, Error> {
        bundle::Engine::cancel(self);
        if request.text.trim().is_empty() {
            return Err(Error::Request("the text to speak is empty".into()));
        }
        let clip = whole_frames(reference)?;
        let mode = if request.reference_text.trim().is_empty() {
            Mode::Xvector
        } else {
            Mode::Icl
        };
        let reference_frames = match mode {
            Mode::Icl => clip.len() / FRAME_SAMPLES,
            Mode::Xvector => 0,
        };
        let prompt_rows = PREFIX_ROWS + 1 + reference_frames;
        let wanted = request
            .max_frames
            .unwrap_or(MAX_FRAMES)
            .clamp(1, MAX_FRAMES);
        let plan = Plan {
            mode,
            reference_frames,
            prompt_rows,
            max_frames: wanted.min(MAX_ROWS - prompt_rows),
        };
        self.job = Some(self.new_job(request, &plan)?);
        if self.voice.samples != clip {
            self.voice = Voice {
                samples: clip.to_vec(),
                embedding: None,
                codes: None,
            };
        }
        self.speech.clear();
        Ok(plan)
    }

    fn new_job(&self, request: &Request, plan: &Plan) -> Result<Job, Error> {
        let reference = match plan.mode {
            Mode::Icl => Some(self.reference_ids(request.reference_text.trim())?),
            Mode::Xvector => None,
        };
        Ok(Job {
            stage: Stage::Embed,
            speech: self.speech_ids(request.text.trim())?,
            language: request.language,
            reference,
            prompt_rows: plan.prompt_rows,
            max_frames: plan.max_frames,
            talker_sampler: Sampler::new(sampling::SAMPLING, request.seed),
            predictor_sampler: Sampler::new(sampling::SAMPLING, request.seed.wrapping_add(1)),
            prompt: None,
            drawn: Vec::new(),
            frames: Vec::new(),
            samples: Vec::new(),
        })
    }

    /// Runs the next stage: the clip's embedding, its frames, the prompt, one frame of speech,
    /// or the speech's likeness to the voice.
    ///
    /// # Errors
    /// When no request is in progress.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no request in progress".into()))?;
        let stage = job.stage;
        let mut similarity = None;
        match stage {
            Stage::Embed => self.embed_voice(&mut job),
            Stage::Encode => self.encode_voice(&mut job),
            Stage::Prefill => self.prefill_job(&mut job),
            Stage::Speak => self.speak(&mut job),
            Stage::Match => similarity = Some(self.likeness(&job)),
        }
        let progress = Progress {
            stage: stage.name(),
            frames: job.frames.len(),
            done: similarity.is_some(),
            similarity,
        };
        if !progress.done {
            self.job = Some(job);
        }
        Ok(progress)
    }

    fn embed_voice(&mut self, job: &mut Job) {
        let voice = &mut self.voice;
        if voice.embedding.is_none() {
            let features = self.speaker.features(&voice.samples);
            voice.embedding = Some(self.speaker.embed(&self.team, &features));
        }
        job.stage = if job.reference.is_some() {
            Stage::Encode
        } else {
            Stage::Prefill
        };
    }

    fn encode_voice(&mut self, job: &mut Job) {
        let voice = &mut self.voice;
        if voice.codes.is_none() {
            voice.codes = Some(self.encoder.encode(&self.team, &voice.samples).codes);
        }
        job.stage = Stage::Prefill;
    }

    /// The prompt through the talker, and a transcript's frames through the decoder; the first
    /// frame's state is the last row's.
    fn prefill_job(&mut self, job: &mut Job) {
        let embedding = self.voice.embedding.as_deref();
        let codes = self.voice.codes.as_deref();
        let reference = job.reference.as_deref().zip(codes);
        let conditioning = Conditioning {
            language: job.language,
            speech: &job.speech,
            speaker: embedding.expect("the embed stage ran"),
            reference: reference.map(|(ids, frames)| Reference { ids, frames }),
        };
        let prompt = Prompt::new(&self.talker, &self.predictor, &conditioning);
        let rows = Rows {
            embeddings: &prompt.rows,
            first: 0,
            positions: Positions::Sequential,
        };
        self.reader.read(&self.team, &self.talker.lm, &rows);
        self.stream.reset();
        if let Some(frames) = codes.filter(|_| job.reference.is_some()) {
            self.decoder.warm(&self.team, &mut self.stream, frames);
        }
        job.prompt = Some(prompt);
        job.stage = Stage::Speak;
    }

    /// One frame: the talker's first code from the last row's logits, the code predictor's other
    /// 15, the frame's samples, and the talker's next row. Speech ends with the end-of-speech
    /// code or the request's last frame.
    fn speak(&mut self, job: &mut Job) {
        let Logits {
            last,
            logits: scores,
            ..
        } = &self.reader.prediction;
        let mut logits = scores.read().to_vec();
        sampling::prepare(&mut logits, &job.drawn);
        let code0 = job.talker_sampler.draw(&mut logits);
        if code0 == CODEC_EOS {
            job.stage = Stage::Match;
            return;
        }
        let model = (&self.team, &self.talker.lm);
        let frame = self
            .predictor
            .frame(model, last.read(), code0, &mut job.predictor_sampler);
        let spoken = job.samples.len();
        let call = std::slice::from_ref(&frame);
        self.decoder
            .decode(&self.team, &mut self.stream, call, &mut job.samples);
        self.speech.extend_from_slice(&job.samples[spoken..]);
        let index = job.frames.len();
        job.drawn.push(code0);
        job.frames.push(frame);
        if job.frames.len() == job.max_frames {
            job.stage = Stage::Match;
            return;
        }
        let prompt = job.prompt.as_ref().expect("the prefill stage ran");
        self.read_frame(prompt, job.prompt_rows + index, (index, &frame));
    }

    /// The cosine of the speech's speaker embedding and the voice's.
    fn likeness(&self, job: &Job) -> f32 {
        let features = self.speaker.features(&job.samples);
        let spoken = self.speaker.embed(&self.team, &features);
        let voice = self
            .voice
            .embedding
            .as_deref()
            .expect("the embed stage ran");
        let dot: f32 = spoken
            .iter()
            .zip(voice)
            .map(|(first, second)| first * second)
            .sum();
        let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
        dot / (norm(&spoken) * norm(voice))
    }
}

/// `reference` cut to whole frames.
fn whole_frames(reference: &[f32]) -> Result<&[f32], Error> {
    let rate = SAMPLE_RATE as usize;
    let lengths = MIN_REFERENCE_SECONDS * rate..=MAX_REFERENCE_SECONDS * rate;
    if !lengths.contains(&reference.len()) {
        return Err(Error::Request(format!(
            "the voice clip is {:.1} s; it takes {MIN_REFERENCE_SECONDS} to \
             {MAX_REFERENCE_SECONDS} s",
            reference.len() as f32 / SAMPLE_RATE as f32
        )));
    }
    Ok(&reference[..reference.len() / FRAME_SAMPLES * FRAME_SAMPLES])
}
