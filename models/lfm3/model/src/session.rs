//! A conversation: `start` takes the user's next message, its images and its clips, cuts the
//! images into patches and builds the conversation's prompt; then each `step` does one stage —
//! an image through the vision tower, a clip through the audio encoder, the prompt through the
//! language model, or one more row of the reply: a token, or a spoken frame through the codec —
//! so that a host streams the reply, its text and its samples (`speech`), and can stop between
//! steps. Each turn reads its whole prompt from its first row, or on from the end of the longest
//! kept prompt it begins with (`keep`): the template writes a reply back as its text alone,
//! without its reasoning or its speech, and the window and conv layers keep only their last rows;
//! the conversation keeps its images' and clips' tokens for it. The stages also run one by one, on
//! given inputs, for checks against the reference.

use ::media::image::Pixels;
use blocks::codec;
use blocks::hybrid::Rows;
use blocks::interleaved::{Prefix, State};
use blocks::lm;
use blocks::rotary::Position;
use blocks::transformer::{Buffers, Mask};
use blocks::vit::merger;
use blocks::vit::preprocessing::Patches;
use blocks::vit::{Grid, tower};
use bundle::{Error, Stages as _, Store};
use serde::{Deserialize, Serialize};
use team::{SharedBuf, Team};

use crate::audio::{self, CODEBOOKS, END_CODE, Frame, Hearing, Voice};
use crate::media::{self, IMAGE};
use crate::model::{
    CONFIG, Config, HIDDEN, LANGUAGE, Model, NORM_EPS, PROJECTOR, TOWER, VOCABULARY, Variant,
    Vision,
};
use crate::output::Output;
use crate::pipeline::Stage;
use crate::prompt::{
    self, AUDIO_OUT_TOKEN, AUDIO_TOKEN, Chat, IMAGE_TOKEN, Mode, Opening, Replied, Turn, reasoned,
    written,
};
use crate::sampling::{self, Constraint, FrameSampler, Sampler, Sampling};

/// The longest conversation, in rows: every turn's text and reply.
pub const MAX_ROWS: usize = 8_192;
/// The longest reply, in rows: tokens and spoken frames.
pub const MAX_TOKENS: usize = 4_096;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 512;
/// The most prompts' first rows kept at once, each with its attention layers' keys and values.
const KEPT_PROMPTS: usize = 4;
const NO_REPLY: &str = "no reply in progress";

/// The user's next message, and how to reply to it.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub text: String,
    /// The conversation's system text, the same on each of its turns.
    #[serde(default)]
    pub system: String,
    /// The tools the system block lists, each its JSON as the template's `tool_to_json` writes
    /// it; none lists none.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Whether a text reply reasons first, in a thinking block.
    #[serde(default)]
    pub thinking: bool,
    /// Text, a transcript, speech, or text and speech in turns.
    #[serde(default)]
    pub mode: Mode,
    /// At most `MAX_TOKENS`, its default, and the rows the conversation has left.
    pub max_tokens: Option<usize>,
    /// The model card's settings (`sampling::preset`) when absent.
    pub sampling: Option<Sampling>,
    pub seed: u64,
}

/// How a turn is read: each image's patch grid (rows, columns), each clip's tokens, the prompt's
/// rows, the longest reply and its sampling.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub images: Vec<[usize; 2]>,
    pub clips: Vec<usize>,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub sampling: Sampling,
}

/// One step done: the reply so far, its reasoning apart when the turn thinks, and its spoken
/// frames (their samples wait in `Engine::speech`).
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub reasoning: Option<String>,
    pub answer: String,
    pub tokens: usize,
    pub frames: usize,
    pub done: bool,
}

/// The last row's state: its final-normed hidden state and its logits.
pub struct Prediction {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct Engine {
    model: Model,
    variant: Variant,
    sight: Option<Sight>,
    hearing: Option<Hearing>,
    voice: Option<Voice>,
    team: Team,
    buffers: Buffers,
    state: State,
    prediction: lm::Prediction,
    history: Vec<Replied>,
    /// The tokens of the history's images and clips, in prompt order, `tokens × hidden`.
    media: Vec<f32>,
    /// The 24 kHz samples spoken and not yet taken.
    speech: Vec<f32>,
    /// Prompts' first rows, kept for later prompts that begin with them, the last used first.
    kept: Vec<Kept>,
    job: Option<Job>,
}

/// A prompt's first rows, and the state after them.
struct Kept {
    ids: Vec<u32>,
    prefix: Prefix,
}

/// The vision of LFM3-VL-2B and LFM3-Omni-2B, and its activations.
struct Sight {
    vision: Vision,
    tower: tower::Buffers,
    projector: merger::Buffers,
}

struct Job {
    turn: Turn,
    images: Vec<Patches>,
    clips: Vec<Vec<f32>>,
    /// The turn's images seen and clips heard so far.
    seen: usize,
    heard: usize,
    /// Their tokens, in prompt order, `tokens × hidden`.
    tokens: Vec<f32>,
    prompt: Vec<u32>,
    /// The system block as written, with its tools, and how the reply opens: what `keep` reads
    /// its prefix with.
    system: String,
    opening: Opening,
    /// The prompt's first rows to keep, from `Engine::keep`.
    keep: Option<usize>,
    sampler: Sampler,
    constraint: Option<Box<dyn Constraint>>,
    output: Output,
    speaking: Option<Speaking>,
    /// The reply's rows: its tokens, and `<|audio_out|>` for each frame it spoke.
    reply: Vec<u32>,
    /// The reply's text so far, its markers excluded.
    written: String,
    ended: bool,
    max_tokens: usize,
}

/// What a reply that speaks keeps: its frames' sampler, the codec's stream, its last frame and
/// how many it spoke.
struct Speaking {
    sampler: FrameSampler,
    stream: codec::Stream,
    frame: Frame,
    frames: usize,
}

/// A row of the reply the language model reads.
#[derive(Clone, Copy)]
enum Row<'frame> {
    Token(u32),
    Frame(&'frame Frame),
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let variant = Config::read(&store.take_blob(CONFIG)?)?.model_type;
        let sight = if variant.sees() {
            Some(Sight {
                vision: Vision::load(store)?,
                tower: TOWER.buffers(IMAGE.max_patches(), threads),
                projector: merger::Buffers::new(&PROJECTOR, IMAGE.max_patches()),
            })
        } else {
            None
        };
        let (hearing, voice) = if variant.hears() {
            (
                Some(Hearing::load(store, threads)?),
                Some(Voice::load(store, threads)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            variant,
            sight,
            hearing,
            voice,
            buffers: LANGUAGE.buffers(PREFILL_ROWS, threads),
            state: State::new(&LANGUAGE, MAX_ROWS, PREFILL_ROWS),
            prediction: lm::Prediction::new(HIDDEN, VOCABULARY, threads),
            model,
            team,
            history: Vec::new(),
            media: Vec::new(),
            speech: Vec::new(),
            kept: Vec::new(),
            job: None,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    /// Stops the reply: what it wrote so far is the turn's reply; a turn stopped before its
    /// prompt was read is dropped.
    fn cancel(&mut self) {
        if let Some(job) = self.job.take() {
            self.finish(job);
        }
    }
}

impl Engine {
    /// Each patch's embedding, RMS-normed, `patches × width`.
    ///
    /// # Errors
    /// When the model reads no images.
    pub fn embed(&mut self, patches: &Patches) -> Result<Vec<f32>, Error> {
        let sight = self
            .sight
            .as_ref()
            .ok_or_else(|| self.lacks("reads no images"))?;
        let (tower, buffers) = (&sight.vision.tower, &sight.tower);
        self.team
            .run(|ctx| tower.embed(ctx, buffers, &patches.values));
        let len = patches.grid.patches() * TOWER.hidden;
        Ok(buffers.encoder.hidden.read()[..len].to_vec())
    }

    /// The tower's layers over `embedded` (`embed`'s output).
    ///
    /// # Errors
    /// When the model reads no images.
    pub fn encode(&mut self, embedded: &[f32], grid: Grid) -> Result<Vec<f32>, Error> {
        let missing = self.lacks("reads no images");
        let sight = self.sight.as_mut().ok_or(missing)?;
        sight.tower.encoder.hidden.get_mut()[..embedded.len()].copy_from_slice(embedded);
        let (tower, buffers) = (&sight.vision.tower, &sight.tower);
        self.team.run(|ctx| tower.encode(ctx, buffers, grid));
        Ok(buffers.encoder.hidden.read()[..embedded.len()].to_vec())
    }

    /// An image's tokens from `encoded` (`encode`'s output), `tokens × hidden`.
    ///
    /// # Errors
    /// When the model reads no images.
    pub fn project(&mut self, encoded: &[f32], grid: Grid) -> Result<Vec<f32>, Error> {
        let missing = self.lacks("reads no images");
        let sight = self.sight.as_mut().ok_or(missing)?;
        sight.tower.encoder.hidden.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (projector, tower, buffers) = (&sight.vision.projector, &sight.tower, &sight.projector);
        self.team
            .run(|ctx| projector.forward(ctx, tower, buffers, grid));
        Ok(image_tokens(sight, grid))
    }

    /// A clip's log-mel features, `frames × mels`.
    ///
    /// # Errors
    /// When the model hears no audio, or the clip is too long.
    pub fn audio_features(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let hearing = self
            .hearing
            .as_ref()
            .ok_or_else(|| self.lacks("hears no audio"))?;
        hearing.features(samples)
    }

    /// The audio encoder's tokens of `features` (`audio_features`' output).
    ///
    /// # Errors
    /// When the model hears no audio.
    pub fn encode_audio(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        let missing = self.lacks("hears no audio");
        let hearing = self.hearing.as_mut().ok_or(missing)?;
        Ok(hearing.encode(&self.team, features))
    }

    /// A clip's tokens from `encoded` (`encode_audio`'s output), `tokens × hidden`.
    ///
    /// # Errors
    /// When the model hears no audio.
    pub fn project_audio(&mut self, encoded: &[f32]) -> Result<Vec<f32>, Error> {
        let missing = self.lacks("hears no audio");
        let hearing = self.hearing.as_mut().ok_or(missing)?;
        Ok(hearing.project(&self.team, encoded))
    }

    /// The depth decoder's first pass over a frame from the language model's final-normed
    /// `hidden` state: every codebook's logits, `codebooks × codes`.
    ///
    /// # Errors
    /// When the model does not speak.
    pub fn depth_logits(&self, hidden: &[f32]) -> Result<Vec<f32>, Error> {
        let voice = self
            .voice
            .as_ref()
            .ok_or_else(|| self.lacks("does not speak"))?;
        Ok(voice.logits(&self.team, &shared(hidden)))
    }

    /// A frame drawn from `hidden` as `sampling::voicing` draws it, seeded by `seed`.
    ///
    /// # Errors
    /// When the model does not speak.
    pub fn frame(&mut self, hidden: &[f32], seed: u64) -> Result<Frame, Error> {
        let missing = self.lacks("does not speak");
        let voice = self.voice.as_mut().ok_or(missing)?;
        let mut sampler = FrameSampler::new(sampling::voicing(self.variant), seed);
        Ok(voice.frame(&self.team, &shared(hidden), &mut sampler))
    }

    /// The 24 kHz samples of `frames`, one stream through the codec.
    ///
    /// # Errors
    /// When the model does not speak, or a frame ends speech.
    pub fn speak(&self, frames: &[Frame]) -> Result<Vec<f32>, Error> {
        let voice = self
            .voice
            .as_ref()
            .ok_or_else(|| self.lacks("does not speak"))?;
        if frames.iter().any(audio::ends) {
            return Err(Error::Request(
                "a frame that ends speech has no samples".into(),
            ));
        }
        let mut stream = voice.codec.stream(1, frames.len(), self.team.threads());
        let mut samples = Vec::new();
        for frame in frames {
            stream.decode(&self.team, &voice.codec, frame, &mut samples);
        }
        stream.finish(&mut samples);
        Ok(samples)
    }

    /// An image's patches through the tower and the projector, in one team run: its tokens.
    fn see(&self, patches: &Patches) -> Vec<f32> {
        let sight = self
            .sight
            .as_ref()
            .expect("a turn has images only when the model reads them");
        let (vision, grid) = (&sight.vision, patches.grid);
        self.team.run(|ctx| {
            vision.tower.embed(ctx, &sight.tower, &patches.values);
            vision.tower.encode(ctx, &sight.tower, grid);
            vision
                .projector
                .forward(ctx, &sight.tower, &sight.projector, grid);
        });
        image_tokens(sight, grid)
    }

    /// The prompt asking for the reply to `next`, after `system` and `history`.
    pub fn prompt(
        &self,
        system: &str,
        history: &[Replied],
        next: &Turn,
        opening: Opening,
    ) -> Result<Vec<u32>, Error> {
        let chat = Chat {
            tokenizer: &self.model.tokenizer,
            template: self.variant.template(),
        };
        prompt::prompt(chat, system, history, next, opening)
    }

    /// The text of reply tokens.
    pub fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.model.tokenizer.decode(ids)
    }

    /// The samples spoken since the last call, 24 kHz.
    pub fn speech(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.speech)
    }

    /// Forgets the conversation, to start a new one; the rows `keep` kept stay.
    pub fn reset(&mut self) {
        self.job = None;
        self.history.clear();
        self.media.clear();
        self.speech.clear();
    }

    /// The prompt through the language model (from the end of the longest kept prompt it begins
    /// with, or its first row), its image and clip rows `media` in order: the last row's
    /// prediction.
    pub fn prefill(&mut self, prompt: &[u32], media: &[f32]) -> Prediction {
        self.read_prompt(prompt, media.as_chunks().0.iter(), None);
        self.prediction()
    }

    /// After `prefill`, `reply`'s tokens read in order: the prediction after each.
    pub fn force(&mut self, prompt: &[u32], reply: &[u32]) -> Vec<Prediction> {
        let read = reply.iter().enumerate().map(|(index, &token)| {
            self.write(prompt.len() + index, Row::Token(token));
            self.prediction()
        });
        read.collect()
    }

    fn prediction(&self) -> Prediction {
        Prediction {
            hidden: self.prediction.last.read().to_vec(),
            logits: self.prediction.logits.read().to_vec(),
        }
    }

    /// The error of a request the model cannot serve: it `lacks` an ability.
    fn lacks(&self, ability: &str) -> Error {
        Error::Request(format!("{} {ability}", self.variant.name()))
    }

    /// Whether the model can reply to `message` with these images and clips.
    fn check(&self, message: &Message, images: &[Pixels], clips: &[Vec<f32>]) -> Result<(), Error> {
        if !images.is_empty() && self.sight.is_none() {
            return Err(self.lacks("reads no images"));
        }
        if !clips.is_empty() && self.hearing.is_none() {
            return Err(self.lacks("hears no audio"));
        }
        if message.mode.speaks() && self.voice.is_none() {
            return Err(self.lacks("does not speak"));
        }
        clips.iter().try_for_each(|clip| audio::check(clip))?;
        let sampling = message.sampling.unwrap_or(sampling::preset(self.variant));
        sampling.check()
    }

    /// Starts replying to `message` with RGB or RGBA `images` and 16 kHz `clips`, the
    /// conversation's next turn.
    ///
    /// # Errors
    /// When the model reads no images and there are some, hears no audio and there are clips,
    /// does not speak and the mode does, an image is empty or too narrow, a clip too long, the
    /// conversation would outgrow `MAX_ROWS`, or the sampling is invalid.
    pub fn start(
        &mut self,
        message: &Message,
        images: &[Pixels],
        clips: &[Vec<f32>],
    ) -> Result<Plan, Error> {
        bundle::Engine::cancel(self);
        self.check(message, images, clips)?;
        let images = images
            .iter()
            .map(media::patches)
            .collect::<Result<Vec<_>, _>>()?;
        let turn = Turn {
            images: images
                .iter()
                .map(|image| IMAGE.tokens(image.grid.patches()))
                .collect(),
            audio: clips.iter().map(|clip| audio::tokens(clip.len())).collect(),
            text: message.text.clone(),
        };
        let opening = Opening {
            mode: message.mode,
            thinking: message.thinking,
        };
        let system = prompt::system(&message.system, &message.tools);
        let prompt = self.prompt(&system, &self.history, &turn, opening)?;
        let rows = prompt.len();
        if rows >= MAX_ROWS {
            return Err(Error::Request(format!(
                "the conversation has {rows} tokens, the most this runtime holds is {MAX_ROWS}; \
                 start a new one"
            )));
        }
        let sampling = message.sampling.unwrap_or(sampling::preset(self.variant));
        let max_tokens = message.max_tokens.unwrap_or(MAX_TOKENS);
        let plan = Plan {
            images: images
                .iter()
                .map(|image| [image.grid.rows, image.grid.columns])
                .collect(),
            clips: turn.audio.clone(),
            prompt_tokens: rows,
            max_tokens: max_tokens.clamp(1, MAX_TOKENS).min(MAX_ROWS - rows),
            sampling,
        };
        self.job = Some(Job {
            speaking: self.speaking(message, plan.max_tokens),
            turn,
            images,
            clips: clips.to_vec(),
            seen: 0,
            heard: 0,
            tokens: Vec::new(),
            prompt,
            system,
            opening,
            keep: None,
            sampler: Sampler::new(sampling, message.seed),
            constraint: None,
            output: Output::of(message.mode),
            reply: Vec::new(),
            written: String::new(),
            ended: false,
            max_tokens: plan.max_tokens,
        });
        Ok(plan)
    }

    /// What a reply in `message`'s mode keeps to speak, when it may.
    fn speaking(&self, message: &Message, max_frames: usize) -> Option<Speaking> {
        let voice = self.voice.as_ref().filter(|_| message.mode.speaks())?;
        Some(Speaking {
            sampler: FrameSampler::new(sampling::voicing(self.variant), message.seed),
            stream: voice.codec.stream(1, max_frames, self.team.threads()),
            frame: [END_CODE; CODEBOOKS],
            frames: 0,
        })
    }

    /// Runs the next stage: the next image or clip, the prompt, or one more row.
    ///
    /// # Errors
    /// When no reply is in progress, or the reply doesn't detokenize.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request(NO_REPLY.into()))?;
        if let Some(stage) = self.take_in(&mut job)? {
            let progress = progress(stage, &job, false);
            self.job = Some(job);
            return Ok(progress);
        }
        let read = if job.reply.is_empty() {
            let media = std::mem::take(&mut self.media);
            let turn = job.tokens.as_chunks().0;
            let rows = media.as_chunks().0.iter().chain(turn);
            self.read_prompt(&job.prompt, rows, job.keep);
            self.media = media;
            Stage::Prefill
        } else {
            self.read_last(&job);
            Stage::Decode
        };
        let stage = if job.output.speaks() {
            self.speak_frame(&mut job);
            Stage::Speak
        } else {
            self.write_token(&mut job)?;
            read
        };
        let done = job.ended || job.reply.len() >= job.max_tokens;
        let progress = progress(stage, &job, done);
        if done {
            self.finish(job);
        } else {
            self.job = Some(job);
        }
        Ok(progress)
    }

    /// Holds the reply `start` began to `constraint`, from its first token.
    ///
    /// # Errors
    /// When no reply is in progress.
    pub fn constrain(&mut self, constraint: Box<dyn Constraint>) -> Result<(), Error> {
        let job = self
            .job
            .as_mut()
            .ok_or_else(|| Error::Request(NO_REPLY.into()))?;
        job.constraint = Some(constraint);
        Ok(())
    }

    /// Keeps the state after the rows of the prompt `start` began that its system block and
    /// `prefix`, the start of its message's text, read as, for every later prompt that begins
    /// with the same rows: such a prompt is read on from there. Only a first turn without images
    /// or clips is kept; any other is read as it was.
    ///
    /// # Errors
    /// When no reply is in progress, or `prefix` doesn't tokenize.
    pub fn keep(&mut self, prefix: &str) -> Result<(), Error> {
        let job = self
            .job
            .as_ref()
            .ok_or_else(|| Error::Request(NO_REPLY.into()))?;
        let alone = Turn {
            text: prefix.to_owned(),
            ..Turn::default()
        };
        let read_alone = self.prompt(&job.system, &[], &alone, job.opening)?;
        let rows = read_alone
            .iter()
            .zip(&job.prompt)
            .take_while(|(alone, read)| alone == read)
            .count();
        let first_turn =
            self.history.is_empty() && job.turn.images.is_empty() && job.turn.audio.is_empty();
        let keepable = first_turn && rows < job.prompt.len();
        let job = self
            .job
            .as_mut()
            .expect("a reply is in progress, as checked above");
        job.keep = keepable.then_some(rows);
        Ok(())
    }

    /// The turn's next image or clip into its tokens, while one is left: its stage.
    fn take_in(&mut self, job: &mut Job) -> Result<Option<Stage>, Error> {
        if let Some(patches) = job.images.get(job.seen) {
            job.tokens.extend(self.see(patches));
            job.seen += 1;
            return Ok(Some(Stage::See));
        }
        let Some(clip) = job.clips.get(job.heard) else {
            return Ok(None);
        };
        let hearing = self
            .hearing
            .as_mut()
            .expect("a turn has clips only when the model hears them");
        job.tokens.extend(hearing.hear(&self.team, clip)?);
        job.heard += 1;
        Ok(Some(Stage::Hear))
    }

    /// The reply's next frame drawn from the last row, its samples through the codec.
    fn speak_frame(&mut self, job: &mut Job) {
        let speaking = job
            .speaking
            .as_mut()
            .expect("a reply speaks only in a mode that keeps a voice");
        let voice = self
            .voice
            .as_mut()
            .expect("a reply speaks only with a voice");
        let frame = voice.frame(&self.team, &self.prediction.last, &mut speaking.sampler);
        job.output.spoke(&frame);
        speaking.frame = if audio::ends(&frame) {
            [END_CODE; CODEBOOKS]
        } else {
            speaking.sampler.keep(&frame);
            speaking
                .stream
                .decode(&self.team, &voice.codec, &frame, &mut self.speech);
            speaking.frames += 1;
            frame
        };
        job.reply.push(AUDIO_OUT_TOKEN);
    }

    /// The reply's next token drawn from the last row's logits.
    fn write_token(&mut self, job: &mut Job) -> Result<(), Error> {
        let logits = self.prediction.logits.get_mut();
        job.output.suppress(logits);
        if let Some(constraint) = &mut job.constraint {
            constraint.mask(logits);
        }
        let token = job.sampler.draw(logits);
        if let Some(constraint) = &mut job.constraint {
            constraint.take(token);
        }
        job.reply.push(token);
        job.ended = job.output.wrote(token);
        job.written = self.text(&written(&job.reply))?;
        Ok(())
    }

    /// Ends the reply: its text and its images' and clips' tokens are the turn's, its speech's
    /// last samples spoken, or the turn is dropped when nothing was read.
    fn finish(&mut self, job: Job) {
        if let Some(mut speaking) = job.speaking {
            speaking.stream.finish(&mut self.speech);
        }
        if !job.reply.is_empty() {
            self.media.extend(job.tokens);
            self.history.push(Replied {
                turn: job.turn,
                reply: job.written,
            });
        }
    }

    /// The prompt's rows from the end of the longest kept prompt it begins with, or its first, in
    /// calls of up to `PREFILL_ROWS` rows, the last of the first `keep` rows ending there and
    /// keeping the state; its image and clip rows `media` in order: the sequence read before
    /// forgotten.
    fn read_prompt<'rows>(
        &mut self,
        prompt: &[u32],
        mut media: impl Iterator<Item = &'rows [f32; HIDDEN]>,
        keep: Option<usize>,
    ) {
        let mut first = self.restore_longest_kept(prompt);
        while first < prompt.len() {
            let stop = keep.filter(|row| *row > first).unwrap_or(prompt.len());
            let last = (first + PREFILL_ROWS).min(stop);
            self.embed_rows(&prompt[first..last], &mut media);
            self.forward(first, last - first);
            if Some(last) == keep {
                self.keep_rows(&prompt[..last]);
            }
            first = last;
        }
    }

    /// Puts back the state after the longest kept prompt `prompt` begins with, as the last used,
    /// or clears it: the row to read from.
    fn restore_longest_kept(&mut self, prompt: &[u32]) -> usize {
        let begun = self.kept.iter().enumerate();
        let longest = begun
            .filter(|(_, kept)| prompt.starts_with(&kept.ids))
            .max_by_key(|(_, kept)| kept.ids.len())
            .map(|(index, _)| index);
        let Some(index) = longest else {
            self.state.clear();
            return 0;
        };
        self.kept[..=index].rotate_right(1);
        let used = &self.kept[0];
        self.state.restore_prefix(&used.prefix);
        used.ids.len()
    }

    /// Keeps the state after `ids`, the prompt's first rows, as the last used; the least recently
    /// used goes once `KEPT_PROMPTS` are kept.
    fn keep_rows(&mut self, ids: &[u32]) {
        let prefix = self.state.keep_prefix(ids.len());
        let kept = Kept {
            ids: ids.to_vec(),
            prefix,
        };
        self.kept.insert(0, kept);
        self.kept.truncate(KEPT_PROMPTS);
    }

    /// The embeddings of `ids` into `buffers.hidden`, an image's or a clip's row the next of
    /// `media`.
    fn embed_rows<'rows>(
        &mut self,
        ids: &[u32],
        media: &mut impl Iterator<Item = &'rows [f32; HIDDEN]>,
    ) {
        let input = self.buffers.hidden.get_mut();
        for (&id, embedding) in ids.iter().zip(input.as_chunks_mut::<HIDDEN>().0) {
            if id == IMAGE_TOKEN || id == AUDIO_TOKEN {
                let row = media.next().expect("the prompt has a row per media token");
                embedding.copy_from_slice(row);
            } else {
                self.model.embed(id, embedding);
            }
        }
    }

    /// The reply's last row through the language model, after the prompt's rows.
    fn read_last(&mut self, job: &Job) {
        let token = *job
            .reply
            .last()
            .expect("the prompt is read before the reply's rows");
        let row = match (token, &job.speaking) {
            (AUDIO_OUT_TOKEN, Some(speaking)) => Row::Frame(&speaking.frame),
            _ => Row::Token(token),
        };
        self.write(job.prompt.len() + job.reply.len() - 1, row);
    }

    /// `row` through the language model at sequence position `position`.
    fn write(&mut self, position: usize, row: Row) {
        let embedding = &mut self.buffers.hidden.get_mut()[..HIDDEN];
        match row {
            Row::Token(id) => self.model.embed(id, embedding),
            Row::Frame(frame) => self
                .voice
                .as_ref()
                .expect("a reply speaks only with a voice")
                .embed(frame, embedding),
        }
        self.forward(position, 1);
    }

    /// `rows` rows from sequence position `first` through the stack, from `buffers.hidden`, and
    /// the prediction from the last of them.
    fn forward(&self, first: usize, rows: usize) {
        let positions: Vec<Position> = (first..first + rows).map(Position::uniform).collect();
        let rows = Rows {
            first,
            positions: &positions,
            mask: Mask::Causal,
        };
        let (model, state, buffers) = (&self.model, &self.state, &self.buffers);
        let prediction = &self.prediction;
        let last = positions.len() - 1;
        self.team.run(|ctx| {
            model.stack.forward(ctx, state, buffers, &rows);
            let hidden = &buffers.hidden.read()[last * HIDDEN..(last + 1) * HIDDEN];
            let norm = (&model.norm, NORM_EPS);
            prediction.predict(ctx, hidden, norm, &model.head);
        });
    }
}

/// The reply so far as the step leaves it: its reasoning apart, when the turn thinks, and its
/// answer as the template writes it back.
fn progress(stage: Stage, job: &Job, done: bool) -> Progress {
    let (reasoning, answer) = match reasoned(&job.written) {
        (Some(reasoning), answer) => (Some(reasoning), answer),
        (None, text) if job.opening.thinking => (Some(text), ""),
        (None, text) => (None, text),
    };
    Progress {
        stage: stage.name(),
        reasoning: reasoning.map(str::to_owned),
        answer: answer.to_owned(),
        tokens: written(&job.reply).len(),
        frames: job.speaking.as_ref().map_or(0, |speaking| speaking.frames),
        done,
    }
}

/// `values` as one row the blocks read.
fn shared(values: &[f32]) -> SharedBuf {
    let mut row = SharedBuf::new(values.len());
    row.get_mut().copy_from_slice(values);
    row
}

/// The tokens the projector left of an image of `grid`, `tokens × hidden`.
fn image_tokens(sight: &Sight, grid: Grid) -> Vec<f32> {
    let len = IMAGE.tokens(grid.patches()) * HIDDEN;
    sight.projector.tokens.read()[..len].to_vec()
}
