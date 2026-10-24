//! A conversation: `start` takes the user's next message and its images and videos, cuts them
//! into patches and builds the conversation's prompt; then each `step` does one stage — an image
//! or a video's group through the vision tower and the merger, the prompt through the language
//! model, or one more token of the reply — so that a host streams the reply and can stop between
//! steps. The language model's state carries from turn to turn, saved at each reply's opener:
//! the template writes a reply back into the next prompt without its opener, so a turn reads the
//! last reply again from there, then its own message. The stages also run one by one, on given
//! inputs, for checks against the reference.

use std::ops::Range;

use ::media::image::Pixels;
use blocks::hybrid::{self, Rows};
use blocks::lm;
use blocks::transformer::Mask;
use blocks::vit::preprocessing::{Patches, Video};
use blocks::vit::{Grid, merger, tower};
use bundle::{Error, Stages as _, Store, python};
use serde::{Deserialize, Serialize};
use team::Team;

use crate::media::{self, IMAGE, Media};
use crate::model::{HIDDEN, LANGUAGE, MERGER, Model, NORM_EPS, TOWER, VOCABULARY};
use crate::pipeline::Stage;
use crate::prompt::{ENDS, Prompt, Replied, Turn, reasoned};
use crate::sampling::{Content, Sampler, Sampling};

/// The longest conversation, in rows: every turn's text, media and reply.
pub const MAX_ROWS: usize = 16_384;
/// The longest reply, in tokens.
pub const MAX_TOKENS: usize = 8_192;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 512;

/// The user's next message, and how to reply to it.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub text: String,
    /// Whether the reply reasons first, in a thinking block.
    #[serde(default)]
    pub thinking: bool,
    /// At most `MAX_TOKENS`, its default, and the rows the conversation has left.
    pub max_tokens: Option<usize>,
    /// The model card's settings for the turn when absent (`Sampling::preset`).
    pub sampling: Option<Sampling>,
    pub seed: u64,
}

/// How a turn is read: its media's grids, the prompt's rows (the whole conversation's) and those
/// the turn reads, the longest reply and its sampling.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub media: media::Plan,
    pub prompt_tokens: usize,
    pub read_tokens: usize,
    pub max_tokens: usize,
    pub sampling: Sampling,
}

/// One step done: the reply so far, its reasoning apart when the turn thinks.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub reasoning: Option<String>,
    pub answer: String,
    pub tokens: usize,
    pub done: bool,
}

/// The last row's state: its final-normed hidden state and its logits.
pub struct Prediction {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct Engine {
    model: Model,
    team: Team,
    tower: tower::Buffers,
    merger: merger::Buffers,
    buffers: hybrid::Buffers,
    state: hybrid::State,
    saved: hybrid::Saved,
    prediction: lm::Prediction,
    history: Vec<Replied>,
    /// The row of the last reply's opener, the state before it in `saved`; none in a new
    /// conversation.
    resume: Option<usize>,
    job: Option<Job>,
}

struct Job {
    turn: Turn,
    media: Media,
    /// The tokens of the images and video groups seen so far, in prompt order, `tokens × hidden`.
    tokens: Vec<f32>,
    seen: usize,
    prompt: Prompt,
    thinking: bool,
    sampler: Sampler,
    reply: Vec<u32>,
    /// The reply's text so far, its end excluded.
    text: String,
    max_tokens: usize,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let patches = IMAGE.max_patches();
        Ok(Self {
            tower: tower::Buffers::new(&TOWER, patches, threads),
            merger: merger::Buffers::new(&MERGER, patches),
            buffers: LANGUAGE.buffers(PREFILL_ROWS, threads),
            state: hybrid::State::new(&LANGUAGE, MAX_ROWS),
            saved: hybrid::Saved::new(&LANGUAGE),
            prediction: lm::Prediction::new(HIDDEN, VOCABULARY, threads),
            model,
            team,
            history: Vec::new(),
            resume: None,
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
    /// Each patch's embedding plus its position, `patches × width`.
    pub fn embed(&mut self, patches: &Patches) -> Vec<f32> {
        let positions = self.model.vision.positions(patches.grid);
        let (vision, buffers) = (&self.model.vision, &self.tower);
        self.team
            .run(|ctx| vision.embed(ctx, buffers, &patches.values, &positions));
        let len = patches.grid.patches() * TOWER.encoder.hidden;
        buffers.encoder.hidden.read()[..len].to_vec()
    }

    /// The tower's encoder over `embedded` (`embed`'s output).
    pub fn encode(&mut self, embedded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..embedded.len()].copy_from_slice(embedded);
        let (vision, buffers) = (&self.model.vision, &self.tower);
        self.team.run(|ctx| vision.encode(ctx, buffers, grid));
        buffers.encoder.hidden.read()[..embedded.len()].to_vec()
    }

    /// An image's or a group's tokens from `encoded` (`encode`'s output), `tokens × hidden`.
    pub fn merge(&mut self, encoded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (merger, tower, buffers) = (&self.model.merger, &self.tower, &self.merger);
        self.team
            .run(|ctx| merger.forward(ctx, tower, buffers, grid));
        self.media_tokens(grid)
    }

    /// An image's or a group's patches through the tower and the merger, in one team run.
    fn see(&mut self, patches: &Patches) -> Vec<f32> {
        let grid = patches.grid;
        let positions = self.model.vision.positions(grid);
        let (model, tower, buffers) = (&self.model, &self.tower, &self.merger);
        self.team.run(|ctx| {
            model.vision.embed(ctx, tower, &patches.values, &positions);
            ctx.barrier();
            model.vision.encode(ctx, tower, grid);
            model.merger.forward(ctx, tower, buffers, grid);
        });
        self.media_tokens(grid)
    }

    fn media_tokens(&self, grid: Grid) -> Vec<f32> {
        let len = IMAGE.tokens(grid.patches()) * HIDDEN;
        self.merger.tokens.read()[..len].to_vec()
    }

    /// The prompt asking for the reply to `next`, after `history`.
    pub fn prompt(
        &self,
        history: &[Replied],
        next: &Turn,
        thinking: bool,
    ) -> Result<Prompt, Error> {
        Prompt::new(&self.model.tokenizer, history, next, thinking)
    }

    /// The text of reply tokens.
    pub fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.model.tokenizer.decode(ids)
    }

    /// Forgets the conversation, to start a new one.
    pub fn reset(&mut self) {
        self.job = None;
        self.history.clear();
        self.resume = None;
    }

    /// The prompt through the language model from the last reply's opener (from its first row
    /// in a new conversation), the media rows from there on `tokens`, in order: the state saved
    /// at the prompt's opener, and the last row's prediction.
    pub fn prefill(&mut self, prompt: &Prompt, tokens: &[f32]) -> Prediction {
        self.read_prompt(prompt, tokens);
        self.prediction()
    }

    /// After `prefill`, `reply`'s tokens read in order: the prediction after each.
    pub fn force(&mut self, prompt: &Prompt, reply: &[u32]) -> Vec<Prediction> {
        let read = (1..=reply.len()).map(|written| {
            self.write(prompt, &reply[..written]);
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

    /// Starts replying to `message` with RGB or RGBA `images` and `videos`, the conversation's
    /// next turn.
    pub fn start(
        &mut self,
        message: &Message,
        images: &[Pixels],
        videos: &[Video],
    ) -> Result<Plan, Error> {
        bundle::Engine::cancel(self);
        let media = Media::new(images, videos)?;
        let turn = Turn {
            text: message.text.clone(),
            media: media.kept(),
        };
        let job = self.job(message, turn, media)?;
        let plan = Plan {
            media: job.media.plan(),
            prompt_tokens: job.prompt.ids.len(),
            read_tokens: job.prompt.ids.len() - self.resume.unwrap_or(0),
            max_tokens: job.max_tokens,
            sampling: job.sampler.sampling(),
        };
        self.job = Some(job);
        Ok(plan)
    }

    /// The job replying to `turn`, from `message`, its media `media`.
    fn job(&self, message: &Message, turn: Turn, media: Media) -> Result<Job, Error> {
        let prompt = self.prompt(&self.history, &turn, message.thinking)?;
        let rows = prompt.ids.len();
        if rows >= MAX_ROWS {
            return Err(Error::Request(format!(
                "the conversation has {rows} tokens, the most this runtime holds is {MAX_ROWS}; \
                 start a new one"
            )));
        }
        let history = self.history.iter().map(|replied| &replied.turn);
        let content = if history.chain([&turn]).any(|turn| !turn.media.is_empty()) {
            Content::Vision
        } else {
            Content::Text
        };
        let sampling = message
            .sampling
            .unwrap_or_else(|| Sampling::preset(message, content));
        sampling.check()?;
        let max_tokens = message.max_tokens.unwrap_or(MAX_TOKENS);
        Ok(Job {
            turn,
            media,
            tokens: Vec::new(),
            seen: 0,
            prompt,
            thinking: message.thinking,
            sampler: Sampler::new(sampling, message.seed),
            reply: Vec::new(),
            text: String::new(),
            max_tokens: max_tokens.clamp(1, MAX_TOKENS).min(MAX_ROWS - rows),
        })
    }

    /// Runs the next stage: the next image or video group, the prompt, or one more token.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no reply in progress".into()))?;
        let stage = if let Some(patches) = job.media.items().nth(job.seen) {
            let tokens = self.see(patches);
            job.tokens.extend(tokens);
            job.seen += 1;
            Stage::Merge
        } else if job.reply.is_empty() {
            self.read_prompt(&job.prompt, &job.tokens);
            self.draw(&mut job);
            Stage::Prefill
        } else {
            self.write(&job.prompt, &job.reply);
            self.draw(&mut job);
            Stage::Decode
        };
        job.text = self.text(written(&job.reply))?;
        let done = ended(&job.reply) || job.reply.len() >= job.max_tokens;
        let progress = progress(stage, &job, done);
        if done {
            self.finish(job);
        } else {
            self.job = Some(job);
        }
        Ok(progress)
    }

    fn draw(&mut self, job: &mut Job) {
        let token = job.sampler.draw(self.prediction.logits.get_mut());
        job.reply.push(token);
    }

    /// Ends the reply: its text is the turn's, or the turn is dropped when nothing was read.
    fn finish(&mut self, job: Job) {
        if !job.reply.is_empty() {
            self.history.push(Replied {
                turn: job.turn,
                reply: job.text,
            });
        }
    }

    /// The prompt's rows from the last reply's opener (its first in a new conversation), in calls
    /// of up to `PREFILL_ROWS` rows, the last before the prompt's opener ending there and
    /// saving the state; the media rows read are `tokens`, in order.
    fn read_prompt(&mut self, prompt: &Prompt, tokens: &[f32]) {
        let mut first = match self.resume {
            Some(row) => {
                self.state.restore(&self.saved);
                row
            }
            None => {
                self.state.clear();
                0
            }
        };
        let (opener, end) = (prompt.opener, prompt.ids.len());
        let mut media_tokens = tokens.as_chunks::<HIDDEN>().0.iter();
        while first < end {
            let last = (first + PREFILL_ROWS).min(if first < opener { opener } else { end });
            self.embed_rows(prompt, first..last, &mut media_tokens);
            let rows = Rows {
                first,
                positions: &prompt.positions[first..last],
                mask: Mask::Causal,
            };
            self.forward(&rows);
            if last == opener {
                self.state.save(&mut self.saved);
            }
            first = last;
        }
        self.resume = Some(opener);
    }

    /// The input of `rows`: each one's token embedding, or the next of `media_tokens` in the
    /// media's rows.
    fn embed_rows<'tokens>(
        &mut self,
        prompt: &Prompt,
        rows: Range<usize>,
        media_tokens: &mut impl Iterator<Item = &'tokens [f32; HIDDEN]>,
    ) {
        let input = &mut self.buffers.stack.hidden.get_mut()[..rows.len() * HIDDEN];
        for (row, embedding) in rows.zip(input.as_chunks_mut::<HIDDEN>().0) {
            if prompt.media.iter().any(|(range, _)| range.contains(&row)) {
                let token = media_tokens
                    .next()
                    .expect("a token for every media row read");
                embedding.copy_from_slice(token);
            } else {
                self.model.language.embed(prompt.ids[row], embedding);
            }
        }
    }

    /// The reply's last token through the language model, in its row after the prompt.
    fn write(&mut self, prompt: &Prompt, reply: &[u32]) {
        let index = reply.len() - 1;
        let embedding = &mut self.buffers.stack.hidden.get_mut()[..HIDDEN];
        self.model.language.embed(reply[index], embedding);
        let position = [prompt.reply_position(index)];
        let rows = Rows {
            first: prompt.ids.len() + index,
            positions: &position,
            mask: Mask::Causal,
        };
        self.forward(&rows);
    }

    /// `rows` through the stack, from `buffers.stack.hidden`, and the prediction from the last
    /// of them.
    fn forward(&self, rows: &Rows) {
        let language = &self.model.language;
        let (state, buffers, prediction) = (&self.state, &self.buffers, &self.prediction);
        let last = rows.positions.len() - 1;
        self.team.run(|ctx| {
            language.stack.forward(ctx, state, buffers, rows);
            let hidden = &buffers.stack.hidden.read()[last * HIDDEN..(last + 1) * HIDDEN];
            let norm = (&language.norm, NORM_EPS);
            prediction.predict(ctx, hidden, norm, &language.head);
        });
    }
}

/// The reply so far as the step leaves it: its reasoning apart, when the turn thinks, and its
/// answer as the template writes it back.
fn progress(stage: Stage, job: &Job, done: bool) -> Progress {
    let (reasoning, answer) = match reasoned(&job.text) {
        (Some(reasoning), answer) => (Some(reasoning), answer),
        (None, text) if job.thinking => (Some(text), ""),
        (None, text) => (None, text),
    };
    Progress {
        stage: stage.name(),
        reasoning: reasoning.map(|reasoning| python::strip(reasoning).to_owned()),
        answer: answer.to_owned(),
        tokens: written(&job.reply).len(),
        done,
    }
}

/// Whether the reply's last token ends it.
fn ended(reply: &[u32]) -> bool {
    reply.last().is_some_and(|token| ENDS.contains(token))
}

/// The reply's tokens, its end excluded.
fn written(reply: &[u32]) -> &[u32] {
    &reply[..reply.len() - usize::from(ended(reply))]
}
