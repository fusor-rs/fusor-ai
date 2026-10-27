//! A conversation: `start` takes the user's next message and builds the conversation's prompt;
//! then each `step` does one stage — the prompt through the language model, or one more token
//! of the reply — so that a host streams the reply and can stop between steps. Each turn reads
//! its whole prompt from its first row: the template writes a reply back without its reasoning,
//! and the window layers keep only their last rows' keys. The stages also run one by one, on
//! given inputs, for checks against the reference.

use blocks::hybrid::Rows;
use blocks::interleaved::State;
use blocks::lm;
use blocks::rotary::Position;
use blocks::transformer::{Buffers, Mask};
use bundle::{Error, Stages as _, Store};
use serde::{Deserialize, Serialize};
use team::Team;

use crate::model::{HIDDEN, LANGUAGE, Model, NORM_EPS, VOCABULARY};
use crate::pipeline::Stage;
use crate::prompt::{self, REPLY_END, Replied, reasoned};
use crate::sampling::{PRESET, Sampler, Sampling};

/// The longest conversation, in rows: every turn's text and reply.
pub const MAX_ROWS: usize = 8_192;
/// The longest reply, in tokens.
pub const MAX_TOKENS: usize = 4_096;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 512;

/// The user's next message, and how to reply to it.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub text: String,
    /// The conversation's system text, the same on each of its turns.
    #[serde(default)]
    pub system: String,
    /// Whether the reply reasons first, in a thinking block.
    #[serde(default)]
    pub thinking: bool,
    /// At most `MAX_TOKENS`, its default, and the rows the conversation has left.
    pub max_tokens: Option<usize>,
    /// The model card's settings (`PRESET`) when absent.
    pub sampling: Option<Sampling>,
    pub seed: u64,
}

/// How a turn is read: the prompt's rows, the longest reply and its sampling.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub prompt_tokens: usize,
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
    buffers: Buffers,
    state: State,
    prediction: lm::Prediction,
    history: Vec<Replied>,
    job: Option<Job>,
}

struct Job {
    text: String,
    prompt: Vec<u32>,
    thinking: bool,
    sampler: Sampler,
    reply: Vec<u32>,
    /// The reply's text so far, its end excluded.
    written: String,
    max_tokens: usize,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        Ok(Self {
            buffers: LANGUAGE.buffers(PREFILL_ROWS, threads),
            state: State::new(&LANGUAGE, MAX_ROWS, PREFILL_ROWS),
            prediction: lm::Prediction::new(HIDDEN, VOCABULARY, threads),
            model,
            team,
            history: Vec::new(),
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
    /// The prompt asking for the reply to `next`, after `system` and `history`.
    pub fn prompt(
        &self,
        system: &str,
        history: &[Replied],
        next: &str,
        thinking: bool,
    ) -> Result<Vec<u32>, Error> {
        prompt::prompt(&self.model.tokenizer, system, history, next, thinking)
    }

    /// The text of reply tokens.
    pub fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.model.tokenizer.decode(ids)
    }

    /// Forgets the conversation, to start a new one.
    pub fn reset(&mut self) {
        self.job = None;
        self.history.clear();
    }

    /// The prompt through the language model: the last row's prediction.
    pub fn prefill(&mut self, prompt: &[u32]) -> Prediction {
        self.read_prompt(prompt);
        self.prediction()
    }

    /// After `prefill`, `reply`'s tokens read in order: the prediction after each.
    pub fn force(&mut self, prompt: &[u32], reply: &[u32]) -> Vec<Prediction> {
        let read = (1..=reply.len()).map(|written| {
            self.write(prompt.len(), &reply[..written]);
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

    /// Starts replying to `message`, the conversation's next turn.
    ///
    /// # Errors
    /// When the conversation would outgrow `MAX_ROWS`, or the sampling is invalid.
    pub fn start(&mut self, message: &Message) -> Result<Plan, Error> {
        bundle::Engine::cancel(self);
        let prompt = self.prompt(
            &message.system,
            &self.history,
            &message.text,
            message.thinking,
        )?;
        let rows = prompt.len();
        if rows >= MAX_ROWS {
            return Err(Error::Request(format!(
                "the conversation has {rows} tokens, the most this runtime holds is {MAX_ROWS}; \
                 start a new one"
            )));
        }
        let sampling = message.sampling.unwrap_or(PRESET);
        sampling.check()?;
        let max_tokens = message.max_tokens.unwrap_or(MAX_TOKENS);
        let job = Job {
            text: message.text.clone(),
            prompt,
            thinking: message.thinking,
            sampler: Sampler::new(sampling, message.seed),
            reply: Vec::new(),
            written: String::new(),
            max_tokens: max_tokens.clamp(1, MAX_TOKENS).min(MAX_ROWS - rows),
        };
        let plan = Plan {
            prompt_tokens: rows,
            max_tokens: job.max_tokens,
            sampling,
        };
        self.job = Some(job);
        Ok(plan)
    }

    /// Runs the next stage: the prompt, or one more token.
    ///
    /// # Errors
    /// When no reply is in progress, or the reply doesn't detokenize.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no reply in progress".into()))?;
        let stage = if job.reply.is_empty() {
            self.read_prompt(&job.prompt);
            Stage::Prefill
        } else {
            self.write(job.prompt.len(), &job.reply);
            Stage::Decode
        };
        let token = job.sampler.draw(self.prediction.logits.get_mut());
        job.reply.push(token);
        job.written = self.text(written(&job.reply))?;
        let done = ended(&job.reply) || job.reply.len() >= job.max_tokens;
        let progress = progress(stage, &job, done);
        if done {
            self.finish(job);
        } else {
            self.job = Some(job);
        }
        Ok(progress)
    }

    /// Ends the reply: its text is the turn's, or the turn is dropped when nothing was read.
    fn finish(&mut self, job: Job) {
        if !job.reply.is_empty() {
            self.history.push(Replied {
                text: job.text,
                reply: job.written,
            });
        }
    }

    /// The prompt's rows from its first, in calls of up to `PREFILL_ROWS` rows.
    fn read_prompt(&mut self, prompt: &[u32]) {
        for first in (0..prompt.len()).step_by(PREFILL_ROWS) {
            let ids = &prompt[first..(first + PREFILL_ROWS).min(prompt.len())];
            let input = self.buffers.hidden.get_mut();
            for (&id, embedding) in ids.iter().zip(input.as_chunks_mut::<HIDDEN>().0) {
                self.model.embed(id, embedding);
            }
            self.forward(first, ids.len());
        }
    }

    /// The reply's last token through the language model, in its row after the prompt's
    /// `prompt_rows`.
    fn write(&mut self, prompt_rows: usize, reply: &[u32]) {
        let index = reply.len() - 1;
        let embedding = &mut self.buffers.hidden.get_mut()[..HIDDEN];
        self.model.embed(reply[index], embedding);
        self.forward(prompt_rows + index, 1);
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
        let last = positions.len() - 1;
        self.team.run(|ctx| {
            model.stack.forward(ctx, state, buffers, &rows);
            let hidden = &buffers.hidden.read()[last * HIDDEN..(last + 1) * HIDDEN];
            let norm = (&model.norm, NORM_EPS);
            self.prediction.predict(ctx, hidden, norm, &model.head);
        });
    }
}

/// The reply so far as the step leaves it: its reasoning apart, when the turn thinks, and its
/// answer as the template writes it back.
fn progress(stage: Stage, job: &Job, done: bool) -> Progress {
    let (reasoning, answer) = match reasoned(&job.written) {
        (Some(reasoning), answer) => (Some(reasoning), answer),
        (None, text) if job.thinking => (Some(text), ""),
        (None, text) => (None, text),
    };
    Progress {
        stage: stage.name(),
        reasoning: reasoning.map(str::to_owned),
        answer: answer.to_owned(),
        tokens: written(&job.reply).len(),
        done,
    }
}

/// Whether the reply's last token ends it.
fn ended(reply: &[u32]) -> bool {
    reply.last() == Some(&REPLY_END)
}

/// The reply's tokens, its end excluded.
fn written(reply: &[u32]) -> &[u32] {
    &reply[..reply.len() - usize::from(ended(reply))]
}
