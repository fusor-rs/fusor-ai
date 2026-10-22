//! Reading an image: `start` cuts it into patches and builds the task's prompt; then each `step`
//! does one stage — the vision tower and projector, the prompt through the language model, or
//! one more token of the answer — so that a host streams the text and can stop between steps.
//! The stages also run one by one, on given inputs, for checks against the reference.

use blocks::vit::Grid;
use bundle::{Store, Tokenizer};
use kernels::image::Pixels;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::Error;
use crate::image::{Layout, Patches};
use crate::language::{Reader, Rows};
use crate::model::{HIDDEN, MERGE, Model, VISION_SHAPE};
use crate::pipeline::Stage;
use crate::prompt::{END, Prompt, Task};
use crate::vision::{self, MAX_PATCHES};

/// The longest answer, in tokens.
pub const MAX_TOKENS: usize = 2048;
/// Prompt rows besides the image's: the chat template and the longest instruction.
const TEXT_ROWS: usize = 32;
/// Rows the language model holds: the longest prompt and the longest answer.
const CAPACITY: usize = MAX_PATCHES / (MERGE * MERGE) + TEXT_ROWS + MAX_TOKENS;

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    team: Team,
    vision: vision::Buffers,
    reader: Reader,
    job: Option<Job>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub task: Task,
    /// At most `MAX_TOKENS`, its default.
    pub max_tokens: Option<usize>,
}

/// How the image is read: its layout, its tokens and the prompt's rows.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub layout: Layout,
    pub image_tokens: usize,
    pub prompt_tokens: usize,
}

/// One step done: the answer so far.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub text: String,
    pub tokens: usize,
    pub done: bool,
}

/// The last row's state after the prompt: its final-normed hidden state and its logits.
pub struct Prediction {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

struct Job {
    patches: Patches,
    prompt: Prompt,
    answer: Vec<u32>,
    max_tokens: usize,
    next: Next,
}

/// What a reading's next step does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Next {
    /// The vision tower and the projector.
    See,
    /// The prompt, and the answer's first token.
    Prompt,
    /// One more token.
    Answer,
}

impl Next {
    fn stage(self) -> Stage {
        match self {
            Next::See => Stage::Encode,
            Next::Prompt => Stage::Prefill,
            Next::Answer => Stage::Decode,
        }
    }
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let tokenizer = Tokenizer::take(store)?;
        let team = Team::global();
        let threads = team.threads();
        Ok(Self {
            model,
            tokenizer,
            vision: vision::Buffers::new(threads),
            reader: Reader::new(CAPACITY, threads),
            team,
            job: None,
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
    pub fn prompt(&self, task: Task, grid: Grid) -> Result<Prompt, Error> {
        let prompt = Prompt::new(&self.tokenizer, task, grid)?;
        if prompt.ids.len() - prompt.image.len() > TEXT_ROWS {
            return Err(Error::Request("the prompt's text is too long".into()));
        }
        Ok(prompt)
    }

    /// The text of answer tokens.
    pub fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer.decode(ids)
    }

    /// Each patch's embedding plus its position, `patches × VISION_SHAPE.hidden`.
    pub fn embed(&mut self, patches: &Patches) -> Vec<f32> {
        let grid = patches.grid;
        let positions = self.model.vision.positions(grid);
        let (vision, buffers) = (&self.model.vision, &self.vision);
        self.team
            .run(|ctx| vision.embed(ctx, buffers, &patches.values, &positions));
        let len = grid.patches() * VISION_SHAPE.hidden;
        buffers.encoder.hidden.read()[..len].to_vec()
    }

    /// The encoder and its final norm over `embedded` (`embed`'s output).
    pub fn encode(&mut self, embedded: &[f32], grid: Grid) -> Vec<f32> {
        self.vision.encoder.hidden.get_mut()[..embedded.len()].copy_from_slice(embedded);
        let (vision, buffers) = (&self.model.vision, &self.vision);
        self.team.run(|ctx| vision.encode(ctx, buffers, grid));
        buffers.seen.read()[..embedded.len()].to_vec()
    }

    /// The projector over `seen` (`encode`'s output): the image's tokens, `tokens × HIDDEN`.
    pub fn project(&mut self, seen: &[f32], grid: Grid) -> Vec<f32> {
        self.vision.seen.get_mut()[..seen.len()].copy_from_slice(seen);
        let (projector, buffers) = (&self.model.projector, &self.vision);
        self.team.run(|ctx| projector.forward(ctx, buffers, grid));
        let len = grid.patches() / (MERGE * MERGE) * HIDDEN;
        buffers.tokens.read()[..len].to_vec()
    }

    /// The prompt with the image's `tokens` (`project`'s output) through the language model.
    pub fn prefill(&mut self, prompt: &Prompt, tokens: &[f32]) -> Prediction {
        self.read_prompt(prompt, tokens);
        Prediction {
            hidden: self.reader.last.read().to_vec(),
            logits: self.reader.logits.read().to_vec(),
        }
    }

    /// The greedy answer to the prompt with the image's `tokens`, its end excluded.
    pub fn answer(&mut self, prompt: &Prompt, tokens: &[f32], max_tokens: usize) -> Vec<u32> {
        self.read_prompt(prompt, tokens);
        let mut answer = vec![self.reader.greedy()];
        while answer.len() < max_tokens && answer.last() != Some(&END) {
            let next = self.write(prompt, &answer);
            answer.push(next);
        }
        answer.retain(|&token| token != END);
        answer
    }

    /// Starts reading an RGB or RGBA image.
    pub fn start(&mut self, image: &Pixels, request: &Request) -> Result<Plan, Error> {
        self.job = None;
        let patches = Patches::of(image)?;
        let prompt = self.prompt(request.task, patches.grid)?;
        let plan = Plan {
            layout: patches.layout(image),
            image_tokens: prompt.image.len(),
            prompt_tokens: prompt.ids.len(),
        };
        self.job = Some(Job {
            patches,
            prompt,
            answer: Vec::new(),
            max_tokens: request
                .max_tokens
                .unwrap_or(MAX_TOKENS)
                .clamp(1, MAX_TOKENS),
            next: Next::See,
        });
        Ok(plan)
    }

    /// Runs the next stage of the reading.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no reading in progress".into()))?;
        let step = job.next;
        match step {
            Next::See => {
                self.see(&job.patches);
                job.next = Next::Prompt;
            }
            Next::Prompt => {
                let tokens = self.image_tokens(job.patches.grid);
                self.read_prompt(&job.prompt, &tokens);
                job.answer.push(self.reader.greedy());
                job.next = Next::Answer;
            }
            Next::Answer => {
                let next = self.write(&job.prompt, &job.answer);
                job.answer.push(next);
            }
        }
        let ended = job.answer.last() == Some(&END);
        let done = ended || job.answer.len() >= job.max_tokens;
        let written = &job.answer[..job.answer.len() - usize::from(ended)];
        let progress = Progress {
            stage: step.stage().name(),
            text: self.text(written)?,
            tokens: written.len(),
            done,
        };
        if !done {
            self.job = Some(job);
        }
        Ok(progress)
    }

    /// The vision tower and the projector over `patches`, in one team run, into `tokens`.
    fn see(&mut self, patches: &Patches) {
        let grid = patches.grid;
        let positions = self.model.vision.positions(grid);
        let (model, buffers) = (&self.model, &self.vision);
        self.team.run(|ctx| {
            model
                .vision
                .embed(ctx, buffers, &patches.values, &positions);
            ctx.barrier();
            model.vision.encode(ctx, buffers, grid);
            model.projector.forward(ctx, buffers, grid);
        });
    }

    fn image_tokens(&self, grid: Grid) -> Vec<f32> {
        let len = grid.patches() / (MERGE * MERGE) * HIDDEN;
        self.vision.tokens.read()[..len].to_vec()
    }

    /// The prompt from the first row, its image rows `tokens`.
    fn read_prompt(&mut self, prompt: &Prompt, tokens: &[f32]) {
        let mut embeddings = vec![0.0; prompt.ids.len() * HIDDEN];
        self.model.language.embed(&prompt.ids, &mut embeddings);
        let image = prompt.image.start * HIDDEN..prompt.image.end * HIDDEN;
        embeddings[image].copy_from_slice(tokens);
        let rows = Rows {
            embeddings: &embeddings,
            positions: &prompt.positions(),
            first: 0,
        };
        self.reader.read(&self.team, &self.model.language, &rows);
    }

    /// The token after `answer`'s last, which the language model reads first.
    fn write(&mut self, prompt: &Prompt, answer: &[u32]) -> u32 {
        let index = prompt.ids.len() + answer.len() - 1;
        let mut embedding = [0.0; HIDDEN];
        self.model
            .language
            .embed(&answer[answer.len() - 1..], &mut embedding);
        let rows = Rows {
            embeddings: &embedding,
            positions: &[prompt.position(index)],
            first: index,
        };
        self.reader.read(&self.team, &self.model.language, &rows);
        self.reader.greedy()
    }
}
