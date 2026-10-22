//! Reading an image to text, as a vision-language model does it: the image cut into patches
//! (`vit::preprocessing`), the vision tower (`vit::tower`) and the model's image tokens, the
//! prompt with those tokens through its language model (`lm`), then the answer, greedily. Each
//! `step` does one stage — the image, the prompt, or one more token of the answer — so that a
//! host streams the text and can stop between steps. The stages also run one by one, on given
//! inputs, for checks against the reference.
//!
//! A model implements `VisionLanguage`, what sets it apart; `Engine` is the reading.

use std::ops::Range;

use bundle::{Error, Stages, Store};
use media::image::Pixels;
use serde::{Deserialize, Serialize};
use team::{Ctx, SharedBuf, Team};

use crate::lm::{self, Lm, Positions, Reader, Rows};
use crate::rotary::Position;
use crate::vit::Grid;
use crate::vit::preprocessing::{Layout, Patches, Preprocessing};
use crate::vit::tower::{self, Tower};

#[derive(Debug, Clone, Deserialize)]
pub struct Request<T> {
    pub task: T,
    /// At most the model's `MAX_TOKENS`, its default.
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

/// A prompt for an image: its token ids, the image's tokens in some of its rows.
pub trait Prompt {
    fn ids(&self) -> &[u32];

    /// The rows of the image's tokens.
    fn image(&self) -> Range<usize>;

    /// The rotary positions of rows `rows`, prompt or answer, or `None` where they are their
    /// sequence positions.
    fn rotary(&self, rows: Range<usize>) -> Option<Vec<Position>>;
}

/// What sets a vision-language model apart: its prompt, the stage that makes its image's tokens
/// from the tower's output, and its dimensions.
pub trait VisionLanguage: Sized + Sync {
    /// What to read off the image.
    type Task: Copy;
    type Prompt: Prompt;
    type Stage: Stages;
    /// The activations of the image-token stage.
    type Buffers: Sync;

    /// The stages of the reading's steps: the image, the prompt, each token after the first.
    const SEE: Self::Stage;
    const PREFILL: Self::Stage;
    const DECODE: Self::Stage;
    /// The tokens an answer ends with.
    const ENDS: &'static [u32];
    /// The longest answer, in tokens.
    const MAX_TOKENS: usize;
    /// Prompt rows besides the image's tokens.
    const TEXT_ROWS: usize;
    /// Prompt rows per call of the language model's stack.
    const PREFILL_ROWS: usize;
    const PREPROCESSING: Preprocessing;
    const TOWER: tower::Shape;
    const LANGUAGE: lm::Shape;

    /// The model and its tokenizer, from a complete store.
    fn load(store: &mut Store) -> Result<Self, Error>;

    /// The image-token stage's activations, for the largest image `PREPROCESSING` makes.
    fn buffers() -> Self::Buffers;

    fn tower(&self) -> &Tower;

    fn language(&self) -> &Lm;

    /// The prompt for `task` on an image of `grid` patches, as the model's chat template writes
    /// it.
    fn compose(&self, task: Self::Task, grid: Grid) -> Result<Self::Prompt, Error>;

    /// The text of answer tokens.
    fn text(&self, ids: &[u32]) -> Result<String, Error>;

    /// The patches of an RGB or RGBA image.
    fn patches(image: &Pixels) -> Result<Patches, Error>;

    /// The image's tokens from the tower's output (`tower.encoder.hidden`), inside a team run.
    fn tokens(&self, ctx: &Ctx, tower: &tower::Buffers, buffers: &Self::Buffers, grid: Grid);

    /// Where `tokens` leaves them, `tokens × hidden` from the first row.
    fn image_tokens(buffers: &Self::Buffers) -> &SharedBuf;

    /// What the model writes into the prompt's rows around the image's tokens (`embeddings`
    /// holds every row); nothing by default.
    fn frame_image(&self, _prompt: &Self::Prompt, _embeddings: &mut [f32]) {}
}

pub struct Engine<M: VisionLanguage> {
    model: M,
    team: Team,
    tower: tower::Buffers,
    buffers: M::Buffers,
    reader: Reader,
    job: Option<Job<M::Prompt>>,
}

struct Job<P> {
    patches: Patches,
    prompt: P,
    answer: Vec<u32>,
    max_tokens: usize,
    next: Next,
}

/// What a reading's next step does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Next {
    /// The vision tower and the image's tokens.
    See,
    /// The prompt, and the answer's first token.
    Prompt,
    /// One more token.
    Answer,
}

impl<M: VisionLanguage> bundle::Engine for Engine<M> {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = M::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let patches = M::PREPROCESSING.max_patches();
        let capacity = M::PREPROCESSING.tokens(patches) + M::TEXT_ROWS + M::MAX_TOKENS;
        Ok(Self {
            model,
            tower: tower::Buffers::new(&M::TOWER, patches, threads),
            buffers: M::buffers(),
            reader: Reader::new(&M::LANGUAGE, capacity, M::PREFILL_ROWS, threads),
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

impl<M: VisionLanguage> Engine<M> {
    /// Each patch's embedding plus its position, `patches × hidden`.
    pub fn embed(&mut self, patches: &Patches) -> Vec<f32> {
        let grid = patches.grid;
        let positions = self.model.tower().positions(grid);
        let (tower, buffers) = (self.model.tower(), &self.tower);
        self.team
            .run(|ctx| tower.embed(ctx, buffers, &patches.values, &positions));
        let len = grid.patches() * M::TOWER.encoder.hidden;
        buffers.encoder.hidden.read()[..len].to_vec()
    }

    /// The tower's encoder (and final norm) over `embedded` (`embed`'s output).
    pub fn encode(&mut self, embedded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..embedded.len()].copy_from_slice(embedded);
        let (tower, buffers) = (self.model.tower(), &self.tower);
        self.team.run(|ctx| tower.encode(ctx, buffers, grid));
        buffers.encoder.hidden.read()[..embedded.len()].to_vec()
    }

    /// The image's tokens from `encoded` (`encode`'s output), `tokens × hidden`.
    pub fn tokens(&mut self, encoded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (model, tower, buffers) = (&self.model, &self.tower, &self.buffers);
        self.team.run(|ctx| model.tokens(ctx, tower, buffers, grid));
        self.image_tokens(grid)
    }

    /// The prompt for `task` on an image of `grid` patches, its text within `TEXT_ROWS`.
    pub fn prompt(&self, task: M::Task, grid: Grid) -> Result<M::Prompt, Error> {
        let prompt = self.model.compose(task, grid)?;
        if prompt.ids().len() - prompt.image().len() > M::TEXT_ROWS {
            return Err(Error::Request("the prompt's text is too long".into()));
        }
        Ok(prompt)
    }

    /// The text of answer tokens.
    pub fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.model.text(ids)
    }

    /// The prompt with the image's `tokens` (`tokens`' output) through the language model.
    pub fn prefill(&mut self, prompt: &M::Prompt, tokens: &[f32]) -> Prediction {
        self.read_prompt(prompt, tokens);
        Prediction {
            hidden: self.reader.last.read().to_vec(),
            logits: self.reader.logits.read().to_vec(),
        }
    }

    /// The greedy answer to the prompt with the image's `tokens`, its end excluded.
    pub fn answer(&mut self, prompt: &M::Prompt, tokens: &[f32], max_tokens: usize) -> Vec<u32> {
        self.read_prompt(prompt, tokens);
        let mut answer = vec![self.reader.greedy()];
        while answer.len() < max_tokens && !ended::<M>(&answer) {
            let next = self.write(prompt, &answer);
            answer.push(next);
        }
        answer.retain(|token| !M::ENDS.contains(token));
        answer
    }

    /// Starts reading an RGB or RGBA image.
    pub fn start(&mut self, image: &Pixels, request: &Request<M::Task>) -> Result<Plan, Error> {
        self.job = None;
        let patches = M::patches(image)?;
        let prompt = self.prompt(request.task, patches.grid)?;
        let plan = Plan {
            layout: M::PREPROCESSING.layout(image, patches.grid),
            image_tokens: prompt.image().len(),
            prompt_tokens: prompt.ids().len(),
        };
        self.job = Some(Job {
            patches,
            prompt,
            answer: Vec::new(),
            max_tokens: request
                .max_tokens
                .unwrap_or(M::MAX_TOKENS)
                .clamp(1, M::MAX_TOKENS),
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
        let stage = match step {
            Next::See => {
                self.see(&job.patches);
                job.next = Next::Prompt;
                M::SEE
            }
            Next::Prompt => {
                let tokens = self.image_tokens(job.patches.grid);
                self.read_prompt(&job.prompt, &tokens);
                job.answer.push(self.reader.greedy());
                job.next = Next::Answer;
                M::PREFILL
            }
            Next::Answer => {
                let next = self.write(&job.prompt, &job.answer);
                job.answer.push(next);
                M::DECODE
            }
        };
        let ended = ended::<M>(&job.answer);
        let done = ended || job.answer.len() >= job.max_tokens;
        let written = &job.answer[..job.answer.len() - usize::from(ended)];
        let progress = Progress {
            stage: stage.name(),
            text: self.text(written)?,
            tokens: written.len(),
            done,
        };
        if !done {
            self.job = Some(job);
        }
        Ok(progress)
    }

    /// The vision tower and the image's tokens over `patches`, in one team run.
    fn see(&mut self, patches: &Patches) {
        let grid = patches.grid;
        let positions = self.model.tower().positions(grid);
        let (model, tower, buffers) = (&self.model, &self.tower, &self.buffers);
        self.team.run(|ctx| {
            model.tower().embed(ctx, tower, &patches.values, &positions);
            ctx.barrier();
            model.tower().encode(ctx, tower, grid);
            model.tokens(ctx, tower, buffers, grid);
        });
    }

    fn image_tokens(&self, grid: Grid) -> Vec<f32> {
        let len = M::PREPROCESSING.tokens(grid.patches()) * M::LANGUAGE.stack.hidden;
        M::image_tokens(&self.buffers).read()[..len].to_vec()
    }

    /// The prompt from the first row, its image rows `tokens`.
    fn read_prompt(&mut self, prompt: &M::Prompt, tokens: &[f32]) {
        let (ids, hidden) = (prompt.ids(), M::LANGUAGE.stack.hidden);
        let mut embeddings = vec![0.0; ids.len() * hidden];
        self.model.language().embed(ids, &mut embeddings);
        let image = prompt.image();
        embeddings[image.start * hidden..image.end * hidden].copy_from_slice(tokens);
        self.model.frame_image(prompt, &mut embeddings);
        let rotary = prompt.rotary(0..ids.len());
        let rows = Rows {
            embeddings: &embeddings,
            first: 0,
            positions: positions(rotary.as_deref()),
        };
        self.reader.read(&self.team, self.model.language(), &rows);
    }

    /// The token after `answer`'s last, which the language model reads first.
    fn write(&mut self, prompt: &M::Prompt, answer: &[u32]) -> u32 {
        let index = prompt.ids().len() + answer.len() - 1;
        let mut embedding = vec![0.0; M::LANGUAGE.stack.hidden];
        self.model
            .language()
            .embed(&answer[answer.len() - 1..], &mut embedding);
        let rotary = prompt.rotary(index..index + 1);
        let rows = Rows {
            embeddings: &embedding,
            first: index,
            positions: positions(rotary.as_deref()),
        };
        self.reader.read(&self.team, self.model.language(), &rows);
        self.reader.greedy()
    }
}

/// Whether the answer's last token ends it.
fn ended<M: VisionLanguage>(answer: &[u32]) -> bool {
    answer.last().is_some_and(|token| M::ENDS.contains(token))
}

fn positions(rotary: Option<&[Position]>) -> Positions<'_> {
    rotary.map_or(Positions::Sequential, Positions::Given)
}
