//! Deciding a request: `start` checks it, cuts its images into patches and compiles its prompt;
//! then each `step` does one stage — an image through the vision tower and merger, or a chunk of
//! the prompt through the language model, the last with the answers. The stages also run one by
//! one, on given inputs, for checks against the reference.

use std::ops::Range;

use blocks::hybrid::{self, Rows};
use blocks::rotary::Position;
use blocks::vit::merger;
use blocks::vit::preprocessing::{Layout, Patches};
use blocks::vit::{Grid, tower};
use bundle::{Error, Store};
use kernels::dot_f32;
use kernels::ops::rms_norm;
use media::image::Pixels;
use serde::Serialize;
use team::Team;

use crate::decision::{self, Response};
use crate::image::{self, PREPROCESSING};
use crate::model::{HIDDEN, LANGUAGE_SHAPE, MERGER, Model, NORM_EPS, TOWER};
use crate::prompt::Prompt;
use crate::request::{Question, Request, SYMBOLS};

/// The longest prompt, in rows: the request's text and its images' tokens.
pub const MAX_ROWS: usize = 4096;
/// The most images a request holds.
pub const MAX_IMAGES: usize = 8;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 128;

/// How a request is read: its images' layouts and the prompt's rows.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub images: Vec<Layout>,
    pub image_tokens: usize,
    pub prompt_tokens: usize,
    pub questions: usize,
}

/// One step done: the prompt's rows read so far, and the response once all are.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub read: usize,
    pub rows: usize,
    pub response: Option<Response>,
}

pub struct Engine {
    model: Model,
    team: Team,
    tower: tower::Buffers,
    merger: merger::Buffers,
    buffers: hybrid::Buffers,
    state: hybrid::State,
    /// The head's rows of the options' symbols, `SYMBOLS × HIDDEN`, in their order.
    symbols: Vec<f32>,
    job: Option<Job>,
}

/// What the language model reads: the prompt, its rows' rotary positions and its images'
/// tokens.
#[derive(Clone, Copy)]
struct Sequence<'job> {
    prompt: &'job Prompt,
    positions: &'job [Position],
    tokens: &'job [f32],
}

struct Job {
    questions: Vec<Question>,
    prompt: Prompt,
    positions: Vec<Position>,
    patches: Vec<Patches>,
    /// Images through the tower so far, and their tokens, in order, `tokens × HIDDEN`.
    seen: usize,
    tokens: Vec<f32>,
    /// The final-normed state of each deciding row read so far.
    hidden: Vec<f32>,
    read: usize,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let patches = PREPROCESSING.max_patches();
        let symbols = symbol_rows(&model)?;
        Ok(Self {
            tower: tower::Buffers::new(&TOWER, patches, threads),
            merger: merger::Buffers::new(&MERGER, patches),
            buffers: LANGUAGE_SHAPE.buffers(PREFILL_ROWS, threads),
            state: hybrid::State::new(&LANGUAGE_SHAPE, MAX_ROWS),
            symbols,
            model,
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

/// The head's rows of the symbols, each symbol one token.
fn symbol_rows(model: &Model) -> Result<Vec<f32>, Error> {
    let mut rows = vec![0.0; SYMBOLS.len() * HIDDEN];
    let table = model.language.embed.matrix();
    for (symbol, row) in SYMBOLS.chars().zip(rows.as_chunks_mut::<HIDDEN>().0) {
        let ids = model.tokenizer.encode(&symbol.to_string())?;
        let [id] = ids[..] else {
            return Err(Error::Format(format!(
                "tokenizer: {symbol} is not one token"
            )));
        };
        table.dequantize_row(id as usize, row);
    }
    Ok(rows)
}

impl Engine {
    /// The questions and prompt of `request`, with images of `grids` patches.
    pub fn prompt(
        &self,
        request: &Request,
        grids: &[Grid],
    ) -> Result<(Vec<Question>, Prompt), Error> {
        let questions = request.questions()?;
        let prompt = Prompt::new(&self.model.tokenizer, &request.state, &questions, grids)?;
        if prompt.ids.len() > MAX_ROWS {
            return Err(Error::Request(format!(
                "the request is {} tokens, its images' included: at most {MAX_ROWS}",
                prompt.ids.len()
            )));
        }
        Ok((questions, prompt))
    }

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

    /// The image's tokens from `encoded` (`encode`'s output), `tokens × HIDDEN`.
    pub fn merge(&mut self, encoded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (merger, tower, buffers) = (&self.model.merger, &self.tower, &self.merger);
        self.team
            .run(|ctx| merger.forward(ctx, tower, buffers, grid));
        self.image_tokens(grid)
    }

    /// An image's patches through the tower and the merger, in one team run: its tokens.
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
        self.image_tokens(grid)
    }

    fn image_tokens(&self, grid: Grid) -> Vec<f32> {
        let len = PREPROCESSING.tokens(grid.patches()) * HIDDEN;
        self.merger.tokens.read()[..len].to_vec()
    }

    /// The prompt, its images' rows `tokens` (every image's, in order), through the language
    /// model: the final-normed state of each deciding row, `questions × HIDDEN`.
    pub fn read(&mut self, prompt: &Prompt, tokens: &[f32]) -> Vec<f32> {
        let positions = prompt.positions();
        let sequence = Sequence {
            prompt,
            positions: &positions,
            tokens,
        };
        let mut hidden = Vec::new();
        for first in (0..prompt.ids.len()).step_by(PREFILL_ROWS) {
            let rows = first..prompt.ids.len().min(first + PREFILL_ROWS);
            hidden.extend(self.read_rows(&sequence, rows));
        }
        hidden
    }

    /// Rows `rows` of `sequence` through the language model, from the first (where the state
    /// starts over): the final-normed state of the deciding rows among them.
    fn read_rows(&mut self, sequence: &Sequence, rows: Range<usize>) -> Vec<f32> {
        let Sequence {
            prompt,
            positions,
            tokens,
        } = *sequence;
        if rows.start == 0 {
            self.state.clear();
        }
        let language = &self.model.language;
        let input = &mut self.buffers.stack.hidden.get_mut()[..rows.len() * HIDDEN];
        for (row, embedding) in rows.clone().zip(input.as_chunks_mut::<HIDDEN>().0) {
            match image_row(prompt, row) {
                Some(token) => embedding.copy_from_slice(&tokens[token * HIDDEN..][..HIDDEN]),
                None => {
                    let table = language.embed.matrix();
                    table.dequantize_row(prompt.ids[row] as usize, embedding);
                }
            }
        }
        let call = Rows {
            first: rows.start,
            positions: &positions[rows.clone()],
        };
        let (state, buffers) = (&self.state, &self.buffers);
        self.team
            .run(|ctx| language.stack.forward(ctx, state, buffers, &call));
        let output = self.buffers.stack.hidden.read();
        let mut hidden = Vec::new();
        for &row in prompt.decisions.iter().filter(|row| rows.contains(row)) {
            let state = &output[(row - rows.start) * HIDDEN..][..HIDDEN];
            let mut normed = [0.0; HIDDEN];
            rms_norm(state, language.norm.values(), NORM_EPS, &mut normed);
            hidden.extend(normed);
        }
        hidden
    }

    /// Each question's logits over its options' symbols, from its deciding row's state.
    pub fn logits(&self, questions: &[Question], hidden: &[f32]) -> Vec<Vec<f32>> {
        let symbols = self.symbols.as_chunks::<HIDDEN>().0.iter();
        questions
            .iter()
            .zip(hidden.as_chunks::<HIDDEN>().0)
            .map(|(question, state)| {
                let options = symbols.clone().take(question.candidates.len());
                options.map(|symbol| dot_f32(state, symbol)).collect()
            })
            .collect()
    }

    /// Starts deciding `request` about RGB or RGBA `images`.
    pub fn start(&mut self, request: &Request, images: &[Pixels]) -> Result<Plan, Error> {
        self.job = None;
        if images.len() > MAX_IMAGES {
            return Err(Error::Request(format!(
                "{} images: at most {MAX_IMAGES}",
                images.len()
            )));
        }
        let patches = images
            .iter()
            .map(image::patches)
            .collect::<Result<Vec<_>, _>>()?;
        let grids: Vec<Grid> = patches.iter().map(|patches| patches.grid).collect();
        let (questions, prompt) = self.prompt(request, &grids)?;
        let plan = Plan {
            images: images
                .iter()
                .zip(&grids)
                .map(|(image, &grid)| PREPROCESSING.layout(image, grid))
                .collect(),
            image_tokens: prompt.images.iter().map(|(rows, _)| rows.len()).sum(),
            prompt_tokens: prompt.ids.len(),
            questions: questions.len(),
        };
        self.job = Some(Job {
            positions: prompt.positions(),
            questions,
            prompt,
            patches,
            seen: 0,
            tokens: Vec::new(),
            hidden: Vec::new(),
            read: 0,
        });
        Ok(plan)
    }

    /// Runs the next stage of the decision: the next image, or the next chunk of the prompt.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no decision in progress".into()))?;
        let rows = job.prompt.ids.len();
        if let Some(patches) = job.patches.get(job.seen) {
            let tokens = self.see(patches);
            job.tokens.extend(tokens);
            job.seen += 1;
            let progress = progress("merge", &job, None);
            self.job = Some(job);
            return Ok(progress);
        }
        let chunk = job.read..rows.min(job.read + PREFILL_ROWS);
        let sequence = Sequence {
            prompt: &job.prompt,
            positions: &job.positions,
            tokens: &job.tokens,
        };
        let hidden = self.read_rows(&sequence, chunk.clone());
        job.hidden.extend(hidden);
        job.read = chunk.end;
        if job.read < rows {
            let progress = progress("read", &job, None);
            self.job = Some(job);
            return Ok(progress);
        }
        let logits = self.logits(&job.questions, &job.hidden);
        let response = decision::respond(&job.questions, &logits, rows);
        Ok(progress("decide", &job, Some(response)))
    }
}

fn progress(stage: &'static str, job: &Job, response: Option<Response>) -> Progress {
    Progress {
        stage,
        read: job.read,
        rows: job.prompt.ids.len(),
        response,
    }
}

/// Row `row`'s index among the images' tokens, if it is one.
fn image_row(prompt: &Prompt, row: usize) -> Option<usize> {
    let mut before = 0;
    for (rows, _) in &prompt.images {
        if rows.contains(&row) {
            return Some(before + row - rows.start);
        }
        before += rows.len();
    }
    None
}
