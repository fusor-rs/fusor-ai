//! Answering a request: `start` checks it and reads its images and videos into patches; then
//! each `step` does one stage — an image or a video's group through the vision tower and the
//! merger, or one question: its prompt through the language model, every row seeing every
//! other in the attention layers (the delta rule's stay causal), and the last row read out over
//! the answer codes. The stages also run one by one, on given inputs, for checks against the
//! reference.

use ::media::image::Pixels;
use blocks::hybrid::{self, Rows};
use blocks::transformer::Mask;
use blocks::vit::preprocessing::Patches;
use blocks::vit::{self, Grid, merger, tower};
use bundle::{Error, Store};
use kernels::dot_f32;
use kernels::ops::{rms_norm, softmax};
use serde::Serialize;
use serde_json::{Map, Value};
use team::Team;

use crate::answer::{self, Failure, Response, Usage};
use crate::media::{self, IMAGE, Media, Video};
use crate::model::Model;
use crate::prompt::Prompt;
use crate::request::{Asked, Question, Request};

/// The longest prompt, in rows: a question's text and the request's images' and videos' tokens.
/// The reference has no limit; a longer prompt is answered `max_length_exceeded`.
pub const MAX_ROWS: usize = 4096;
pub const MAX_IMAGES: usize = 8;
pub const MAX_VIDEOS: usize = 4;

/// How a request is read: its images' and videos' patch grids and its questions.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub media: media::Plan,
    pub questions: usize,
}

/// One step done: the images, video groups and questions read so far of all, and the response
/// once every question is answered.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub done: usize,
    pub steps: usize,
    pub response: Option<Response>,
}

pub struct Engine {
    model: Model,
    team: Team,
    tower: tower::Buffers,
    merger: merger::Buffers,
    buffers: hybrid::Buffers,
    state: hybrid::State,
    job: Option<Job>,
}

struct Job {
    state: Value,
    questions: Vec<Asked>,
    media: Media,
    /// The tokens of the images and video groups seen so far, in prompt order, `tokens × hidden`.
    tokens: Vec<f32>,
    seen: usize,
    asked: usize,
    answers: Map<String, Value>,
    input_tokens: usize,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let size = model.size;
        let patches = IMAGE.max_patches();
        Ok(Self {
            tower: tower::Buffers::new(&size.tower(), patches, threads),
            merger: merger::Buffers::new(&size.merger(), patches),
            buffers: size.language().buffers(MAX_ROWS, threads),
            state: hybrid::State::new(&size.language(), MAX_ROWS),
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

impl Engine {
    /// The prompt of `question` about `state` and `media`.
    pub fn prompt(
        &self,
        state: &Value,
        question: &Question,
        media: &Media,
    ) -> Result<Prompt, Error> {
        let model = &self.model;
        Prompt::new(
            &model.tokenizer,
            state,
            question,
            &model.decision.codes,
            media,
        )
    }

    /// Each patch's embedding plus its position, `patches × width`.
    pub fn embed(&mut self, patches: &Patches) -> Vec<f32> {
        let positions = self.model.vision.positions(patches.grid);
        let (vision, buffers) = (&self.model.vision, &self.tower);
        self.team
            .run(|ctx| vision.embed(ctx, buffers, &patches.values, &positions));
        let len = patches.grid.patches() * self.model.size.vision().hidden;
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
        let len = IMAGE.tokens(grid.patches()) * self.model.size.hidden();
        self.merger.tokens.read()[..len].to_vec()
    }

    /// The prompt through the language model, the media's rows `tokens` (every image's and
    /// group's, in order): the last row's final-normed state.
    pub fn read(&mut self, prompt: &Prompt, tokens: &[f32]) -> Vec<f32> {
        let hidden = self.model.size.hidden();
        let language = &self.model.language;
        let rows = prompt.ids.len();
        let input = &mut self.buffers.stack.hidden.get_mut()[..rows * hidden];
        let mut media_tokens = tokens.chunks_exact(hidden);
        for (row, embedding) in input.chunks_exact_mut(hidden).enumerate() {
            let media = prompt.media.iter().any(|(range, _)| range.contains(&row));
            match media.then(|| media_tokens.next()).flatten() {
                Some(token) => embedding.copy_from_slice(token),
                None => language
                    .embed
                    .matrix()
                    .dequantize_row(prompt.ids[row] as usize, embedding),
            }
        }
        let positions = vit::positions(rows, &prompt.media);
        let call = Rows {
            first: 0,
            positions: &positions,
            mask: Mask::Full,
        };
        self.state.clear();
        let (state, buffers) = (&self.state, &self.buffers);
        self.team
            .run(|ctx| language.stack.forward(ctx, state, buffers, &call));
        let last = &self.buffers.stack.hidden.read()[(rows - 1) * hidden..rows * hidden];
        let mut normed = vec![0.0; hidden];
        rms_norm(
            last,
            language.norm.values(),
            self.model.size.language().norm_eps,
            &mut normed,
        );
        normed
    }

    /// The probabilities of the first `options` answer codes from the last row's state.
    pub fn probabilities(&self, normed: &[f32], options: usize) -> Vec<f32> {
        let readout = self.model.readout.matrix();
        let mut row = vec![0.0; normed.len()];
        let mut logits: Vec<f32> = (0..options)
            .map(|code| {
                readout.dequantize_row(code, &mut row);
                dot_f32(&row, normed) / self.model.decision.temperature
            })
            .collect();
        softmax(&mut logits);
        logits
    }

    /// Starts answering `request` about RGB or RGBA `images` and `videos`.
    pub fn start(
        &mut self,
        request: &Request,
        images: &[Pixels],
        videos: &[Video],
    ) -> Result<Plan, Error> {
        self.job = None;
        if images.len() > MAX_IMAGES || videos.len() > MAX_VIDEOS {
            return Err(Error::Request(format!(
                "{} images and {} videos: at most {MAX_IMAGES} and {MAX_VIDEOS}",
                images.len(),
                videos.len()
            )));
        }
        let questions = request.questions()?;
        let media = Media::new(images, videos)?;
        let plan = Plan {
            media: media.plan(),
            questions: questions.len(),
        };
        self.job = Some(Job {
            state: request.state.clone(),
            questions,
            media,
            tokens: Vec::new(),
            seen: 0,
            asked: 0,
            answers: Map::new(),
            input_tokens: 0,
        });
        Ok(plan)
    }

    /// Runs the next stage: the next image or video group, or the next question.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no request in progress".into()))?;
        let items = job.media.items().count();
        if job.seen < items {
            let patches = job
                .media
                .items()
                .nth(job.seen)
                .expect("an item is left to see");
            let tokens = self.see(patches);
            job.tokens.extend(tokens);
            job.seen += 1;
            let progress = progress("see", &job, None);
            self.job = Some(job);
            return Ok(progress);
        }
        self.ask(&mut job);
        job.asked += 1;
        if job.asked < job.questions.len() {
            let progress = progress("ask", &job, None);
            self.job = Some(job);
            return Ok(progress);
        }
        let response = Response {
            model: format!("d3-{}", self.model.size.name()),
            answers: std::mem::take(&mut job.answers),
            usage: Usage {
                input_tokens: job.input_tokens,
                output_tokens: 0,
            },
        };
        Ok(progress("answer", &job, Some(response)))
    }

    /// Answers the job's next question, or records why it cannot be answered.
    fn ask(&mut self, job: &mut Job) {
        let Asked { name, question } = &job.questions[job.asked];
        let answer = match question {
            Err(invalid) => {
                answer::failed(&invalid.kind, Failure::InvalidQuestion, &invalid.message)
            }
            Ok(question) => match self.prompt(&job.state, question, &job.media) {
                Err(error) => {
                    let message = error.to_string();
                    answer::failed(&question.kind.into(), Failure::InvalidQuestion, &message)
                }
                Ok(prompt) if prompt.ids.len() > MAX_ROWS => {
                    let message = format!(
                        "the question prompt has {} tokens, over the {MAX_ROWS} this runtime \
                         reads at once; nothing was truncated",
                        prompt.ids.len()
                    );
                    answer::failed(&question.kind.into(), Failure::MaxLengthExceeded, &message)
                }
                Ok(prompt) => {
                    job.input_tokens += prompt.ids.len();
                    let normed = self.read(&prompt, &job.tokens);
                    answer::answer(question, &self.probabilities(&normed, question.keys.len()))
                }
            },
        };
        job.answers.insert(name.clone(), answer);
    }
}

fn progress(stage: &'static str, job: &Job, response: Option<Response>) -> Progress {
    Progress {
        stage,
        done: job.seen + job.asked,
        steps: job.media.items().count() + job.questions.len(),
        response,
    }
}
