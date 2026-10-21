//! Generation: `start` encodes a request's prompts once and draws its starting latents; each
//! `step` then runs one denoising step (both branches of classifier-free guidance as one UNet
//! call), and the step after the last decodes the image. The stages also run one at a time, on
//! given inputs (`encode_text`, `predict_noise`, `decode`), to check and time them.

use blocks::dpm::{NoiseSchedule, Schedule, Solver};
use blocks::{unet, vae};
use bundle::{Store, Tokenizer};
use kernels::conv::Grid;
use kernels::rng::Rng;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::Error;
use crate::model::{
    BETA_END, BETA_START, IMAGE_CHANNELS, LATENT_CHANNELS, LATENT_FACTOR, Model, TEXT_SHAPE,
    TOKENS, TRAINING_TIMESTEPS, UNET_SHAPE, VAE_SHAPE, VOCABULARY,
};
use crate::picture;
use crate::prompt::{Tokens, tokenize};
use crate::text::Activations;

/// An image's sides are multiples of `SIDE_STEP` (the latents', a multiple of the UNet's
/// three halvings) from `MIN_SIDE` to `MAX_SIDE`, its area at most `MAX_PIXELS`.
pub const SIDE_STEP: usize = 64;
pub const MIN_SIDE: usize = 256;
pub const MAX_SIDE: usize = 768;
pub const MAX_PIXELS: usize = 512 * 768;
pub const MAX_STEPS: usize = 100;
pub const MAX_GUIDANCE: f32 = 20.0;
/// The UNet's images per call: the negative prompt's branch, then the prompt's.
const BRANCHES: usize = 2;
const CONTEXT: usize = TOKENS * TEXT_SHAPE.hidden;

/// What to generate.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub prompt: String,
    /// What the image should not show: the prompt of guidance's unconditional branch.
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// The classifier-free guidance scale; at 1 or below only the prompt's branch runs.
    pub guidance: f32,
    pub seed: u64,
}

/// A started generation.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub steps: usize,
    pub width: usize,
    pub height: usize,
    /// The prompt's and the negative prompt's tokens past what CLIP reads, which it ignores.
    pub dropped: [usize; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Denoising steps done.
    pub step: usize,
    pub steps: usize,
    /// The image is decoded (`take_image`).
    pub done: bool,
}

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    team: Team,
    noise: NoiseSchedule,
    text: Activations,
    unet: unet::Buffers,
    context: unet::Context,
    decoder: vae::Buffers,
    job: Option<Job>,
}

struct Job {
    /// The latents' grid: one image.
    grid: Grid,
    schedule: Schedule,
    solver: Solver,
    guidance: Option<f32>,
    latents: Vec<f32>,
    /// The guided noise of the last step, and the clean latents it predicts.
    noise: Vec<f32>,
    denoised: Vec<f32>,
    step: usize,
    /// RGBA, once decoded.
    image: Option<Vec<u8>>,
}

impl Engine {
    pub fn new(store: &mut Store) -> Result<Self, Error> {
        store.require_complete()?;
        let model = Model::load(store)?;
        let tokenizer = Tokenizer::take(store)?;
        let team = Team::global();
        let threads = team.threads();
        let pixels = MAX_PIXELS / (LATENT_FACTOR * LATENT_FACTOR);
        Ok(Self {
            tokenizer,
            noise: NoiseSchedule::scaled_linear(BETA_START, BETA_END, TRAINING_TIMESTEPS),
            text: Activations::new(threads),
            unet: UNET_SHAPE.buffers(BRANCHES, pixels, threads),
            context: model.unet.context(BRANCHES, TOKENS),
            decoder: VAE_SHAPE.buffers(pixels, threads),
            model,
            team,
            job: None,
        })
    }

    pub fn threads(&self) -> usize {
        self.team.threads()
    }

    pub fn tokens(&self, text: &str) -> Result<Tokens, Error> {
        tokenize(&self.tokenizer, text)
    }

    /// CLIP's last hidden states of `tokens` (`TOKENS` of them), `TOKENS × 768`.
    pub fn encode_text(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        if tokens.len() != TOKENS || tokens.iter().any(|&token| token as usize >= VOCABULARY) {
            return Err(Error::Request(format!(
                "{} tokens: {TOKENS} below {VOCABULARY} expected",
                tokens.len()
            )));
        }
        let mut states = vec![0.0; CONTEXT];
        let Self {
            model, team, text, ..
        } = self;
        model.text.encode(team, text, tokens, &mut states);
        Ok(states)
    }

    /// The UNet's noise in `latents` (pixel-major images of `grid`) at `timestep`, each image
    /// conditioned on its `TOKENS × 768` rows of `text`.
    pub fn predict_noise(
        &mut self,
        (latents, grid): (&[f32], Grid),
        timestep: f32,
        text: &[f32],
    ) -> Result<Vec<f32>, Error> {
        check_latents(grid, BRANCHES)?;
        let len = grid.pixels() * LATENT_CHANNELS;
        if latents.len() != len || text.len() != grid.images * CONTEXT {
            return Err(Error::Request(format!(
                "{} latent and {} text values for {} images of {}×{}",
                latents.len(),
                text.len(),
                grid.images,
                grid.width,
                grid.height
            )));
        }
        self.condition(text);
        self.unet.sample.get_mut()[..len].copy_from_slice(latents);
        self.run_unet(grid, timestep);
        Ok(self.unet.prediction.read()[..len].to_vec())
    }

    /// The image of `latents` (one image of `grid`), pixel-major RGB in `[−1, 1]`.
    pub fn decode(&mut self, (latents, grid): (&[f32], Grid)) -> Result<Vec<f32>, Error> {
        check_latents(grid, 1)?;
        if latents.len() != grid.pixels() * LATENT_CHANNELS {
            return Err(Error::Request(format!(
                "{} latent values for {}×{}",
                latents.len(),
                grid.width,
                grid.height
            )));
        }
        Ok(self.run_decoder(latents, grid).to_vec())
    }

    /// The noise levels and timesteps of a run of `steps` steps.
    pub fn schedule(&self, steps: usize) -> Schedule {
        Schedule::karras(&self.noise, steps)
    }

    /// Starts generating `request`.
    pub fn start(&mut self, request: &Request) -> Result<Plan, Error> {
        check(request)?;
        self.job = None;
        let prompt = self.tokens(&request.prompt)?;
        let negative = self.tokens(&request.negative)?;
        let guidance = (request.guidance > 1.0).then_some(request.guidance);
        let mut text = Vec::with_capacity(BRANCHES * CONTEXT);
        if guidance.is_some() {
            text.extend(self.encode_text(&negative.ids)?);
        }
        text.extend(self.encode_text(&prompt.ids)?);
        let grid = Grid {
            images: 1,
            height: request.height / LATENT_FACTOR,
            width: request.width / LATENT_FACTOR,
        };
        let mut latents = vec![0.0; grid.pixels() * LATENT_CHANNELS];
        Rng::new(request.seed).fill_normal(&mut latents);
        self.begin(&text, (latents, grid), (request.steps, guidance));
        Ok(Plan {
            steps: request.steps,
            width: request.width,
            height: request.height,
            dropped: [prompt.dropped, negative.dropped],
        })
    }

    /// The latents after `steps` denoising steps from `latents` (one image of `grid`), guided
    /// at `guidance` between the two images' `TOKENS × 768` rows of `text` (the negative
    /// prompt's, then the prompt's).
    pub fn sample(
        &mut self,
        text: &[f32],
        (latents, grid): (&[f32], Grid),
        (steps, guidance): (usize, f32),
    ) -> Result<Vec<f32>, Error> {
        check_latents(grid, 1)?;
        if latents.len() != grid.pixels() * LATENT_CHANNELS || text.len() != BRANCHES * CONTEXT {
            return Err(Error::Request(format!(
                "{} latent and {} text values: one image of {}×{} and two prompts expected",
                latents.len(),
                text.len(),
                grid.width,
                grid.height
            )));
        }
        self.begin(text, (latents.to_vec(), grid), (steps, Some(guidance)));
        let mut job = self.job.take().ok_or_else(no_generation)?;
        while job.step < steps {
            self.denoise(&mut job);
        }
        Ok(job.latents)
    }

    /// A generation from `latents` conditioned on `text` (one context per branch).
    fn begin(
        &mut self,
        text: &[f32],
        (latents, grid): (Vec<f32>, Grid),
        (steps, guidance): (usize, Option<f32>),
    ) {
        self.condition(text);
        let len = latents.len();
        self.job = Some(Job {
            grid,
            schedule: self.schedule(steps),
            solver: Solver::default(),
            guidance,
            latents,
            noise: vec![0.0; len],
            denoised: vec![0.0; len],
            step: 0,
            image: None,
        });
    }

    /// Runs the next denoising step, or decodes the image after the last.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self.job.take().ok_or_else(no_generation)?;
        let steps = job.schedule.steps();
        if job.step < steps {
            self.denoise(&mut job);
        } else if job.image.is_none() {
            let image = self.run_decoder(&job.latents, job.grid);
            job.image = Some(picture::rgba(image));
        }
        let progress = Progress {
            step: job.step,
            steps,
            done: job.image.is_some(),
        };
        self.job = Some(job);
        Ok(progress)
    }

    /// A preview of the clean image the last step predicts, RGBA rows top to bottom at the
    /// latents' resolution (`LATENT_FACTOR` times smaller along each side).
    pub fn preview(&self) -> Result<Vec<u8>, Error> {
        let job = self.job.as_ref().filter(|job| job.step > 0);
        Ok(picture::preview(&job.ok_or_else(no_generation)?.denoised))
    }

    /// The decoded image, RGBA rows top to bottom; ends the generation.
    pub fn take_image(&mut self) -> Result<Vec<u8>, Error> {
        let image = self.job.as_mut().and_then(|job| job.image.take());
        let image = image.ok_or_else(no_generation)?;
        self.job = None;
        Ok(image)
    }

    pub fn cancel(&mut self) {
        self.job = None;
    }

    /// Every cross-attention's keys and values of `text`, one context per image.
    fn condition(&mut self, text: &[f32]) {
        let Self {
            model,
            team,
            unet,
            context,
            ..
        } = self;
        team.run(|ctx| model.unet.encode(ctx, unet, text, context));
    }

    /// The UNet over `self.unet.sample` (`grid`) at `timestep`, into `self.unet.prediction`.
    fn run_unet(&mut self, grid: Grid, timestep: f32) {
        let Self {
            model,
            team,
            unet,
            context,
            ..
        } = self;
        team.run(|ctx| model.unet.forward(ctx, unet, context, (grid, timestep)));
    }

    /// One step of `job`: both branches' noise in one UNet call, guided, then the solver.
    fn denoise(&mut self, job: &mut Job) {
        let branches = if job.guidance.is_some() { BRANCHES } else { 1 };
        let len = job.latents.len();
        let sample = self.unet.sample.get_mut();
        for image in sample[..branches * len].chunks_exact_mut(len) {
            image.copy_from_slice(&job.latents);
        }
        let grid = Grid {
            images: branches,
            ..job.grid
        };
        self.run_unet(grid, job.schedule.timestep(job.step));
        let prediction = &self.unet.prediction.read()[..branches * len];
        let (first, last) = prediction.split_at(len);
        match job.guidance {
            Some(scale) => {
                let pairs = job.noise.iter_mut().zip(first.iter().zip(last));
                for (noise, (unconditional, conditional)) in pairs {
                    *noise = unconditional + scale * (conditional - unconditional);
                }
            }
            None => job.noise.copy_from_slice(first),
        }
        let at = (&job.schedule, job.step);
        job.solver
            .step(at, &mut job.latents, &job.noise, &mut job.denoised);
        job.step += 1;
    }

    /// The decoder over `latents` (one image of `grid`): pixel-major RGB in `[−1, 1]`.
    fn run_decoder(&mut self, latents: &[f32], grid: Grid) -> &[f32] {
        self.decoder.latents.get_mut()[..latents.len()].copy_from_slice(latents);
        let Self {
            model,
            team,
            decoder,
            ..
        } = self;
        team.run(|ctx| model.decoder.decode(ctx, decoder, grid));
        let pixels = self.model.decoder.output_grid(grid).pixels();
        &self.decoder.image.read()[..pixels * IMAGE_CHANNELS]
    }
}

fn no_generation() -> Error {
    Error::Request("no generation in progress".into())
}

/// Latents of `grid` the buffers hold, at most `images` of them.
fn check_latents(grid: Grid, images: usize) -> Result<(), Error> {
    let latent_step = SIDE_STEP / LATENT_FACTOR;
    let sides = grid.height.is_multiple_of(latent_step) && grid.width.is_multiple_of(latent_step);
    let pixels = MAX_PIXELS / (LATENT_FACTOR * LATENT_FACTOR);
    if (1..=images).contains(&grid.images) && sides && grid.area() <= pixels {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{} latent images of {}×{}: at most {images}, sides multiples of {latent_step}, at most \
         {pixels} pixels",
        grid.images, grid.width, grid.height
    )))
}

fn check(request: &Request) -> Result<(), Error> {
    let side =
        |length: usize| (MIN_SIDE..=MAX_SIDE).contains(&length) && length.is_multiple_of(SIDE_STEP);
    if !side(request.width) || !side(request.height) || request.width * request.height > MAX_PIXELS
    {
        return Err(Error::Request(format!(
            "{}×{}: sides are multiples of {SIDE_STEP} from {MIN_SIDE} to {MAX_SIDE}, at most \
             {MAX_PIXELS} pixels",
            request.width, request.height
        )));
    }
    if !(1..=MAX_STEPS).contains(&request.steps) {
        return Err(Error::Request(format!(
            "{} steps: 1 to {MAX_STEPS}",
            request.steps
        )));
    }
    if !(0.0..=MAX_GUIDANCE).contains(&request.guidance) {
        return Err(Error::Request(format!(
            "guidance {}: 0 to {MAX_GUIDANCE}",
            request.guidance
        )));
    }
    Ok(())
}
