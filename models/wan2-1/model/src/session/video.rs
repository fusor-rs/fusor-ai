//! The video engine: `start` conditions the transformer on the states of a request's prompt
//! and negative prompt and draws its starting latents; each `step` then runs one denoising step
//! (the two branches of classifier-free guidance as two transformer calls), and the steps after
//! the last decode the video a latent frame at a time. The stages also run one at a time, on
//! given inputs (`condition`, `predict`, `sample`, `decode`), to check and time them.

use blocks::dpm::{Schedule, Solver};
use blocks::video::Patching;
use blocks::{wan_dit, wan_vae};
use bundle::Store;
use kernels::conv::Grid;
use kernels::rng::Rng;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::model::{
    DIT_SHAPE, FRAME_STEP, IMAGE_CHANNELS, LATENT_CHANNELS, LATENT_FACTOR, TEXT_SHAPE, TEXT_TOKENS,
    TRAINING_TIMESTEPS, VAE_SHAPE, VideoModel,
};
use crate::{Error, Video};
use media::image::rgba_from_signed;

/// A video's sides are multiples of `SIDE_STEP` (the latents' 8 times the patch's 2) from
/// `MIN_SIDE` to `MAX_SIDE`, its area at most `MAX_PIXELS`; its frames `FRAME_STEP·n + 1`, at
/// most `MAX_FRAMES`. The limits keep the decoder's frame histories and every activation, with
/// the bundle, inside a wasm32 memory (4 GiB).
const SIDE_STEP: usize = 16;
const MIN_SIDE: usize = 64;
const MAX_SIDE: usize = 512;
const MAX_PIXELS: usize = 256 * 256;
const MAX_FRAMES: usize = 33;
const MAX_STEPS: usize = 100;
const MAX_GUIDANCE: f32 = 20.0;
const MAX_SHIFT: f32 = 20.0;
const MAX_LATENT_PIXELS: usize = MAX_PIXELS / (LATENT_FACTOR * LATENT_FACTOR);
const MAX_TOKENS: usize =
    latent_frames(MAX_FRAMES) * MAX_LATENT_PIXELS / (DIT_SHAPE.patch * DIT_SHAPE.patch);
const BRANCHES: usize = 2;
/// The decoder decodes whole frames, one strip each: no frame is wider or taller than its pixels.
const DECODER: wan_vae::Capacity = wan_vae::Capacity {
    pixels: MAX_LATENT_PIXELS,
    width: MAX_LATENT_PIXELS,
    video: MAX_TOKENS * DIT_SHAPE.patch * DIT_SHAPE.patch,
    strip: MAX_LATENT_PIXELS,
};

/// What to generate.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: usize,
    /// The classifier-free guidance scale, at least 1; at 1 only the prompt's branch runs.
    pub guidance: f32,
    pub shift: f32,
    pub seed: u64,
}

impl Request {
    /// The latents of the requested video, if the request is within the limits.
    pub fn video(&self) -> Result<Video, Error> {
        check(self)?;
        Ok(Video {
            frames: latent_frames(self.frames),
            height: self.height / LATENT_FACTOR,
            width: self.width / LATENT_FACTOR,
        })
    }
}

/// A started generation.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub steps: usize,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Denoising steps done.
    pub step: usize,
    pub steps: usize,
    /// Video frames decoded so far, and in all.
    pub decoded: usize,
    pub frames: usize,
    /// Every frame is decoded (`take_frames`).
    pub done: bool,
}

/// A branch of classifier-free guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Prompt,
    Negative,
}

pub struct VideoEngine {
    model: VideoModel,
    team: Team,
    dit: wan_dit::Buffers,
    /// The cross-attention keys and values of each branch's text, indexed by `Branch`.
    contexts: [wan_dit::Context; BRANCHES],
    conditioned: [bool; BRANCHES],
    decoder: wan_vae::Buffers,
    job: Option<Job>,
}

struct Job {
    video: Video,
    schedule: Schedule,
    solver: Solver,
    guidance: Option<f32>,
    latents: Vec<f32>,
    /// The guided velocity of the last step, and the clean latents it predicts.
    velocity: Vec<f32>,
    denoised: Vec<f32>,
    step: usize,
    /// Latent frames decoded.
    decoded: usize,
    /// RGBA of the frames decoded since `take_frames`.
    frames: Vec<u8>,
}

/// Latent frames of a video of `frames` frames.
const fn latent_frames(frames: usize) -> usize {
    (frames - 1) / FRAME_STEP + 1
}

impl bundle::Engine for VideoEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = VideoModel::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        Ok(Self {
            dit: DIT_SHAPE.buffers(MAX_TOKENS, threads),
            contexts: std::array::from_fn(|_| DIT_SHAPE.context()),
            conditioned: [false; BRANCHES],
            decoder: VAE_SHAPE.decoder_buffers(DECODER, threads),
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

impl VideoEngine {
    /// Conditions `branch` on `text`: umT5's states, rows of `TEXT_SHAPE.hidden`, at most
    /// `TEXT_TOKENS` of them.
    pub fn condition(&mut self, branch: Branch, text: &[f32]) -> Result<(), Error> {
        let hidden = TEXT_SHAPE.hidden;
        let rows = text.len() / hidden;
        if text.is_empty() || !text.len().is_multiple_of(hidden) || rows > TEXT_TOKENS {
            return Err(Error::Request(format!(
                "{} text values: 1 to {TEXT_TOKENS} rows of {hidden}",
                text.len()
            )));
        }
        let Self {
            model,
            team,
            dit,
            contexts,
            ..
        } = self;
        let context = &contexts[branch as usize];
        team.run(|ctx| model.dit.condition(ctx, dit, text, context));
        self.conditioned[branch as usize] = true;
        Ok(())
    }

    /// The transformer's velocity for `latents` (`video`) at `timestep` under `branch`'s text.
    pub fn predict(
        &mut self,
        (latents, video): (&[f32], Video),
        timestep: f32,
        branch: Branch,
    ) -> Result<Vec<f32>, Error> {
        check_latents(latents, video)?;
        self.check_conditioned(branch)?;
        self.dit.latents.get_mut()[..latents.len()].copy_from_slice(latents);
        self.set_timestep(timestep);
        self.run_dit(video, branch);
        Ok(self.dit.velocity.read()[..latents.len()].to_vec())
    }

    /// The noise levels and timesteps of a run of `steps` steps.
    pub fn schedule(&self, steps: usize, shift: f32) -> Schedule {
        Schedule::flow(steps, f64::from(shift), TRAINING_TIMESTEPS)
    }

    /// The latents after `steps` denoising steps from `latents` (`video`), guided at `guidance`
    /// between the conditioned branches.
    pub fn sample(
        &mut self,
        (latents, video): (&[f32], Video),
        (steps, guidance, shift): (usize, f32, f32),
    ) -> Result<Vec<f32>, Error> {
        check_latents(latents, video)?;
        let guided = guidance > 1.0;
        self.check_conditioned(Branch::Prompt)?;
        if guided {
            self.check_conditioned(Branch::Negative)?;
        }
        let mut job = self.begin(
            latents.to_vec(),
            video,
            (steps, guided.then_some(guidance), shift),
        );
        while job.step < steps {
            self.denoise(&mut job);
        }
        Ok(job.latents)
    }

    /// The video of `latents` (`video`): pixel-major RGB in `[−1, 1]`, frame after frame.
    pub fn decode(&mut self, (latents, video): (&[f32], Video)) -> Result<Vec<f32>, Error> {
        check_latents(latents, video)?;
        let mut frames = Vec::new();
        for index in 0..video.frames {
            let decoded = self.decode_frame(latents, video, index);
            frames.extend(decoded.iter().map(|value| value.clamp(-1.0, 1.0)));
        }
        Ok(frames)
    }

    /// Starts generating `request` from the states of its prompt and of its negative prompt
    /// (the latter read only when the guidance is above 1).
    pub fn start(
        &mut self,
        request: &Request,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<Plan, Error> {
        let video = request.video()?;
        self.job = None;
        self.condition(Branch::Prompt, prompt)?;
        let guidance = (request.guidance > 1.0).then_some(request.guidance);
        if guidance.is_some() {
            self.condition(Branch::Negative, negative)?;
        }
        let mut latents = vec![0.0; video.pixels() * LATENT_CHANNELS];
        Rng::new(request.seed).fill_normal(&mut latents);
        let steps = (request.steps, guidance, request.shift);
        self.job = Some(self.begin(latents, video, steps));
        Ok(Plan {
            steps: request.steps,
            width: request.width,
            height: request.height,
            frames: request.frames,
        })
    }

    /// Runs the next denoising step, or decodes the next latent frame once the steps are done.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self.job.take().ok_or_else(no_generation)?;
        let steps = job.schedule.steps();
        if job.step < steps {
            self.denoise(&mut job);
        } else if job.decoded < job.video.frames {
            self.decode_next(&mut job);
        }
        let progress = Progress {
            step: job.step,
            steps,
            decoded: (0..job.decoded).map(|index| VAE_SHAPE.frames(index)).sum(),
            frames: (0..job.video.frames)
                .map(|index| VAE_SHAPE.frames(index))
                .sum(),
            done: job.decoded == job.video.frames,
        };
        self.job = Some(job);
        Ok(progress)
    }

    /// The first frame of the clean video the last step predicts, at full resolution: RGBA rows
    /// top to bottom. Between the first step and the decoding.
    pub fn preview(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.take().ok_or_else(no_generation)?;
        let first = (job.step > 0 && job.decoded == 0)
            .then(|| rgba_from_signed(self.decode_frame(&job.denoised, job.video, 0)));
        self.job = Some(job);
        first.ok_or_else(|| {
            Error::Request("a preview exists between the first step and decoding".into())
        })
    }

    /// The RGBA bytes of the frames decoded since the last call, one after another.
    pub fn take_frames(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.as_mut().ok_or_else(no_generation)?;
        Ok(std::mem::take(&mut job.frames))
    }

    fn check_conditioned(&self, branch: Branch) -> Result<(), Error> {
        if self.conditioned[branch as usize] {
            return Ok(());
        }
        Err(Error::Request(format!(
            "the {branch:?} branch has no text yet"
        )))
    }

    /// A generation from `latents` (`video`), `steps` steps, `guidance` (when guided) and `shift`.
    fn begin(
        &self,
        latents: Vec<f32>,
        video: Video,
        (steps, guidance, shift): (usize, Option<f32>, f32),
    ) -> Job {
        let len = latents.len();
        Job {
            video,
            schedule: self.schedule(steps, shift),
            solver: Solver::default(),
            guidance,
            latents,
            velocity: vec![0.0; len],
            denoised: vec![0.0; len],
            step: 0,
            decoded: 0,
            frames: Vec::new(),
        }
    }

    /// The transformer's modulations at `timestep`, which every branch of the step shares.
    fn set_timestep(&self, timestep: f32) {
        let Self {
            model, team, dit, ..
        } = self;
        team.run(|ctx| model.dit.time(ctx, dit, timestep));
    }

    /// The transformer over `self.dit.latents` (`video`) under `branch`'s text at the timestep
    /// set, into `self.dit.velocity`.
    fn run_dit(&self, video: Video, branch: Branch) {
        let Self {
            model,
            team,
            dit,
            contexts,
            ..
        } = self;
        let context = &contexts[branch as usize];
        team.run(|ctx| model.dit.forward(ctx, dit, context, video));
    }

    /// One step of `job`: the prompt's velocity, and with guidance the negative prompt's, guided
    /// as `u + g(c − u)`, then the solver.
    fn denoise(&mut self, job: &mut Job) {
        let len = job.latents.len();
        let timestep = job.schedule.timestep(job.step);
        self.dit.latents.get_mut()[..len].copy_from_slice(&job.latents);
        self.set_timestep(timestep);
        self.run_dit(job.video, Branch::Prompt);
        job.velocity
            .copy_from_slice(&self.dit.velocity.read()[..len]);
        if let Some(scale) = job.guidance {
            self.run_dit(job.video, Branch::Negative);
            let negative = &self.dit.velocity.read()[..len];
            for (velocity, unconditional) in job.velocity.iter_mut().zip(negative) {
                *velocity = unconditional + scale * (*velocity - unconditional);
            }
        }
        let at = (&job.schedule, job.step);
        job.solver
            .step(at, &mut job.latents, &job.velocity, &mut job.denoised);
        job.step += 1;
    }

    /// Latent frame `job.decoded` of `job.latents` through the decoder, its frames' RGBA onto
    /// `job.frames`.
    fn decode_next(&mut self, job: &mut Job) {
        let decoded = self.decode_frame(&job.latents, job.video, job.decoded);
        job.frames.extend(rgba_from_signed(decoded));
        job.decoded += 1;
    }

    /// Latent frame `index` of `latents` (`video`) through the decoder: the pixel-major RGB of
    /// its frames, unclamped.
    fn decode_frame(&mut self, latents: &[f32], video: Video, index: usize) -> &[f32] {
        let grid = Grid {
            images: 1,
            height: video.height,
            width: video.width,
        };
        let values = grid.pixels() * LATENT_CHANNELS;
        let frame = &latents[index * values..(index + 1) * values];
        self.decoder.latent.get_mut()[..values].copy_from_slice(frame);
        let Self {
            model,
            team,
            decoder,
            ..
        } = self;
        team.run(|ctx| {
            model.decoder.decode_latents(ctx, decoder, (grid, index));
            model
                .decoder
                .decode_strip(ctx, decoder, (grid, index), 0..grid.height);
        });
        let pixels = model.decoder.output_grid(grid).pixels() * VAE_SHAPE.frames(index);
        &self.decoder.frames.read()[..pixels * IMAGE_CHANNELS]
    }
}

fn no_generation() -> Error {
    Error::Request("no generation in progress".into())
}

/// Latents of `video` the buffers hold.
fn check_latents(latents: &[f32], video: Video) -> Result<(), Error> {
    let latent_step = SIDE_STEP / LATENT_FACTOR;
    let sides = video.height.is_multiple_of(latent_step) && video.width.is_multiple_of(latent_step);
    let patching = Patching {
        video,
        patch: DIT_SHAPE.patch,
    };
    let tokens = patching.tokens();
    if sides
        && video.frames > 0
        && video.height > 0
        && video.width > 0
        && video.height * video.width <= MAX_LATENT_PIXELS
        && tokens <= MAX_TOKENS
        && latents.len() == video.pixels() * LATENT_CHANNELS
    {
        return Ok(());
    }
    Err(Error::Request(format!(
        "{} latent values for {} frames of {}×{}: sides multiples of {latent_step}, at most \
         {MAX_LATENT_PIXELS} pixels a frame and {MAX_TOKENS} tokens",
        latents.len(),
        video.frames,
        video.width,
        video.height
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
    if !(1..=MAX_FRAMES).contains(&request.frames)
        || !(request.frames - 1).is_multiple_of(FRAME_STEP)
    {
        return Err(Error::Request(format!(
            "{} frames: {FRAME_STEP}·n + 1, up to {MAX_FRAMES}",
            request.frames
        )));
    }
    if !(1..=MAX_STEPS).contains(&request.steps) {
        return Err(Error::Request(format!(
            "{} steps: 1 to {MAX_STEPS}",
            request.steps
        )));
    }
    if !(1.0..=MAX_GUIDANCE).contains(&request.guidance) {
        return Err(Error::Request(format!(
            "guidance {}: 1 (the prompt alone) to {MAX_GUIDANCE}",
            request.guidance
        )));
    }
    if !(request.shift > 0.0 && request.shift <= MAX_SHIFT) {
        return Err(Error::Request(format!(
            "shift {}: above 0 to {MAX_SHIFT}",
            request.shift
        )));
    }
    Ok(())
}
