//! The video engine: `start` conditions the transformer on the states of a request's prompt and
//! negative prompt (and of its first image, when it has one), encodes the first frame's latents
//! and draws the starting latents; each `step` then runs one denoising step (the two branches of
//! guidance as two transformer calls), and the steps after the last decode the video a strip of
//! rows at a time. The stages also run one at a time, on given inputs (`condition`, `predict`,
//! `sample`, `decode`), to check and time them.

use std::ops::Range;

use blocks::apg::{Guidance, Settings};
use blocks::dpm::{Schedule, Solver};
use blocks::rotary::Position;
use blocks::video::Patching;
use blocks::wan_vae;
use bundle::Store;
use kernels::conv::Grid;
use kernels::rng::Rng;
use media::image::rgba_from_signed;
use serde::Serialize;
use team::Team;

use super::request::{
    DECODER, MAX_LATENT_PIXELS, MAX_TOKENS, MIN_IMAGE_FRAMES, Request, check_size, check_values,
};
use crate::model::{
    CONDITION_CHANNELS, GUIDANCE, IMAGE_CHANNELS, IMAGE_HIDDEN, IMAGE_TOKENS, INPUT_CHANNELS,
    LATENT_CHANNELS, LATENT_FACTOR, LATENT_MEAN, LATENT_STD, LOG_VARIANCE, PATCH, SHIFT,
    TEXT_HIDDEN, TEXT_TOKENS, TRAINING_TIMESTEPS, VAE_SHAPE, VideoModel,
};
use crate::transformer::{self, BRANCHES, Call, Contexts};
use crate::{Error, Size, Video};

/// Values of a latent pixel's moments: its means, then its log-variances.
const MOMENTS: usize = 2 * LATENT_CHANNELS;
/// Bytes of a frame's pixel (`rgba_from_signed`'s).
const RGBA: usize = 4;

/// A started generation.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub steps: usize,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    /// The video starts from a first image.
    pub image: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Denoising steps done.
    pub step: usize,
    pub steps: usize,
    /// Strips decoded so far, and in all: the decoder decodes every frame a band of rows at a
    /// time.
    pub decoded: usize,
    pub strips: usize,
    /// Every frame is decoded (`take_frames`).
    pub done: bool,
}

/// A branch of guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Prompt,
    Negative,
}

/// A first image as the text engine gives it: SigLIP's states and the frame
/// (`image::first_frame`'s).
pub struct FirstImage {
    pub states: Vec<f32>,
    pub frame: Vec<f32>,
}

/// The activations of one stage at a time: at `WIDTH × HEIGHT` the transformer's and the
/// autoencoder's do not fit a wasm32 memory beside each other and the bundle.
enum Activations {
    /// Between two stages: one stage's are dropped before the next one's are made.
    Released,
    Transformer(Box<transformer::Buffers>),
    Encoder(wan_vae::Buffers),
    Decoder(wan_vae::Buffers),
}

impl Activations {
    /// The activations `held` picks, made by `make` in place of the others' when not held.
    fn switch<T>(
        &mut self,
        held: fn(&mut Self) -> Option<&mut T>,
        make: impl FnOnce() -> Self,
    ) -> &mut T {
        if held(self).is_none() {
            *self = Self::Released;
            *self = make();
        }
        held(self).expect("`make` makes the activations `held` picks")
    }

    fn transformer(&mut self, threads: usize) -> &mut transformer::Buffers {
        self.switch(
            |held| match held {
                Self::Transformer(buffers) => Some(buffers),
                _ => None,
            },
            || Self::Transformer(Box::new(transformer::Buffers::new(MAX_TOKENS, threads))),
        )
    }

    fn encoder(&mut self, threads: usize) -> &mut wan_vae::Buffers {
        self.switch(
            |held| match held {
                Self::Encoder(buffers) => Some(buffers),
                _ => None,
            },
            || Self::Encoder(VAE_SHAPE.encoder_buffers(MAX_LATENT_PIXELS, threads)),
        )
    }

    fn decoder(&mut self, threads: usize) -> &mut wan_vae::Buffers {
        self.switch(
            |held| match held {
                Self::Decoder(buffers) => Some(buffers),
                _ => None,
            },
            || Self::Decoder(VAE_SHAPE.decoder_buffers(DECODER, threads)),
        )
    }
}

pub struct VideoEngine {
    model: VideoModel,
    team: Team,
    activations: Activations,
    contexts: Contexts,
    conditioned: [bool; BRANCHES],
    /// The first image of the next `start`.
    image: Option<FirstImage>,
    job: Option<Job>,
}

struct Job {
    video: Video,
    schedule: Schedule,
    solver: Solver,
    guidance: Option<Guidance>,
    latents: Vec<f32>,
    conditioning: Vec<f32>,
    /// The step's velocity under each branch, and guided.
    conditional: Vec<f32>,
    unconditional: Vec<f32>,
    guided: Vec<f32>,
    /// The clean latents the last step predicts.
    denoised: Vec<f32>,
    step: usize,
    /// Strips decoded.
    decoded: usize,
    /// RGBA of every frame, as the strips decoded fill it in.
    frames: Vec<u8>,
}

/// Each latent pixel's conditioning (`CONDITION_CHANNELS`): `first`'s latents (one latent frame,
/// pixel-major) and a mask of 1 at latent frame 0, zeros elsewhere; zeros everywhere without
/// `first`.
pub fn conditioning(first: Option<&[f32]>, video: Video) -> Vec<f32> {
    let mut conditioning = vec![0.0; video.pixels() * CONDITION_CHANNELS];
    let Some(first) = first else {
        return conditioning;
    };
    let pixels = conditioning.as_chunks_mut::<CONDITION_CHANNELS>().0;
    for (pixel, latents) in pixels
        .iter_mut()
        .zip(first.as_chunks::<LATENT_CHANNELS>().0)
    {
        let (target, mask) = pixel.split_at_mut(LATENT_CHANNELS);
        target.copy_from_slice(latents);
        mask.fill(1.0);
    }
    conditioning
}

/// A first frame's latents from its `moments` (per pixel `LATENT_CHANNELS` means, then as many
/// log-variances) and `noise`: `mean + exp(½ · clamp(log-variance)) · ε`, then
/// `(z − LATENT_MEAN) / LATENT_STD`.
pub fn first_latents(moments: &[f32], noise: &[f32]) -> Vec<f32> {
    let mut latents = Vec::with_capacity(noise.len());
    let pixels = moments
        .as_chunks::<MOMENTS>()
        .0
        .iter()
        .zip(noise.as_chunks::<LATENT_CHANNELS>().0);
    for (moment, epsilon) in pixels {
        let (means, log_variances) = moment.split_at(LATENT_CHANNELS);
        for channel in 0..LATENT_CHANNELS {
            let log_variance =
                log_variances[channel].clamp(*LOG_VARIANCE.start(), *LOG_VARIANCE.end());
            let sample = means[channel] + (0.5 * log_variance).exp() * epsilon[channel];
            latents.push((sample - LATENT_MEAN[channel]) / LATENT_STD[channel]);
        }
    }
    latents
}

/// Each latent pixel's transformer input: its latents, then its conditioning.
pub fn transformer_input<'a>(
    latents: &'a [f32],
    conditioning: &'a [f32],
) -> impl Iterator<Item = f32> + 'a {
    let pixels = latents
        .as_chunks::<LATENT_CHANNELS>()
        .0
        .iter()
        .zip(conditioning.as_chunks::<CONDITION_CHANNELS>().0);
    pixels
        .flat_map(|(latent, condition)| latent.iter().chain(condition))
        .copied()
}

impl bundle::Engine for VideoEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            model: VideoModel::load(store)?,
            team: Team::global(),
            activations: Activations::Released,
            contexts: Contexts::new(),
            conditioned: [false; BRANCHES],
            image: None,
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
    /// Conditions `branch` on `text`: T5Gemma2's states, rows of `TEXT_HIDDEN`, at most
    /// `TEXT_TOKENS` of them.
    pub fn condition(&mut self, branch: Branch, text: &[f32]) -> Result<(), Error> {
        let rows = text.len() / TEXT_HIDDEN;
        if text.is_empty() || !text.len().is_multiple_of(TEXT_HIDDEN) || rows > TEXT_TOKENS {
            return Err(Error::Request(format!(
                "{} text values: 1 to {TEXT_TOKENS} rows of {TEXT_HIDDEN}",
                text.len()
            )));
        }
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            contexts,
            ..
        } = self;
        let buffers = activations.transformer(threads);
        team.run(|ctx| {
            model
                .transformer
                .embed_text(ctx, buffers, (text, branch), contexts);
        });
        contexts.set_text_rows(branch, rows);
        self.conditioned[branch as usize] = true;
        Ok(())
    }

    /// Conditions every branch on SigLIP's `states` of a first image (`IMAGE_TOKENS` rows of
    /// `IMAGE_HIDDEN`), or on no image.
    pub fn condition_image(&mut self, states: Option<&[f32]>) -> Result<(), Error> {
        if states.is_some_and(|states| states.len() != IMAGE_TOKENS * IMAGE_HIDDEN) {
            return Err(Error::Request(format!(
                "the image's states are {IMAGE_TOKENS} rows of {IMAGE_HIDDEN}"
            )));
        }
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            contexts,
            ..
        } = self;
        let buffers = activations.transformer(threads);
        team.run(|ctx| {
            model
                .transformer
                .embed_image(ctx, buffers, (states, contexts));
        });
        contexts.set_image_rows(states.map_or(0, |_| IMAGE_TOKENS));
        Ok(())
    }

    /// The autoencoder's moments of a first frame of `size` (`image::first_frame`'s): per
    /// latent pixel `LATENT_CHANNELS` means, then as many log-variances.
    pub fn encode_frame(&mut self, frame: &[f32], size: Size) -> Result<Vec<f32>, Error> {
        check_size(size.width, size.height)?;
        let area = size.width * size.height;
        if frame.len() != area * IMAGE_CHANNELS {
            return Err(Error::Request(format!(
                "{} frame values for {}×{} RGB pixels",
                frame.len(),
                size.width,
                size.height
            )));
        }
        let grid = Grid {
            images: 1,
            height: size.height,
            width: size.width,
        };
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            ..
        } = self;
        let vae = activations.encoder(threads);
        vae.frames.get_mut()[..frame.len()].copy_from_slice(frame);
        team.run(|ctx| model.encoder.encode(ctx, vae, grid));
        let pixels = area / (LATENT_FACTOR * LATENT_FACTOR);
        Ok(vae.latent.read()[..pixels * MOMENTS].to_vec())
    }

    /// The transformer's velocity for `input` (`transformer_input`'s, of `video`) at `timestep`
    /// under `branch`'s text.
    pub fn predict(
        &mut self,
        (input, video): (&[f32], Video),
        timestep: f32,
        branch: Branch,
    ) -> Result<Vec<f32>, Error> {
        check_values(input, video, INPUT_CHANNELS)?;
        self.check_conditioned(branch)?;
        let buffers = self.activations.transformer(self.team.threads());
        buffers.input.get_mut()[..input.len()].copy_from_slice(input);
        self.set_timestep(timestep);
        let mut velocity = vec![0.0; video.pixels() * LATENT_CHANNELS];
        self.velocity(video, branch, &mut velocity);
        Ok(velocity)
    }

    /// The noise levels and timesteps of a run of `steps` steps.
    pub fn schedule(&self, steps: usize) -> Schedule {
        Schedule::flow_sigmas(steps, SHIFT, TRAINING_TIMESTEPS)
    }

    /// The latents after `steps` denoising steps from `noise` (`video`) under `conditioning`,
    /// guided at `guidance` between the conditioned branches.
    pub fn sample(
        &mut self,
        (noise, conditioning): (&[f32], &[f32]),
        video: Video,
        (steps, guidance): (usize, f32),
    ) -> Result<Vec<f32>, Error> {
        check_values(noise, video, LATENT_CHANNELS)?;
        check_values(conditioning, video, CONDITION_CHANNELS)?;
        self.check_conditioned(Branch::Prompt)?;
        if guidance > 0.0 {
            self.check_conditioned(Branch::Negative)?;
        }
        let start = (noise.to_vec(), conditioning.to_vec());
        let mut job = self.begin(start, video, (steps, guidance));
        while job.step < steps {
            self.denoise(&mut job);
        }
        Ok(job.latents)
    }

    /// The video of `latents` (`video`): pixel-major RGB in `[−1, 1]`, frame after frame.
    pub fn decode(&mut self, (latents, video): (&[f32], Video)) -> Result<Vec<f32>, Error> {
        check_values(latents, video, LATENT_CHANNELS)?;
        let frame = frame_pixels(video) * IMAGE_CHANNELS;
        let mut frames = vec![0.0; video_frames(video) * frame];
        for strip in 0..strips(video) {
            self.decode_strip((latents, video), strip, |index, rows, values| {
                let row = values.len() / rows.len();
                let at = index * frame + rows.start * row;
                for (slot, value) in frames[at..at + values.len()].iter_mut().zip(values) {
                    *slot = value.clamp(-1.0, 1.0);
                }
            });
        }
        Ok(frames)
    }

    /// Keeps `image` for the next `start`, which conditions the video on it.
    pub fn first_image(&mut self, image: FirstImage) {
        self.image = Some(image);
    }

    /// Starts generating `request` from the states of its prompt and of its negative prompt (the
    /// latter read only when the guidance is above 0), and from the kept first image if any.
    pub fn start(
        &mut self,
        request: &Request,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<Plan, Error> {
        let image = self.image.take();
        let video = request.video()?;
        self.job = None;
        if let Some(image) = &image {
            check_image(image, request)?;
        }
        self.condition(Branch::Prompt, prompt)?;
        if request.guidance > 0.0 {
            self.condition(Branch::Negative, negative)?;
        }
        self.condition_image(image.as_ref().map(|image| image.states.as_slice()))?;
        let mut rng = Rng::new(request.seed);
        let mut latents = vec![0.0; video.pixels() * LATENT_CHANNELS];
        rng.fill_normal(&mut latents);
        let first = image
            .as_ref()
            .map(|image| self.first_frame_latents(image, video, &mut rng))
            .transpose()?;
        let conditions = conditioning(first.as_deref(), video);
        let steps = (request.steps, request.guidance);
        self.job = Some(self.begin((latents, conditions), video, steps));
        Ok(Plan {
            steps: request.steps,
            width: request.width,
            height: request.height,
            frames: request.frames,
            image: image.is_some(),
        })
    }

    /// Runs the next denoising step, or decodes the next latent frame once the steps are done.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self.job.take().ok_or_else(no_generation)?;
        let steps = job.schedule.steps();
        let strips = strips(job.video);
        if job.step < steps {
            self.denoise(&mut job);
        } else if job.decoded < strips {
            self.decode_next(&mut job);
        }
        let progress = Progress {
            step: job.step,
            steps,
            decoded: job.decoded,
            strips,
            done: job.decoded == strips,
        };
        self.job = Some(job);
        Ok(progress)
    }

    /// The first frame of the clean video the last step predicts, at full resolution: RGBA rows
    /// top to bottom. Between the first step and the decoding.
    pub fn preview(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.take().ok_or_else(no_generation)?;
        let first = (job.step > 0 && job.decoded == 0).then(|| {
            let video = Video {
                frames: 1,
                ..job.video
            };
            let latents = &job.denoised[..video.pixels() * LATENT_CHANNELS];
            let mut rgba = vec![0; frame_pixels(video) * RGBA];
            self.decode_rgba((latents, video), 0..strips(video), &mut rgba);
            rgba
        });
        self.job = Some(job);
        first.ok_or_else(|| {
            Error::Request("a preview exists between the first step and decoding".into())
        })
    }

    /// The RGBA bytes of every frame, one after another, once the last strip is decoded; empty
    /// before, and after the first call that returns them.
    pub fn take_frames(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.as_mut().ok_or_else(no_generation)?;
        if job.decoded < strips(job.video) {
            return Ok(Vec::new());
        }
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

    /// The latents of `image`'s frame (a video of `video`'s size), drawn from the encoder's
    /// posterior with `rng`.
    fn first_frame_latents(
        &mut self,
        image: &FirstImage,
        video: Video,
        rng: &mut Rng,
    ) -> Result<Vec<f32>, Error> {
        let size = Size {
            width: video.width * LATENT_FACTOR,
            height: video.height * LATENT_FACTOR,
        };
        let moments = self.encode_frame(&image.frame, size)?;
        let mut noise = vec![0.0; video.height * video.width * LATENT_CHANNELS];
        rng.fill_normal(&mut noise);
        Ok(first_latents(&moments, &noise))
    }

    /// A generation from `latents` and their `conditioning` (`video`), `steps` steps and
    /// `guidance` (unguided at 0).
    fn begin(
        &self,
        (latents, conditioning): (Vec<f32>, Vec<f32>),
        video: Video,
        (steps, guidance): (usize, f32),
    ) -> Job {
        let len = latents.len();
        let settings = Settings {
            scale: guidance,
            ..GUIDANCE
        };
        Job {
            video,
            schedule: self.schedule(steps),
            solver: Solver::default(),
            guidance: (guidance > 0.0).then(|| Guidance::new(settings)),
            latents,
            conditioning,
            conditional: vec![0.0; len],
            unconditional: vec![0.0; len],
            guided: vec![0.0; len],
            denoised: vec![0.0; len],
            step: 0,
            decoded: 0,
            frames: Vec::new(),
        }
    }

    /// The transformer's modulations at `timestep`, which every branch of the step shares.
    fn set_timestep(&mut self, timestep: f32) {
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            ..
        } = self;
        let buffers = activations.transformer(threads);
        team.run(|ctx| model.transformer.time(ctx, buffers, timestep));
    }

    /// The velocity of the transformer's input (`video`) under `branch`'s context at the
    /// timestep set, into `out`.
    fn velocity(&mut self, video: Video, branch: Branch, out: &mut [f32]) {
        let patching = Patching {
            video,
            patch: PATCH,
        };
        let positions: Vec<Position> = (0..patching.tokens())
            .map(|token| patching.position(token))
            .collect();
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            contexts,
            ..
        } = self;
        let buffers = activations.transformer(threads);
        let call = Call {
            contexts,
            branch,
            patching,
            positions: &positions,
        };
        team.run(|ctx| model.transformer.forward(ctx, buffers, call));
        out.copy_from_slice(&buffers.velocity.read()[..out.len()]);
    }

    /// One step of `job`: the prompt's velocity, and with guidance the negative prompt's, guided
    /// adaptively, then the solver.
    fn denoise(&mut self, job: &mut Job) {
        let input = transformer_input(&job.latents, &job.conditioning);
        let buffers = self.activations.transformer(self.team.threads());
        for (slot, value) in buffers.input.get_mut().iter_mut().zip(input) {
            *slot = value;
        }
        self.set_timestep(job.schedule.timestep(job.step));
        self.velocity(job.video, Branch::Prompt, &mut job.conditional);
        match &mut job.guidance {
            Some(guidance) => {
                self.velocity(job.video, Branch::Negative, &mut job.unconditional);
                let group = job.video.height * job.video.width * LATENT_CHANNELS;
                let velocities = (&job.conditional[..], &job.unconditional[..]);
                guidance.guide(velocities, group, &mut job.guided);
            }
            None => job.guided.copy_from_slice(&job.conditional),
        }
        let at = (&job.schedule, job.step);
        job.solver
            .step(at, &mut job.latents, &job.guided, &mut job.denoised);
        job.step += 1;
    }

    /// Strip `job.decoded` of `job.latents` through the decoder, into `job.frames`.
    fn decode_next(&mut self, job: &mut Job) {
        if job.decoded == 0 {
            job.frames = vec![0; video_frames(job.video) * frame_pixels(job.video) * RGBA];
        }
        let strip = job.decoded..job.decoded + 1;
        self.decode_rgba((&job.latents, job.video), strip, &mut job.frames);
        job.decoded += 1;
    }

    /// Strips `strips` of the video of `latents` (`video`) through the decoder, into `rgba`: the
    /// video's frames one after another, RGBA rows top to bottom.
    fn decode_rgba(
        &mut self,
        (latents, video): (&[f32], Video),
        strips: Range<usize>,
        rgba: &mut [u8],
    ) {
        let frame = frame_pixels(video) * RGBA;
        for strip in strips {
            self.decode_strip((latents, video), strip, |index, rows, values| {
                let bytes = rgba_from_signed(values);
                let row = bytes.len() / rows.len();
                let at = index * frame + rows.start * row;
                rgba[at..at + bytes.len()].copy_from_slice(&bytes);
            });
        }
    }

    /// Strip `strip` of every latent frame of `latents` (`video`) through the decoder, after the
    /// steps every strip reads when it is the first: `place(frame, rows, values)` for each video
    /// frame's output rows `rows`, their pixel-major RGB, unclamped.
    fn decode_strip(
        &mut self,
        (latents, video): (&[f32], Video),
        strip: usize,
        mut place: impl FnMut(usize, Range<usize>, &[f32]),
    ) {
        let grid = Grid {
            images: 1,
            height: video.height,
            width: video.width,
        };
        let threads = self.team.threads();
        let Self {
            model,
            team,
            activations,
            ..
        } = self;
        let vae = activations.decoder(threads);
        if strip == 0 {
            let values = grid.area() * LATENT_CHANNELS;
            for (index, frame) in latents.chunks_exact(values).enumerate() {
                vae.latent.get_mut()[..values].copy_from_slice(frame);
                team.run(|ctx| model.decoder.decode_latents(ctx, vae, (grid, index)));
            }
        }
        let rows = strip * DECODER.strip..((strip + 1) * DECODER.strip).min(video.height);
        let output = rows.start * LATENT_FACTOR..rows.end * LATENT_FACTOR;
        let band = output.len() * video.width * LATENT_FACTOR * IMAGE_CHANNELS;
        let mut first = 0;
        for index in 0..video.frames {
            team.run(|ctx| {
                model
                    .decoder
                    .decode_strip(ctx, vae, (grid, index), rows.clone())
            });
            let frames = VAE_SHAPE.frames(index);
            let decoded = vae.frames.read()[..frames * band].chunks_exact(band);
            for (frame, values) in (first..first + frames).zip(decoded) {
                place(frame, output.clone(), values);
            }
            first += frames;
        }
    }
}

/// Strips of `video`'s latent rows the decoder decodes it in.
fn strips(video: Video) -> usize {
    video.height.div_ceil(DECODER.strip)
}

/// Frames of a video of `video`'s latents.
fn video_frames(video: Video) -> usize {
    (0..video.frames).map(|index| VAE_SHAPE.frames(index)).sum()
}

/// Pixels of each of `video`'s frames.
fn frame_pixels(video: Video) -> usize {
    video.height * video.width * LATENT_FACTOR * LATENT_FACTOR
}

fn no_generation() -> Error {
    Error::Request("no generation in progress".into())
}

/// The first image's frame is a video of `request`'s size, and the video has a latent frame
/// beyond the first to generate.
fn check_image(image: &FirstImage, request: &Request) -> Result<(), Error> {
    let values = request.width * request.height * IMAGE_CHANNELS;
    if request.frames >= MIN_IMAGE_FRAMES && image.frame.len() == values {
        return Ok(());
    }
    Err(Error::Request(format!(
        "a first image needs a frame of {}×{} RGB pixels (it has {} values) and at least \
         {MIN_IMAGE_FRAMES} frames (the request has {})",
        request.width,
        request.height,
        image.frame.len(),
        request.frames
    )))
}
