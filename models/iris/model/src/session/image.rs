//! The image engine: `start_generation`, `start_depth` and `start_upscale` begin a job on the
//! transformer the bundle holds, and each `step` runs one denoising step, the depth pass or one
//! tile of a restoration. The stages also run one at a time, on given inputs (`condition`,
//! `predict`, `sample`, `restore_tiles`), to check and time them.

use blocks::dpm::Schedule;
use bundle::Store;
use kernels::Vector;
use kernels::rng::Rng;
use media::image::{Size, rgba_from_signed};
use team::{SharedBuf, Team};

use super::jobs::Job;
use super::request::{Generation, Plan, Progress, check_image, check_size, check_tiled};
use crate::Error;
use crate::adapter::{Adapter, AdapterBuffers};
use crate::dit::{self, Dit};
use crate::model::{
    EMPTY_PROMPT, HIDDEN, IMAGE_CHANNELS, LARGEST_LEVEL, PATCH, TEXT_LAYERS, TEXT_ROWS,
    TRAINING_TIMESTEPS, VARIANT_PART, Variant,
};
use crate::restore::{Fusion, tiles};

/// A branch of guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Prompt,
    Negative,
}

pub struct ImageEngine {
    pub(super) model: Dit,
    pub(super) variant: Variant,
    pub(super) team: Team,
    pub(super) buffers: dit::Buffers,
    adapter: AdapterBuffers,
    contexts: [SharedBuf; 2],
    conditioned: [bool; 2],
    pub(super) job: Option<Job>,
}

impl bundle::Engine for ImageEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let variant = Variant::read(&store.take_blob(VARIANT_PART)?)?;
        let model = Dit::load(store, variant)?;
        let team = Team::global();
        let threads = team.threads();
        let mut engine = Self {
            model,
            variant,
            buffers: dit::Buffers::new(threads),
            adapter: Adapter::buffers(threads),
            contexts: std::array::from_fn(|_| SharedBuf::new(TEXT_ROWS * HIDDEN)),
            conditioned: [false; 2],
            job: None,
            team,
        };
        if variant.has_empty_prompt() {
            let empty = Vector::new(store.take(EMPTY_PROMPT)?)?;
            engine.condition_on(Branch::Prompt, empty.values())?;
        }
        Ok(engine)
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {
        self.job = None;
    }
}

impl ImageEngine {
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Conditions `branch` on `states`: the text engine's (`rows × 12 × HIDDEN`, at most
    /// `TEXT_ROWS` rows). Depth and the restorer read the bundle's empty prompt instead.
    pub fn condition(&mut self, branch: Branch, states: &[f32]) -> Result<(), Error> {
        self.require(&[Variant::Base, Variant::Turbo])?;
        self.condition_on(branch, states)
    }

    fn condition_on(&mut self, branch: Branch, states: &[f32]) -> Result<(), Error> {
        let row_values = TEXT_LAYERS.len() * HIDDEN;
        let rows = states.len() / row_values;
        if states.is_empty() || !states.len().is_multiple_of(row_values) || rows > TEXT_ROWS {
            return Err(Error::Request(format!(
                "{} text values: 1 to {TEXT_ROWS} rows of {} layers of {HIDDEN}",
                states.len(),
                TEXT_LAYERS.len()
            )));
        }
        let Self {
            model,
            team,
            adapter,
            contexts,
            ..
        } = self;
        let context = &contexts[branch as usize];
        team.run(|ctx| {
            model
                .adapter
                .condition(ctx, adapter, (states, rows), context)
        });
        self.conditioned[branch as usize] = true;
        Ok(())
    }

    /// The adapted text of `branch` (`TEXT_ROWS × HIDDEN`).
    pub fn adapted(&self, branch: Branch) -> Result<Vec<f32>, Error> {
        self.check_conditioned(branch)?;
        Ok(self.contexts[branch as usize].read().to_vec())
    }

    /// The transformer's prediction (a velocity, or depth's relative log depth) for `image`
    /// (pixel-major RGB of `size`) at model time `timestep` under `branch`'s text.
    pub fn predict(
        &mut self,
        (image, size): (&[f32], Size),
        timestep: f32,
        branch: Branch,
    ) -> Result<Vec<f32>, Error> {
        check_size(size, PATCH)?;
        check_image(image, size)?;
        self.check_conditioned(branch)?;
        self.buffers.image.get_mut()[..image.len()].copy_from_slice(image);
        self.set_timestep(timestep);
        let mut prediction = vec![0.0; size.width * size.height * self.variant.outputs()];
        self.velocity(size, branch, &mut prediction);
        Ok(prediction)
    }

    /// The levels and timesteps of a base run of `steps` steps at `shift`.
    pub fn schedule(&self, steps: usize, shift: f32) -> Schedule {
        let shift = f64::from(shift);
        Schedule::flow_continuous(steps, shift, LARGEST_LEVEL, TRAINING_TIMESTEPS)
    }

    /// The clean image, unclamped, after `request`'s steps from `image` (pixel-major RGB of the
    /// request's size) under the conditioned branches.
    pub fn sample(&mut self, image: &[f32], request: &Generation) -> Result<Vec<f32>, Error> {
        self.check_request(request)?;
        check_image(image, request.size())?;
        self.check_conditioned(Branch::Prompt)?;
        let mut job = self.begin(image.to_vec(), request);
        if job.guided() {
            self.check_conditioned(Branch::Negative)?;
        }
        while !job.done() {
            self.denoise(&mut job);
        }
        Ok(job.into_pixels())
    }

    /// Starts generating `request`, at one of `SIZES`, from the text engine's states of its
    /// prompt and (base, when guided) of its negative prompt.
    pub fn start_generation(
        &mut self,
        request: &Generation,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<Plan, Error> {
        self.job = None;
        self.check_request(request)?;
        request.check_trained_size()?;
        self.condition_on(Branch::Prompt, prompt)?;
        let size = request.size();
        let mut image = vec![0.0; size.width * size.height * IMAGE_CHANNELS];
        Rng::new(request.seed).fill_normal(&mut image);
        let job = self.begin(image, request);
        if job.guided() {
            self.condition_on(Branch::Negative, negative)?;
        }
        let steps = job.steps();
        self.job = Some(Job::Generation(job));
        Ok(Plan {
            steps,
            width: size.width,
            height: size.height,
        })
    }

    /// Runs the next step of the job started: a denoising step, the depth pass, or a tile.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self.job.take().ok_or_else(no_job)?;
        if !job.done() {
            match &mut job {
                Job::Generation(generation) => self.denoise(generation),
                Job::Depth(depth) => self.depth_pass(depth),
                Job::Upscale(upscale) => self.restore_next(upscale),
            }
        }
        let progress = job.progress();
        self.job = Some(job);
        Ok(progress)
    }

    /// The clean image the last denoising step predicts, at full size: RGBA rows top to bottom.
    /// Between the first step and the image.
    pub fn preview(&self) -> Result<Vec<u8>, Error> {
        match &self.job {
            Some(Job::Generation(job)) if job.step() > 0 => Ok(rgba_from_signed(job.denoised())),
            _ => Err(Error::Request(
                "a preview exists after the first denoising step".into(),
            )),
        }
    }

    /// The finished job's image: RGBA rows top to bottom. Ends the job.
    pub fn take_image(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.take().ok_or_else(no_job)?;
        if !job.done() {
            self.job = Some(job);
            return Err(Error::Request("the job has steps left".into()));
        }
        Ok(job.into_rgba())
    }

    /// Every tile of the restorer's `work` image (`size`, padded; `restore::work`'s) restored and
    /// fused, as `[−1, 1]` pixel-major RGB.
    pub fn restore_tiles(&mut self, (work, size): (&[f32], Size)) -> Result<Vec<f32>, Error> {
        self.require(&[Variant::Upscaler])?;
        check_tiled(work.len(), size)?;
        let mut fusion = Fusion::new(size);
        for corner in tiles(size) {
            let restored = self.restore_tile((work, size), corner);
            fusion.add(corner, &restored);
        }
        Ok(fusion.finish())
    }

    /// An error unless this bundle holds one of `variants`.
    pub(super) fn require(&self, variants: &[Variant]) -> Result<(), Error> {
        if variants.contains(&self.variant) {
            return Ok(());
        }
        Err(Error::Request(format!(
            "this bundle holds the {} model, not {}",
            self.variant.name(),
            variants
                .iter()
                .map(|variant| variant.name())
                .collect::<Vec<_>>()
                .join(" or ")
        )))
    }

    /// An error unless `request` is one this bundle's prompt-reading model serves.
    fn check_request(&self, request: &Generation) -> Result<(), Error> {
        self.require(&[Variant::Base, Variant::Turbo])?;
        match self.variant {
            Variant::Base => request.check_base(),
            _ => request.check_size(),
        }
    }

    fn check_conditioned(&self, branch: Branch) -> Result<(), Error> {
        if self.conditioned[branch as usize] {
            return Ok(());
        }
        Err(Error::Request(format!(
            "the {branch:?} branch has no text yet"
        )))
    }

    /// The transformer's modulations at `timestep`, which every branch of the step shares.
    pub(super) fn set_timestep(&self, timestep: f32) {
        let Self {
            model,
            team,
            buffers,
            ..
        } = self;
        team.run(|ctx| model.time(ctx, buffers, timestep));
    }

    /// The prediction of `self.buffers.image` (`size`) under `branch`'s text at the timestep
    /// set, into `out`.
    pub(super) fn velocity(&self, size: Size, branch: Branch, out: &mut [f32]) {
        let Self {
            model,
            team,
            buffers,
            contexts,
            ..
        } = self;
        let context = &contexts[branch as usize];
        team.run(|ctx| model.forward(ctx, buffers, context, size));
        out.copy_from_slice(&buffers.velocity.read()[..out.len()]);
    }
}

fn no_job() -> Error {
    Error::Request("no job in progress".into())
}
