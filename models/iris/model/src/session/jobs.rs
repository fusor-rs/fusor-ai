//! The jobs of the image engine and their steps: a generation's denoising steps (DPM-Solver++(2M)
//! for the base model, Euler steps over Turbo's levels), the one pass of depth, and a
//! restoration's tiles.

use blocks::dpm::{Schedule, Solver};
use blocks::flow::{self, euler_step};
use media::image::{Pixels, Size, rgba_from_signed};

use super::image::{Branch, ImageEngine};
use super::request::{
    DepthRequest, Generation, Plan, Progress, Upscale, check_photo, check_restored, check_size,
};
use crate::Error;
use crate::depth;
use crate::model::{
    DEPTH_TIME, IMAGE_CHANNELS, PATCH, RESTORE_LEVEL, TILE, TRAINING_TIMESTEPS, TURBO_LEVELS,
    Variant,
};
use crate::picture::{colorize, opaque};
use crate::restore::{self, Fusion, Work};

pub(super) enum Job {
    Generation(GenerationJob),
    Depth(DepthJob),
    Upscale(UpscaleJob),
}

impl Job {
    pub(super) fn done(&self) -> bool {
        let progress = self.progress();
        progress.step == progress.steps
    }

    pub(super) fn progress(&self) -> Progress {
        let (step, steps) = match self {
            Job::Generation(job) => (job.step, job.sampler.steps()),
            Job::Depth(job) => (usize::from(job.result.is_some()), 1),
            Job::Upscale(job) => (job.next, job.corners.len()),
        };
        Progress {
            step,
            steps,
            done: step == steps,
        }
    }

    /// The finished job's RGBA image.
    pub(super) fn into_rgba(self) -> Vec<u8> {
        match self {
            Job::Generation(job) => rgba_from_signed(&job.image),
            Job::Depth(DepthJob { result, .. }) | Job::Upscale(UpscaleJob { result, .. }) => {
                result.expect("a finished job has its image")
            }
        }
    }
}

enum Sampler {
    Base { schedule: Schedule, solver: Solver },
    Turbo { schedule: flow::Schedule },
}

impl Sampler {
    fn steps(&self) -> usize {
        match self {
            Sampler::Base { schedule, .. } => schedule.steps(),
            Sampler::Turbo { schedule } => schedule.steps(),
        }
    }

    /// The model time of step `step`: the level asked at, over 1000.
    fn timestep(&self, step: usize) -> f32 {
        match self {
            Sampler::Base { schedule, .. } => schedule.timestep(step),
            Sampler::Turbo { schedule } => schedule.times()[step] * TRAINING_TIMESTEPS as f32,
        }
    }

    /// `image` moved to the next level by `velocity`; `denoised` gets the clean image the step
    /// predicts.
    fn advance(
        &mut self,
        step: usize,
        image: &mut [f32],
        (velocity, denoised): (&[f32], &mut [f32]),
    ) {
        match self {
            Sampler::Base { schedule, solver } => {
                solver.step((schedule, step), image, velocity, denoised);
            }
            Sampler::Turbo { schedule } => {
                let level = schedule.times()[step];
                for ((clean, value), change) in denoised.iter_mut().zip(image.iter()).zip(velocity)
                {
                    *clean = value - level * change;
                }
                euler_step(image, velocity, None, schedule.dt(step));
            }
        }
    }
}

pub(super) struct GenerationJob {
    size: Size,
    sampler: Sampler,
    /// The guidance scale, when the prompt's branch is guided by the negative prompt's.
    guidance: Option<f32>,
    image: Vec<f32>,
    conditional: Vec<f32>,
    velocity: Vec<f32>,
    denoised: Vec<f32>,
    step: usize,
}

impl GenerationJob {
    pub(super) fn guided(&self) -> bool {
        self.guidance.is_some()
    }

    pub(super) fn steps(&self) -> usize {
        self.sampler.steps()
    }

    pub(super) fn done(&self) -> bool {
        self.step == self.steps()
    }

    pub(super) fn step(&self) -> usize {
        self.step
    }

    /// The clean image the last step predicts.
    pub(super) fn denoised(&self) -> &[f32] {
        &self.denoised
    }

    pub(super) fn into_pixels(self) -> Vec<f32> {
        self.image
    }
}

pub(super) struct DepthJob {
    photo: Size,
    model: Size,
    input: Vec<f32>,
    result: Option<Vec<u8>>,
}

pub(super) struct UpscaleJob {
    reference: Vec<f32>,
    output: Size,
    work: Vec<f32>,
    layout: Work,
    corners: Vec<(usize, usize)>,
    fusion: Fusion,
    next: usize,
    result: Option<Vec<u8>>,
}

impl ImageEngine {
    /// A generation from `image` (pixel-major RGB of the request's size).
    pub(super) fn begin(&self, image: Vec<f32>, request: &Generation) -> GenerationJob {
        let sampler = match self.variant {
            Variant::Base => Sampler::Base {
                schedule: self.schedule(request.steps, request.shift),
                solver: Solver::default(),
            },
            _ => Sampler::Turbo {
                schedule: flow::Schedule::new(TURBO_LEVELS.to_vec(), |_, _| false),
            },
        };
        let len = image.len();
        GenerationJob {
            size: request.size(),
            sampler,
            guidance: request.guided(self.variant).then_some(request.guidance),
            image,
            conditional: vec![0.0; len],
            velocity: vec![0.0; len],
            denoised: vec![0.0; len],
            step: 0,
        }
    }

    /// One step of `job`: the prompt's velocity, and with guidance the negative prompt's, then
    /// the sampler.
    pub(super) fn denoise(&mut self, job: &mut GenerationJob) {
        self.buffers.image.get_mut()[..job.image.len()].copy_from_slice(&job.image);
        self.set_timestep(job.sampler.timestep(job.step));
        self.velocity(job.size, Branch::Prompt, &mut job.conditional);
        match job.guidance {
            Some(scale) => {
                self.velocity(job.size, Branch::Negative, &mut job.velocity);
                for (guided, conditional) in job.velocity.iter_mut().zip(&job.conditional) {
                    *guided += scale * (conditional - *guided);
                }
            }
            None => job.velocity.copy_from_slice(&job.conditional),
        }
        job.sampler.advance(
            job.step,
            &mut job.image,
            (&job.velocity[..], &mut job.denoised[..]),
        );
        job.step += 1;
    }

    /// Starts the depth map of `photo`, its prediction at `request.max_side`.
    pub fn start_depth(&mut self, request: &DepthRequest, photo: &Pixels) -> Result<Plan, Error> {
        self.require(&[Variant::Depth])?;
        self.job = None;
        check_photo(photo)?;
        let model = depth::model_size(photo.size, request.max_side);
        check_size(model, PATCH)?;
        let job = DepthJob {
            photo: photo.size,
            model,
            input: depth::input(photo, model),
            result: None,
        };
        self.job = Some(Job::Depth(job));
        Ok(Plan {
            steps: 1,
            width: photo.size.width,
            height: photo.size.height,
        })
    }

    /// The depth pass: the prediction, back to the photo's size, coloured.
    pub(super) fn depth_pass(&mut self, job: &mut DepthJob) {
        self.buffers.image.get_mut()[..job.input.len()].copy_from_slice(&job.input);
        self.set_timestep(DEPTH_TIME);
        let mut prediction = vec![0.0; job.model.width * job.model.height];
        self.velocity(job.model, Branch::Prompt, &mut prediction);
        let depth = depth::upsample(&prediction, job.model, job.photo);
        job.result = Some(opaque(&colorize(&depth)));
    }

    /// Starts restoring `photo` enlarged four times, shrunk to `request`'s budget first.
    pub fn start_upscale(&mut self, request: &Upscale, photo: &Pixels) -> Result<Plan, Error> {
        self.require(&[Variant::Upscaler])?;
        self.job = None;
        check_photo(photo)?;
        let budgeted = restore::budget(photo, (request.short_side, request.long_side));
        let output = restore::enlarged(budgeted.size);
        let layout = check_restored(output)?;
        let reference = restore::upsample(&budgeted, output);
        let work = restore::work(&reference, output, &layout);
        let corners = restore::tiles(layout.padded);
        let steps = corners.len();
        self.job = Some(Job::Upscale(UpscaleJob {
            reference,
            output,
            work,
            layout,
            corners,
            fusion: Fusion::new(layout.padded),
            next: 0,
            result: None,
        }));
        Ok(Plan {
            steps,
            width: output.width,
            height: output.height,
        })
    }

    /// The next tile of `job`; after the last, the restored image.
    pub(super) fn restore_next(&mut self, job: &mut UpscaleJob) {
        let corner = job.corners[job.next];
        let restored = self.restore_tile((&job.work, job.layout.padded), corner);
        job.fusion.add(corner, &restored);
        job.next += 1;
        if job.next < job.corners.len() {
            return;
        }
        let fused = job.fusion.finish();
        let image = restore::finish(&fused, &job.layout, job.output);
        let fixed = restore::color_fix(&image, &job.reference, job.output);
        job.result = Some(opaque(&restore::bytes(&fixed)));
    }

    /// The tile of `work` (`padded`) whose top left corner is `corner`, restored in one step:
    /// `tile − RESTORE_LEVEL · velocity`.
    pub(super) fn restore_tile(
        &mut self,
        (work, padded): (&[f32], Size),
        (top, left): (usize, usize),
    ) -> Vec<f32> {
        let row_values = TILE * IMAGE_CHANNELS;
        for row in 0..TILE {
            let source = ((top + row) * padded.width + left) * IMAGE_CHANNELS;
            let target = row * row_values..(row + 1) * row_values;
            self.buffers.image.get_mut()[target]
                .copy_from_slice(&work[source..source + row_values]);
        }
        self.set_timestep(RESTORE_LEVEL * TRAINING_TIMESTEPS as f32);
        let tile = Size {
            width: TILE,
            height: TILE,
        };
        let mut velocity = vec![0.0; TILE * row_values];
        self.velocity(tile, Branch::Prompt, &mut velocity);
        let input = self.buffers.image.read();
        input
            .iter()
            .zip(&velocity)
            .map(|(value, change)| value - RESTORE_LEVEL * change)
            .collect()
    }
}
