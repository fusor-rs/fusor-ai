//! The image engine: `start` conditions the transformer on the states of a request's caption and
//! negative prompt and draws its starting noise; each `step` then runs one denoising step (the
//! two branches of guidance as two transformer calls). The stages also run one at a time, on given
//! inputs (`condition`, `predict`, `sample`), to check and time them.

use blocks::apg::{Guidance, Settings};
use blocks::flow::Schedule;
use blocks::pyramid_jit::{Buffers, Dit, Text};
use bundle::{Load, Store};
use kernels::ops::round_bf16;
use kernels::rng::Rng;
use media::image::rgba_from_signed;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::model::{
    CHANNELS, DIT, GUIDANCE, IMAGE, NOISE_SCALE, SIZES, TEXT_TOKENS, TEXT_WIDTH, TRAINING_TIMESTEPS,
};
use crate::{Error, Size};

/// A generation is one of `SIZES`. The stages alone (`predict`, `sample`) take any image whose
/// sides are `MIN_SIDE` to `MAX_SIDE` pixels, any length (the DiT pads them to whole patches and
/// crops back), its patches at most `MAX_PATCHES`: the largest grid of `SIZES` (864 × 1216, 27 × 38
/// patches). The DiT's buffers are sized for it.
const MIN_SIDE: usize = 64;
const MAX_SIDE: usize = 1280;
const MAX_PATCHES: usize = 27 * 38;
const MAX_STEPS: usize = 100;
const MAX_GUIDANCE: f32 = 30.0;
/// The sampler's start, noise.
const START: f32 = 1.0;
/// The sampler's end, the clean image.
const END: f32 = 0.0;
const BRANCHES: usize = 2;

/// What to generate.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Adaptive projected guidance's scale.
    pub guidance: f32,
    pub seed: u64,
}

impl Request {
    /// The requested image's size, if it is one P-JiT 21K was trained at.
    fn size(&self) -> Result<Size, Error> {
        if SIZES.contains(&(self.width, self.height)) {
            return Ok(Size {
                width: self.width,
                height: self.height,
            });
        }
        let trained: Vec<String> = SIZES
            .iter()
            .map(|(width, height)| format!("{width}×{height}"))
            .collect();
        Err(Error::Request(format!(
            "{}×{}: P-JiT 21K generates only the sizes it was trained at, {}",
            self.width,
            self.height,
            trained.join(", ")
        )))
    }
}

/// A started generation.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub steps: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub step: usize,
    pub steps: usize,
    /// The last step ran: the image is ready (`take_image`).
    pub done: bool,
}

/// A branch of guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Prompt,
    Negative,
}

pub struct ImageEngine {
    dit: Dit,
    team: Team,
    buffers: Buffers,
    texts: [Text; BRANCHES],
    conditioned: [bool; BRANCHES],
    job: Option<Job>,
}

struct Job {
    image: Size,
    schedule: Schedule,
    /// Adaptive projected guidance, with the running sum of the differences.
    guidance: Guidance,
    /// The image being denoised, and the DiT's input of the last step (it rounded to bf16).
    state: Vec<f32>,
    seen: Vec<f32>,
    /// The last step's velocities of both branches, and the guided velocity.
    conditional: Vec<f32>,
    other: Vec<f32>,
    guided: Vec<f32>,
    /// The clean image the last step predicts.
    clean: Vec<f32>,
    step: usize,
}

/// The `steps + 1` knots from `START` to `END` as `torch.linspace(1, 0, steps + 1)` computes them
/// in f32: from the start for the first half, from the end for the rest, each one fused
/// multiply-add (one rounding, as torch's CPU and CUDA kernels compile it).
pub fn knots(steps: usize) -> Vec<f32> {
    let points = steps + 1;
    let step = (END - START) / steps as f32;
    let halfway = points / 2;
    (0..points)
        .map(|index| {
            if index < halfway {
                step.mul_add(index as f32, START)
            } else {
                (-step).mul_add((points - index - 1) as f32, END)
            }
        })
        .collect()
}

/// The integer timestep the DiT is asked at, from a knot: `trunc(t · 1000)` in f32.
pub fn timestep(time: f32) -> f32 {
    (time * TRAINING_TIMESTEPS as f32).trunc()
}

impl bundle::Engine for ImageEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let dit = Dit::load(store, IMAGE, DIT)?;
        let team = Team::global();
        Ok(Self {
            dit,
            buffers: DIT.buffers(MAX_PATCHES, team.threads()),
            texts: std::array::from_fn(|_| DIT.text()),
            conditioned: [false; BRANCHES],
            job: None,
            team,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {
        self.job = None;
    }
}

impl ImageEngine {
    /// Conditions `branch` on `states`: the text engine's, rows of `TEXT_WIDTH`, at most
    /// `TEXT_TOKENS` of them.
    pub fn condition(&mut self, branch: Branch, states: &[f32]) -> Result<(), Error> {
        if states.is_empty()
            || !states.len().is_multiple_of(TEXT_WIDTH)
            || states.len() > TEXT_TOKENS * TEXT_WIDTH
        {
            return Err(Error::Request(format!(
                "{} text values: 1 to {TEXT_TOKENS} rows of {TEXT_WIDTH}",
                states.len()
            )));
        }
        let Self {
            dit,
            team,
            buffers,
            texts,
            ..
        } = self;
        dit.condition(team, buffers, states, &mut texts[branch as usize]);
        self.conditioned[branch as usize] = true;
        Ok(())
    }

    /// The DiT's clean image for `seen` (`image`, pixel-major RGB) at `timestep` under `branch`'s
    /// text.
    pub fn predict(
        &mut self,
        (seen, image): (&[f32], Size),
        timestep: f32,
        branch: Branch,
    ) -> Result<Vec<f32>, Error> {
        check_image(image, seen.len())?;
        self.check_conditioned(branch)?;
        self.buffers.image.get_mut()[..seen.len()].copy_from_slice(seen);
        self.run_step(image, timestep, branch);
        Ok(self.buffers.prediction.read()[..seen.len()].to_vec())
    }

    /// The image after `steps` steps from `noise` (`image`, standard normals, pixel-major),
    /// guided at `guidance`, unclamped.
    pub fn sample(
        &mut self,
        (noise, image): (&[f32], Size),
        (steps, guidance): (usize, f32),
    ) -> Result<Vec<f32>, Error> {
        check_image(image, noise.len())?;
        check_run(steps, guidance)?;
        self.check_conditioned(Branch::Prompt)?;
        self.check_conditioned(Branch::Negative)?;
        let mut job = self.begin(noise, image, (steps, guidance));
        while job.step < steps {
            self.denoise(&mut job);
        }
        Ok(job.state)
    }

    /// Starts generating `request` from the states of its caption and of its negative prompt.
    pub fn start(
        &mut self,
        request: &Request,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<Plan, Error> {
        let image = request.size()?;
        let values = image.pixels() * CHANNELS;
        check_image(image, values)?;
        check_run(request.steps, request.guidance)?;
        self.job = None;
        self.condition(Branch::Prompt, prompt)?;
        self.condition(Branch::Negative, negative)?;
        let mut noise = vec![0.0; values];
        Rng::new(request.seed).fill_normal(&mut noise);
        let run = (request.steps, request.guidance);
        self.job = Some(self.begin(&noise, image, run));
        Ok(Plan {
            steps: request.steps,
            width: request.width,
            height: request.height,
        })
    }

    /// Runs the next denoising step; the last one leaves the image ready.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self.job.take().ok_or_else(no_generation)?;
        let steps = job.schedule.steps();
        if job.step < steps {
            self.denoise(&mut job);
        }
        let progress = Progress {
            step: job.step,
            steps,
            done: job.step == steps,
        };
        self.job = Some(job);
        Ok(progress)
    }

    /// The clean image the last step predicts: RGBA rows top to bottom.
    pub fn preview(&self) -> Result<Vec<u8>, Error> {
        let job = self.job.as_ref().ok_or_else(no_generation)?;
        if job.step == 0 {
            return Err(Error::Request(
                "a preview exists after the first step".into(),
            ));
        }
        Ok(rgba_from_signed(&job.clean))
    }

    /// The finished image's RGBA bytes, rows top to bottom; ends the generation.
    pub fn take_image(&mut self) -> Result<Vec<u8>, Error> {
        let job = self.job.as_ref().ok_or_else(no_generation)?;
        if job.step < job.schedule.steps() {
            return Err(Error::Request("the image is not finished".into()));
        }
        let image = rgba_from_signed(&job.state);
        self.job = None;
        Ok(image)
    }

    fn check_conditioned(&self, branch: Branch) -> Result<(), Error> {
        if self.conditioned[branch as usize] {
            return Ok(());
        }
        Err(Error::Request(format!(
            "the {branch:?} branch has no text yet"
        )))
    }

    /// A generation from `noise` (`image`), `steps` steps at `guidance`.
    fn begin(&self, noise: &[f32], image: Size, (steps, guidance): (usize, f32)) -> Job {
        let zeros = || vec![0.0; noise.len()];
        let settings = Settings {
            scale: guidance,
            ..GUIDANCE
        };
        Job {
            image,
            schedule: Schedule::new(knots(steps), |_, _| true),
            guidance: Guidance::new(settings),
            state: noise.iter().map(|value| value * NOISE_SCALE).collect(),
            seen: zeros(),
            conditional: zeros(),
            other: zeros(),
            guided: zeros(),
            clean: zeros(),
            step: 0,
        }
    }

    /// The modulations at `timestep`, the image's refined tokens and the DiT under `branch`'s
    /// text, for `self.buffers.image`, into `self.buffers.prediction`.
    fn run_step(&self, image: Size, timestep: f32, branch: Branch) {
        let Self {
            dit,
            team,
            buffers,
            texts,
            ..
        } = self;
        team.run(|ctx| {
            dit.time(ctx, buffers, timestep);
            dit.embed_image(ctx, buffers, image);
            dit.forward(ctx, buffers, &texts[branch as usize], image);
        });
    }

    /// The DiT under `branch`'s text, after `run_step` has refined the image.
    fn run_branch(&self, image: Size, branch: Branch) {
        let Self {
            dit,
            team,
            buffers,
            texts,
            ..
        } = self;
        team.run(|ctx| dit.forward(ctx, buffers, &texts[branch as usize], image));
    }

    /// One step of `job`: the image rounded to bfloat16 as the DiT's input, both branches'
    /// velocities, guided adaptively, then an Euler step.
    fn denoise(&mut self, job: &mut Job) {
        let time = job.schedule.times()[job.step];
        let step_size = job.schedule.dt(job.step);
        for (seen, &value) in job.seen.iter_mut().zip(&job.state) {
            *seen = round_bf16(value);
        }
        self.buffers.image.get_mut()[..job.seen.len()].copy_from_slice(&job.seen);
        self.run_step(job.image, timestep(time), Branch::Prompt);
        let predicted = &self.buffers.prediction.read()[..job.seen.len()];
        velocity(&job.seen, predicted, time, &mut job.conditional);
        self.run_branch(job.image, Branch::Negative);
        let predicted = &self.buffers.prediction.read()[..job.seen.len()];
        velocity(&job.seen, predicted, time, &mut job.other);
        job.guidance.round_running_to_bf16();
        let velocities = (&job.conditional[..], &job.other[..]);
        job.guidance
            .guide(velocities, job.conditional.len(), &mut job.guided);
        let cleaned = job.clean.iter_mut().zip(&job.seen).zip(&job.guided);
        for ((clean, &seen), &guided) in cleaned {
            *clean = seen - time * guided;
        }
        for (state, &clean) in job.state.iter_mut().zip(&job.clean) {
            *state += step_size * ((*state - clean) / time);
        }
        job.step += 1;
    }
}

/// `out` = the velocity `(seen − predicted) / time` of each value.
fn velocity(seen: &[f32], predicted: &[f32], time: f32, out: &mut [f32]) {
    for ((velocity, &input), &clean) in out.iter_mut().zip(seen).zip(predicted) {
        *velocity = (input - clean) / time;
    }
}

fn no_generation() -> Error {
    Error::Request("no generation in progress".into())
}

/// `image` is within the limits, and `values` are its pixels' channels.
fn check_image(image: Size, values: usize) -> Result<(), Error> {
    let sides = MIN_SIDE..=MAX_SIDE;
    let tokens = DIT.tokens(image);
    if sides.contains(&image.width) && sides.contains(&image.height) && tokens <= MAX_PATCHES {
        if values == image.pixels() * CHANNELS {
            return Ok(());
        }
        return Err(Error::Request(format!(
            "{values} image values for {}×{} RGB pixels",
            image.width, image.height
        )));
    }
    Err(Error::Request(format!(
        "{}×{}: sides from {MIN_SIDE} to {MAX_SIDE}, at most {MAX_PATCHES} patches of {} pixels",
        image.width, image.height, DIT.patch
    )))
}

fn check_run(steps: usize, guidance: f32) -> Result<(), Error> {
    if !(1..=MAX_STEPS).contains(&steps) {
        return Err(Error::Request(format!("{steps} steps: 1 to {MAX_STEPS}")));
    }
    if !(0.0..=MAX_GUIDANCE).contains(&guidance) {
        return Err(Error::Request(format!(
            "guidance {guidance}: 0 to {MAX_GUIDANCE}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Request, knots, timestep};
    use crate::Size;

    #[test]
    fn knots_round_once_as_torch_linspace_does() {
        let knots = knots(50);
        assert_eq!(knots[17], 0.66);
        assert_eq!(timestep(knots[17]), 660.0);
    }

    #[test]
    fn generates_only_the_trained_sizes() {
        let cases = [
            (1024, 1024, true),
            (864, 1216, true),
            (1152, 864, true),
            (1248, 832, true),
            (1280, 720, true),
            (1216, 864, false),
            (720, 1280, false),
            (648, 912, false),
            (200, 288, false),
        ];
        for (width, height, trained) in cases {
            let request = Request {
                width,
                height,
                steps: 1,
                guidance: 0.0,
                seed: 0,
            };
            let expected = Size { width, height };
            let accepted = matches!(request.size(), Ok(size) if size == expected);
            assert_eq!(accepted, trained, "{width}×{height}");
        }
    }
}
