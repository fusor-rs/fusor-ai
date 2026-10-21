//! DPM-Solver++(2M) (Lu et al., 2022) for diffusion models trained to predict the noise `ε` of a
//! discrete-time variance-preserving schedule, `x = α·x₀ + σ·ε`, sampled at the noise levels of
//! Karras et al. (2022) from the schedule's largest to its smallest, then 0: diffusers'
//! `DPMSolverMultistepScheduler` with `use_karras_sigmas` and its defaults otherwise.
//!
//! A noise level is `s = σ / α`. The model is asked at the training timestep whose level is
//! `s`, interpolated in `ln s` between neighbouring timesteps and rounded. A step turns its
//! prediction into the clean sample it implies, `x₀ = (x − σ ε) / α`, and moves `x` to the next
//! level with the last two such predictions, the first step and the last (to level 0, which lands
//! on `x₀`) with one.

/// Karras et al.'s `ρ`, which spaces the levels.
const RHO: f64 = 7.0;

/// The noise levels of a discrete-time schedule, one per training timestep, increasing.
pub struct NoiseSchedule {
    levels: Vec<f32>,
}

impl NoiseSchedule {
    /// `β` linear in `√β` from `√start` to `√end` over `timesteps` ("scaled_linear"), computed
    /// in f32 as diffusers does (`torch.linspace` from both ends, `cumprod` accumulated in f64).
    pub fn scaled_linear(start: f64, end: f64, timesteps: usize) -> Self {
        let (first, last) = (start.sqrt() as f32, end.sqrt() as f32);
        let step = (last - first) / (timesteps - 1) as f32;
        let halfway = timesteps / 2;
        let mut product = 1.0f64;
        let levels = (0..timesteps).map(|timestep| {
            let root = if timestep < halfway {
                first + step * timestep as f32
            } else {
                last - step * (timesteps - 1 - timestep) as f32
            };
            product *= f64::from(1.0 - root * root);
            let cumulative = product as f32;
            ((1.0 - cumulative) / cumulative).sqrt()
        });
        Self {
            levels: levels.collect(),
        }
    }

    /// The training timestep of level `level`: interpolated in `ln s` between the two timesteps
    /// around it (clamped to the schedule), rounded half to even.
    fn timestep(&self, level: f64) -> f64 {
        let log = level.max(1e-10).ln();
        let logs = |timestep: usize| f64::from(self.levels[timestep].ln());
        let below = self
            .levels
            .iter()
            .rposition(|&known| f64::from(known.ln()) <= log);
        let low = below.unwrap_or(0).min(self.levels.len() - 2);
        let weight = ((logs(low) - log) / (logs(low) - logs(low + 1))).clamp(0.0, 1.0);
        ((1.0 - weight) * low as f64 + weight * (low + 1) as f64).round_ties_even()
    }
}

/// The levels a sampling run visits and the timestep the model is asked at for each.
pub struct Schedule {
    /// `steps + 1` levels, the last 0.
    levels: Vec<f32>,
    timesteps: Vec<f32>,
}

impl Schedule {
    /// `steps` levels from the largest of `noise` to its smallest, `(s_max^(1/ρ) + r (s_min^(1/ρ)
    /// − s_max^(1/ρ)))^ρ` for `r` evenly from 0 to 1, then 0.
    ///
    /// # Panics
    /// When `steps` is 0.
    pub fn karras(noise: &NoiseSchedule, steps: usize) -> Self {
        assert!(steps > 0, "a run has steps");
        let smallest = f64::from(noise.levels[0]).powf(1.0 / RHO);
        let largest = f64::from(noise.levels[noise.levels.len() - 1]).powf(1.0 / RHO);
        let span = (steps - 1).max(1) as f64;
        let levels: Vec<f64> = (0..steps)
            .map(|step| (largest + step as f64 / span * (smallest - largest)).powf(RHO))
            .collect();
        let timesteps = levels.iter().map(|&level| noise.timestep(level) as f32);
        Self {
            timesteps: timesteps.collect(),
            levels: levels
                .iter()
                .map(|&level| level as f32)
                .chain([0.0])
                .collect(),
        }
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }

    /// The timestep the model is asked at in step `step`.
    pub fn timestep(&self, step: usize) -> f32 {
        self.timesteps[step]
    }

    /// The level of each step, then 0.
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }
}

/// `(α, σ)` and `λ = ln(α / σ)` of noise level `s`.
fn coefficients(level: f32) -> (f64, f64, f64) {
    let level = f64::from(level);
    let alpha = 1.0 / (level * level + 1.0).sqrt();
    let sigma = level * alpha;
    (alpha, sigma, (alpha / sigma).ln())
}

/// A sampling run's state between steps: the clean sample the previous step predicted.
#[derive(Default)]
pub struct Solver {
    previous: Vec<f32>,
}

impl Solver {
    /// Moves `sample` from level `step` to level `step + 1` of `schedule`, given the model's
    /// noise `noise` (classifier-free guidance applied); `denoised` gets the clean sample this
    /// step predicts. Steps run in order from 0.
    pub fn step(
        &mut self,
        (schedule, step): (&Schedule, usize),
        sample: &mut [f32],
        noise: &[f32],
        denoised: &mut [f32],
    ) {
        let levels = schedule.levels();
        let (alpha, sigma, lambda) = coefficients(levels[step]);
        for ((clean, x), epsilon) in denoised.iter_mut().zip(sample.iter()).zip(noise) {
            *clean = ((f64::from(*x) - sigma * f64::from(*epsilon)) / alpha) as f32;
        }
        let next = levels[step + 1];
        if next == 0.0 {
            sample.copy_from_slice(denoised);
        } else {
            let (next_alpha, next_sigma, next_lambda) = coefficients(next);
            let span = next_lambda - lambda;
            let keep = (next_sigma / sigma) as f32;
            let towards = next_alpha * (1.0 - (-span).exp());
            let weight = (step > 0).then(|| {
                let (_, _, last_lambda) = coefficients(levels[step - 1]);
                (0.5 * towards * span / (lambda - last_lambda)) as f32
            });
            for (i, x) in sample.iter_mut().enumerate() {
                let correction =
                    weight.map_or(0.0, |weight| weight * (denoised[i] - self.previous[i]));
                *x = keep * *x + towards as f32 * denoised[i] + correction;
            }
        }
        self.previous.clear();
        self.previous.extend_from_slice(denoised);
    }
}

#[cfg(test)]
mod tests {
    use super::{NoiseSchedule, Schedule};

    /// Stable Diffusion's schedule at 12 steps, as diffusers 0.41's
    /// `DPMSolverMultistepScheduler(use_karras_sigmas=True)` gives it.
    #[test]
    fn karras_levels_and_timesteps_match_diffusers() {
        let noise = NoiseSchedule::scaled_linear(0.000_85, 0.012, 1000);
        let schedule = Schedule::karras(&noise, 12);
        let timesteps: Vec<f32> = (0..12).map(|step| schedule.timestep(step)).collect();
        let want = [
            999.0, 933.0, 854.0, 761.0, 646.0, 506.0, 345.0, 190.0, 78.0, 23.0, 4.0, 0.0,
        ];
        assert_eq!(timesteps, want);
        let levels = [
            14.614647,
            9.945206,
            6.617722,
            4.2940726,
            2.7080412,
            1.6531787,
            0.97211784,
            0.5472423,
            0.2926245,
            0.14713623,
            0.06863961,
            0.029167533,
            0.0,
        ];
        assert_eq!(schedule.levels(), levels);
    }
}
