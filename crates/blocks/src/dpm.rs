//! DPM-Solver++(2M) (Lu et al., 2022) for diffusion models trained to predict the noise of a
//! discrete-time variance-preserving schedule at Karras levels (diffusers'
//! `DPMSolverMultistepScheduler(use_karras_sigmas=True)`), or the velocity of flow matching at
//! shifted levels (Wan's `FlowDPMSolverMultistepScheduler`, diffusers' with `use_flow_sigmas`,
//! Iris's `FlowDPMSolver` at unrounded timesteps).
//!
//! A variance-preserving sample is `x = α·x₀ + σ·ε` at noise level `s = σ / α`, sampled at the
//! levels of Karras et al. (2022) from the schedule's largest to its smallest, then 0; the model is
//! asked at the training timestep whose level is `s`, interpolated in `ln s` between neighbouring
//! timesteps and rounded. A flow-matching sample is `x = (1 − σ)·x₀ + σ·ε` at level `σ`, and the
//! model predicts the velocity `ε − x₀`. A step turns the prediction into the clean sample it
//! implies and moves `x` to the next level with the last two such predictions, the first step and
//! the last (to level 0, which lands on `x₀`) with one.

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

/// How a schedule's levels relate a sample to its clean part, and what the model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Process {
    /// `x = α·x₀ + σ·ε` at level `s = σ/α`, `α = 1/√(s² + 1)`; the model predicts `ε`.
    VariancePreserving,
    /// `x = (1 − σ)·x₀ + σ·ε` at level `σ`; the model predicts the velocity `ε − x₀`.
    Flow,
}

impl Process {
    /// `(α, σ)` and `λ = ln(α / σ)` of `level`; `α = 0` at flow level 1 gives `λ = −∞`.
    fn coefficients(self, level: f32) -> (f64, f64, f64) {
        let level = f64::from(level);
        let (alpha, sigma) = match self {
            Process::VariancePreserving => {
                let alpha = 1.0 / (level * level + 1.0).sqrt();
                (alpha, level * alpha)
            }
            Process::Flow => (1.0 - level, level),
        };
        (alpha, sigma, (alpha / sigma).ln())
    }

    /// The clean sample `x` implies under `prediction`.
    fn clean(self, (alpha, sigma): (f64, f64), x: f32, prediction: f32) -> f32 {
        let numerator = f64::from(x) - sigma * f64::from(prediction);
        match self {
            Process::VariancePreserving => (numerator / alpha) as f32,
            Process::Flow => numerator as f32,
        }
    }
}

/// The levels a sampling run visits and the timestep the model is asked at for each.
pub struct Schedule {
    process: Process,
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
            process: Process::VariancePreserving,
            timesteps: timesteps.collect(),
            levels: levels
                .iter()
                .map(|&level| level as f32)
                .chain([0.0])
                .collect(),
        }
    }

    /// `steps` flow-matching levels for `sᵢ = 1 − i/steps` (Wan's `get_sampling_sigmas` with its
    /// `FlowDPMSolverMultistepScheduler`), shifted as `shifted` does.
    ///
    /// # Panics
    /// When `steps` is 0.
    pub fn flow(steps: usize, shift: f64, timesteps: usize) -> Self {
        assert!(steps > 0, "a run has steps");
        let spacing = -1.0 / steps as f64;
        let samples = (0..steps).map(|step| step as f64 * spacing + 1.0);
        Self::shifted(samples, shift, Self::floored(timesteps))
    }

    /// `steps` flow-matching levels `σᵢ = shift·sᵢ / (1 + (shift − 1)·sᵢ)` for `sᵢ = largest·(1 −
    /// i/steps)`, then 0, each asked at timestep `σᵢ · timesteps` unrounded (Iris's
    /// `FlowDPMSolver`, whose levels start at its training schedule's largest, 0.999), in f64.
    ///
    /// # Panics
    /// When `steps` is 0.
    pub fn flow_continuous(steps: usize, shift: f64, largest: f64, timesteps: usize) -> Self {
        assert!(steps > 0, "a run has steps");
        let spacing = -1.0 / steps as f64;
        let samples = (0..steps).map(|step| largest * (step as f64 * spacing + 1.0));
        Self::shifted(samples, shift, move |level| level * timesteps as f64)
    }

    /// `steps` flow-matching levels for `sᵢ = 1 − αₖ`, `k = steps … 1`, of `α` from 1 to
    /// `1/timesteps` in `steps` equal parts (diffusers' `DPMSolverMultistepScheduler` with
    /// `use_flow_sigmas`: NumPy's `linspace`, its last value exact), shifted as `shifted` does.
    ///
    /// # Panics
    /// When `steps` is 0.
    pub fn flow_sigmas(steps: usize, shift: f64, timesteps: usize) -> Self {
        assert!(steps > 0, "a run has steps");
        let last = 1.0 / timesteps as f64;
        let spacing = (last - 1.0) / steps as f64;
        let samples = (1..=steps).rev().map(|index| {
            let alpha = if index == steps {
                last
            } else {
                index as f64 * spacing + 1.0
            };
            1.0 - alpha
        });
        Self::shifted(samples, shift, Self::floored(timesteps))
    }

    /// The timestep of a level as Wan and diffusers ask it: `⌊σ · timesteps⌋`.
    fn floored(timesteps: usize) -> impl Fn(f64) -> f64 {
        move |level| (level * timesteps as f64).trunc()
    }

    /// Flow-matching levels `σᵢ = shift·sᵢ / (1 + (shift − 1)·sᵢ)` of the unshifted `samples`,
    /// then 0, each asked at `timestep(σᵢ)`, in f64 as NumPy computes them.
    fn shifted(
        samples: impl Iterator<Item = f64>,
        shift: f64,
        timestep: impl Fn(f64) -> f64,
    ) -> Self {
        let levels: Vec<f64> = samples
            .map(|sample| shift * sample / (1.0 + (shift - 1.0) * sample))
            .collect();
        Self {
            process: Process::Flow,
            timesteps: levels.iter().map(|&level| timestep(level) as f32).collect(),
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

/// A sampling run's state between steps: the clean sample the previous step predicted.
#[derive(Default)]
pub struct Solver {
    previous: Vec<f32>,
}

impl Solver {
    /// Moves `sample` from level `step` to level `step + 1` of `schedule`, given the model's
    /// prediction (the noise or the velocity, classifier-free guidance applied); `denoised` gets
    /// the clean sample this step predicts. Steps run in order from 0.
    pub fn step(
        &mut self,
        (schedule, step): (&Schedule, usize),
        sample: &mut [f32],
        prediction: &[f32],
        denoised: &mut [f32],
    ) {
        let levels = schedule.levels();
        let process = schedule.process;
        let (alpha, sigma, lambda) = process.coefficients(levels[step]);
        for ((clean, x), predicted) in denoised.iter_mut().zip(sample.iter()).zip(prediction) {
            *clean = process.clean((alpha, sigma), *x, *predicted);
        }
        let next = levels[step + 1];
        if next == 0.0 {
            sample.copy_from_slice(denoised);
        } else {
            let (next_alpha, next_sigma, next_lambda) = process.coefficients(next);
            let span = next_lambda - lambda;
            let keep = (next_sigma / sigma) as f32;
            let towards = next_alpha * (1.0 - (-span).exp());
            let weight = (step > 0).then(|| {
                let (_, _, last_lambda) = process.coefficients(levels[step - 1]);
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

    /// Wan's own values for 8 steps at shift 5, printed by its code.
    #[test]
    fn flow_levels_and_timesteps_match_wan() {
        let schedule = Schedule::flow(8, 5.0, 1000);
        let timesteps: Vec<f32> = (0..8).map(|step| schedule.timestep(step)).collect();
        let want = [1000.0, 972.0, 937.0, 892.0, 833.0, 750.0, 625.0, 416.0];
        assert_eq!(timesteps, want);
        let levels = [
            1.0,
            0.972_222_2,
            0.9375,
            0.892_857_13,
            0.833_333_3,
            0.75,
            0.625,
            0.416_666_66,
            0.0,
        ];
        assert_eq!(schedule.levels(), levels);
    }

    /// Motif-Video's scheduler: diffusers' values for 8 steps at shift 15, from NumPy 2.5.4.
    #[test]
    fn flow_sigmas_levels_and_timesteps_match_diffusers() {
        let schedule = Schedule::flow_sigmas(8, 15.0, 1000);
        let timesteps: Vec<f32> = (0..8).map(|step| schedule.timestep(step)).collect();
        let want = [999.0, 990.0, 978.0, 961.0, 937.0, 899.0, 833.0, 681.0];
        assert_eq!(timesteps, want);
        let levels = [
            0.999_933_24,
            0.990_491_2,
            0.978_175_7,
            0.961_439_7,
            0.937_382_7,
            0.899_855_85,
            0.833_148,
            0.681_570_1,
            0.0,
        ];
        assert_eq!(schedule.levels(), levels);
    }

    /// Iris's `FlowDPMSolver.time_grid(4, 4)`: levels from 0.999 at shift 4, timesteps unrounded.
    #[test]
    fn flow_continuous_levels_match_iris() {
        let schedule = Schedule::flow_continuous(4, 4.0, 0.999, 1000);
        let timesteps: Vec<f32> = (0..4).map(|step| schedule.timestep(step)).collect();
        assert_eq!(timesteps, [999.749_8, 922.792_7, 799.679_8, 571.101_9]);
        let levels = [0.999_749_84, 0.922_792_7, 0.799_679_8, 0.571_101_9, 0.0];
        assert_eq!(schedule.levels(), levels);
    }
}
