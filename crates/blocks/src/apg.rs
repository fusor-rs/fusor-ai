//! Adaptive projected guidance (Sadat et al., 2025): the guided prediction
//! `c + scale · (⊥ + eta · ∥)` from the conditional prediction `c` and the unconditional `u`. The
//! difference `c − u` keeps a running sum across a run's steps (`r = d + momentum · r`); in each
//! group of values (a video's latent frame, an image) the sum is scaled down to a norm of at most
//! `rescale`, then split into its part along that group of `c` (`∥`) and the rest (`⊥`), the split
//! in f64 (diffusers' `normalized_guidance` under `use_original_formulation`, pjit-21k's
//! `adaptive_projected_guidance`). pjit-21k rounds the running sum to bfloat16 between steps
//! (`round_running_to_bf16`).

use kernels::ops::round_bf16;

/// `F.normalize`'s floor under a norm.
const NORM_FLOOR: f64 = 1e-12;

/// What a run guides with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub scale: f32,
    pub momentum: f32,
    pub rescale: f32,
    pub eta: f32,
}

/// A run's guidance: its settings and the running sum of the differences.
pub struct Guidance {
    settings: Settings,
    running: Vec<f32>,
}

impl Guidance {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            running: Vec::new(),
        }
    }

    /// Rounds the running sum to bfloat16, as P-JiT's sampler stores it between steps (call before
    /// `guide`).
    pub fn round_running_to_bf16(&mut self) {
        for value in &mut self.running {
            *value = round_bf16(*value);
        }
    }

    /// `guided` = the guided prediction of this step, groups of `group` consecutive values each
    /// normed and split alone. Steps run in order from the run's first.
    ///
    /// # Panics
    /// When the predictions' lengths differ or are not whole groups.
    pub fn guide(
        &mut self,
        (conditional, unconditional): (&[f32], &[f32]),
        group: usize,
        guided: &mut [f32],
    ) {
        assert!(
            conditional.len() == unconditional.len()
                && conditional.len() == guided.len()
                && conditional.len().is_multiple_of(group),
            "the predictions are whole groups of one length"
        );
        let differences = conditional
            .iter()
            .zip(unconditional)
            .map(|(conditional, unconditional)| conditional - unconditional);
        if self.running.is_empty() {
            self.running = differences.collect();
        } else {
            let momentum = self.settings.momentum;
            for (running, difference) in self.running.iter_mut().zip(differences) {
                *running = difference + momentum * *running;
            }
        }
        let groups = self
            .running
            .chunks_exact(group)
            .zip(conditional.chunks_exact(group))
            .zip(guided.chunks_exact_mut(group));
        for ((running, conditional), guided) in groups {
            self.settings.guide_group(running, conditional, guided);
        }
    }
}

impl Settings {
    /// One group of `guided`: `running` scaled down to `rescale`, split along `conditional`.
    fn guide_group(&self, running: &[f32], conditional: &[f32], guided: &mut [f32]) {
        let factor = if self.rescale > 0.0 {
            (f64::from(self.rescale) / norm(running)).min(1.0) as f32
        } else {
            1.0
        };
        let length = norm(conditional).max(NORM_FLOOR);
        let direction = |value: f32| f64::from(value) / length;
        let along: f64 = running
            .iter()
            .zip(conditional)
            .map(|(&sum, &value)| f64::from(sum * factor) * direction(value))
            .sum();
        for ((guided, &sum), &value) in guided.iter_mut().zip(running).zip(conditional) {
            let parallel = along * direction(value);
            let orthogonal = f64::from(sum * factor) - parallel;
            *guided = value + self.scale * (orthogonal as f32 + self.eta * parallel as f32);
        }
    }
}

/// `√Σ values²`, summed in f64.
fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::{Guidance, Settings};

    #[test]
    fn guidance_clips_projects_and_carries_momentum() {
        let mut guidance = Guidance::new(Settings {
            scale: 2.0,
            momentum: 0.5,
            rescale: 1.0,
            eta: 0.0,
        });
        let conditional = [3.0, 4.0, 0.0, 1.0];
        let steps = [
            ([1.0, 1.0, 0.0, 1.0], [2.911_248, 4.066_564, 0.0, 1.0]),
            ([3.0, 3.0, 0.0, 1.0], [2.584_042_4, 4.311_968, 0.0, 1.0]),
        ];
        for (unconditional, want) in steps {
            let mut guided = [0.0; 4];
            guidance.guide((&conditional, &unconditional), 2, &mut guided);
            for (got, want) in guided.iter().zip(want) {
                assert!((got - want).abs() < 1e-5, "{guided:?} against {want}");
            }
        }
    }

    #[test]
    fn a_running_sum_rounded_to_bf16_is_what_the_next_step_adds_to() {
        let mut guidance = Guidance::new(Settings {
            scale: 1.0,
            momentum: 1.0,
            rescale: 0.0,
            eta: 1.0,
        });
        let (conditional, unconditional) = ([2.0], [1.0 - 1.0 / 256.0]);
        let (first, second) = (3.0 + 1.0 / 256.0, 4.0 + 1.0 / 256.0);
        let mut guided = [0.0];
        guidance.guide((&conditional, &unconditional), 1, &mut guided);
        assert_eq!(guided, [first]);
        guidance.round_running_to_bf16();
        guidance.guide((&conditional, &unconditional), 1, &mut guided);
        assert_eq!(guided, [second]);
    }
}
