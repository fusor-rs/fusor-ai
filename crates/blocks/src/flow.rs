//! Flow-matching integration: Euler steps from noise at `t = 0` to a sample at `t = 1` over a
//! fixed knot schedule, with classifier-free guidance on the steps that ask for it.

use crate::SINUSOID_BASE;

/// The knots `0 = t_0 < … < t_steps = 1` and, per step, whether it is guided (evaluates the
/// unconditional branch too).
pub struct Schedule {
    knots: Vec<f32>,
    guided: Vec<bool>,
}

impl Schedule {
    /// `guided(step, t_step)` decides which steps use classifier-free guidance.
    pub fn new(knots: Vec<f32>, guided: impl Fn(usize, f32) -> bool) -> Self {
        let guided = knots
            .iter()
            .take(knots.len() - 1)
            .enumerate()
            .map(|(step, &time)| guided(step, time))
            .collect();
        Self { knots, guided }
    }

    pub fn steps(&self) -> usize {
        self.guided.len()
    }

    /// The time each step starts at.
    pub fn times(&self) -> &[f32] {
        &self.knots[..self.steps()]
    }

    pub fn guided(&self, step: usize) -> bool {
        self.guided[step]
    }

    pub fn dt(&self, step: usize) -> f32 {
        self.knots[step + 1] - self.knots[step]
    }
}

/// Sway sampling (F5-TTS): `t + coefficient · (cos(π t / 2) − 1 + t)`; −1 crowds the knots
/// towards `t = 0`.
pub fn sway(time: f32, coefficient: f32) -> f32 {
    time + coefficient * ((std::f32::consts::FRAC_PI_2 * time).cos() - 1.0 + time)
}

/// The sinusoidal timestep embedding `[sin(scale·t·fᵢ), cos(scale·t·fᵢ)]` for
/// `i < half`, `fᵢ = exp(−i · ln(10000) / (half − 1))`.
pub fn timestep_embedding(time: f32, half: usize, scale: f32) -> Vec<f32> {
    let step = SINUSOID_BASE.ln() / (half - 1) as f32;
    let angles: Vec<f32> = (0..half)
        .map(|i| scale * time * (i as f32 * -step).exp())
        .collect();
    let sines = angles.iter().map(|angle| angle.sin());
    sines
        .chain(angles.iter().map(|angle| angle.cos()))
        .collect()
}

/// A guided step's unconditional prediction `u` and its guidance scale.
#[derive(Clone, Copy)]
pub struct Guidance<'step> {
    pub unconditional: &'step [f32],
    pub scale: f32,
}

/// One Euler step `y += dt · v`, with `v = c + (c − u) · scale` for a guided step and `v = c`
/// otherwise.
pub fn euler_step(y: &mut [f32], conditional: &[f32], guidance: Option<Guidance>, dt: f32) {
    for (i, value) in y.iter_mut().enumerate() {
        let prediction = conditional[i];
        let velocity = match guidance {
            Some(Guidance {
                unconditional,
                scale,
            }) => prediction + (prediction - unconditional[i]) * scale,
            None => prediction,
        };
        *value += velocity * dt;
    }
}
