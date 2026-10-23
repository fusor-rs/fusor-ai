//! A seeded random number generator for sampling: xoshiro256** seeded through SplitMix64,
//! normals by Box–Muller and categorical draws. Deterministic across hosts, and needs no entropy
//! import.

pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut mixed = seed;
        let mut next = || {
            mixed = mixed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = mixed;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let state = &mut self.state;
        let result = state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let shifted = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= shifted;
        state[3] = state[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal values.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for pair in out.chunks_mut(2) {
            let radius = (-2.0 * self.uniform().ln()).sqrt();
            let angle = std::f64::consts::TAU * self.uniform();
            pair[0] = (radius * angle.cos()) as f32;
            if let Some(second) = pair.get_mut(1) {
                *second = (radius * angle.sin()) as f32;
            }
        }
    }

    /// An index drawn with probability proportional to its weight.
    ///
    /// # Panics
    /// When no weight is positive.
    pub fn choose(&mut self, weights: &[f32]) -> usize {
        let total: f64 = weights.iter().map(|&weight| f64::from(weight)).sum();
        let mut remaining = self.uniform() * total;
        for (index, &weight) in weights.iter().enumerate() {
            remaining -= f64::from(weight);
            if weight > 0.0 && remaining <= 0.0 {
                return index;
            }
        }
        weights
            .iter()
            .rposition(|&weight| weight > 0.0)
            .expect("a positive weight")
    }
}
