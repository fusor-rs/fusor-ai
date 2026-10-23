//! Drawing a token's half from its logits as the reference's `sample_from_logits` does: the
//! logits over a temperature, then the `top_k` most likely kept or, without a `top_k`, the
//! smallest set of most likely values whose probability reaches `top_p` (nucleus sampling),
//! drawn in proportion to their probabilities.

use kernels::ops::softmax;
use kernels::rng::Rng;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct Sampling {
    pub temperature: f32,
    /// Keep this many values; 0 keeps them by `top_p` instead. 1 is greedy.
    pub top_k: usize,
    pub top_p: f32,
}

/// The reference predictor's defaults.
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
        }
    }
}

impl Sampling {
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> usize {
        let mut probabilities: Vec<f32> = logits
            .iter()
            .map(|logit| logit / self.temperature)
            .collect();
        softmax(&mut probabilities);
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|&first, &second| probabilities[second].total_cmp(&probabilities[first]));
        let kept = if self.top_k > 0 {
            let least = probabilities[order[self.top_k.min(order.len()) - 1]];
            order
                .iter()
                .take_while(|&&index| probabilities[index] >= least)
                .count()
        } else {
            self.nucleus(&probabilities, &order)
        };
        for &index in &order[kept..] {
            probabilities[index] = 0.0;
        }
        rng.choose(&probabilities)
    }

    /// How many of the most likely values, in `order`, nucleus sampling keeps: each while the
    /// probability of those before it is at most `top_p`.
    fn nucleus(&self, probabilities: &[f32], order: &[usize]) -> usize {
        if self.top_p >= 1.0 {
            return order.len();
        }
        let mut before = 0.0;
        let mut kept = 0;
        for &index in order {
            if kept > 0 && before > self.top_p {
                break;
            }
            before += probabilities[index];
            kept += 1;
        }
        kept
    }
}
