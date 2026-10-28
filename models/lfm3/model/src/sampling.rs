//! Drawing a reply's next token (`blocks::lm::sampling`), the model card's settings the defaults
//! (`preset`), and a spoken frame's codes (`FrameSampler`), as the model's own generation draws
//! them: the codebooks revealed in a random order, each code's logits penalized for repeating
//! that codebook's recent codes, over a temperature, the `top_k` likeliest kept, then drawn.

use std::collections::VecDeque;

pub use blocks::lm::sampling::{Constraint, Sampler, Sampling};
use kernels::ops::softmax;
use kernels::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::audio::{CODEBOOKS, Frame};
use crate::model::Variant;

/// LFM3-1.7B's and LFM3-VL-2B's `generation_config.json`: a temperature of 0.1 over the 50
/// likeliest tokens. Its repetition penalty of 1.1 is not applied: the sampler penalizes by
/// presence.
pub const PRESET: Sampling = Sampling {
    temperature: 0.1,
    top_p: 1.0,
    top_k: 50,
    presence_penalty: 0.0,
};

/// LFM3-Audio-2B's and LFM3-Omni-2B's: greedy text.
pub const GREEDY: Sampling = Sampling {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 1,
    presence_penalty: 0.0,
};

/// The variant's text sampling, as its model card sets it.
pub fn preset(variant: Variant) -> Sampling {
    match variant {
        Variant::Text | Variant::Vision => PRESET,
        Variant::Audio | Variant::Omni => GREEDY,
    }
}

/// How a frame's codes are drawn. A temperature of 0 takes each codebook's likeliest code.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Voicing {
    pub temperature: f32,
    /// All codes when absent.
    pub top_k: Option<usize>,
    /// Divides a positive logit of a recent code, multiplies a negative one.
    pub repetition_penalty: f32,
    /// The frames whose codes count as recent.
    pub window: usize,
}

/// The variant's frame sampling, as its `generation_config.json` sets it.
pub fn voicing(variant: Variant) -> Voicing {
    match variant {
        Variant::Omni => Voicing {
            temperature: 0.4,
            top_k: None,
            repetition_penalty: 1.2,
            window: 64,
        },
        Variant::Text | Variant::Vision | Variant::Audio => Voicing {
            temperature: 1.0,
            top_k: Some(128),
            repetition_penalty: 1.1,
            window: 4096,
        },
    }
}

/// One reply's frames: the settings, the seeded generator, the recent frames and each draw's
/// scratch.
pub struct FrameSampler {
    voicing: Voicing,
    rng: Rng,
    recent: VecDeque<Frame>,
    codes: Vec<u32>,
    ranked: Vec<f32>,
}

impl FrameSampler {
    pub fn new(voicing: Voicing, seed: u64) -> Self {
        Self {
            voicing,
            rng: Rng::new(seed),
            recent: VecDeque::new(),
            codes: Vec::new(),
            ranked: Vec::new(),
        }
    }

    /// The order a frame's codebooks are revealed in, uniformly random.
    pub fn order(&mut self) -> [usize; CODEBOOKS] {
        let mut order = std::array::from_fn(|codebook| codebook);
        for last in (1..CODEBOOKS).rev() {
            let pick = self.rng.choose(&[1.0; CODEBOOKS][..=last]);
            order.swap(last, pick);
        }
        order
    }

    /// Codebook `codebook`'s code from its `logits` (changed in place).
    pub fn draw(&mut self, codebook: usize, logits: &mut [f32]) -> u32 {
        let Voicing {
            temperature,
            top_k,
            repetition_penalty,
            ..
        } = self.voicing;
        self.codes.clear();
        self.codes
            .extend(self.recent.iter().map(|frame| frame[codebook]));
        self.codes.sort_unstable();
        self.codes.dedup();
        for &code in &self.codes {
            let logit = &mut logits[code as usize];
            *logit = if *logit < 0.0 {
                *logit * repetition_penalty
            } else {
                *logit / repetition_penalty
            };
        }
        if temperature == 0.0 {
            let codes = 0..logits.len();
            let likeliest = codes.reduce(|best, code| {
                if logits[code] > logits[best] {
                    code
                } else {
                    best
                }
            });
            return likeliest.expect("a codebook has codes") as u32;
        }
        for logit in logits.iter_mut() {
            *logit /= temperature;
        }
        if let Some(kept) = top_k.filter(|&kept| kept < logits.len()) {
            self.ranked.clear();
            self.ranked.extend_from_slice(logits);
            let (_, &mut least, _) = self
                .ranked
                .select_nth_unstable_by(kept - 1, |first, second| second.total_cmp(first));
            for logit in logits.iter_mut().filter(|logit| **logit < least) {
                *logit = f32::NEG_INFINITY;
            }
        }
        softmax(logits);
        self.rng.choose(logits) as u32
    }

    /// The frame drawn, recent for the next ones.
    pub fn keep(&mut self, frame: &Frame) {
        if self.recent.len() == self.voicing.window {
            self.recent.pop_front();
        }
        self.recent.push_back(*frame);
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameSampler, Voicing};

    /// Recent codes 0 and 2: a positive logit is divided by the penalty and a negative one
    /// multiplied, so 2.0 / 1.25 = 1.6 falls below code 1's 1.9, and −1.0 becomes −1.25.
    #[test]
    fn penalizes_a_codebooks_recent_codes() {
        let voicing = Voicing {
            temperature: 0.0,
            top_k: None,
            repetition_penalty: 1.25,
            window: 4,
        };
        let mut sampler = FrameSampler::new(voicing, 0);
        sampler.keep(&[0; 12]);
        sampler.keep(&[2; 12]);
        let mut logits = [2.0, 1.9, -1.0];
        assert_eq!(sampler.draw(0, &mut logits), 1);
        assert_eq!(logits, [1.6, 1.9, -1.25]);
    }
}
