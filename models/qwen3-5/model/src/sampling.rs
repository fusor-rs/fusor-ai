//! Drawing a reply's next token from its logits as serving frameworks do (vLLM's order): every
//! token already in the reply penalized once by `presence_penalty`, the logits over a
//! temperature, the `top_k` most likely kept, then the fewest of those whose probability reaches
//! `top_p`, drawn in proportion to their probabilities. A temperature of 0 takes the most likely
//! token instead (the first of equals). The model card's settings are the defaults (`preset`).

use std::collections::BTreeSet;

use kernels::ops::softmax;
use kernels::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::session::Message;

/// The model card's `top_k`, in every mode.
const PRESET_TOP_K: usize = 20;

/// What a reply must keep to, held by its host: a grammar or a JSON schema, token by token. A
/// reply with none draws from every token.
pub trait Constraint: Send {
    /// Leaves the tokens the reply may take next, setting every other one's logit to negative
    /// infinity.
    fn mask(&mut self, logits: &mut [f32]);
    /// The token the reply took.
    fn take(&mut self, token: u32);
}

/// What a conversation holds: text only, or images or videos too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Content {
    Text,
    Vision,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
}

impl Sampling {
    /// The model card's settings for `message`, in a conversation of `content`.
    pub fn preset(message: &Message, content: Content) -> Self {
        let (temperature, top_p, presence_penalty) = match (message.thinking, content) {
            (false, Content::Text) => (1.0, 1.0, 2.0),
            (false, Content::Vision) => (0.7, 0.8, 1.5),
            (true, Content::Text) => (1.0, 0.95, 1.5),
            (true, Content::Vision) => (0.6, 0.95, 0.0),
        };
        Self {
            temperature,
            top_p,
            top_k: PRESET_TOP_K,
            presence_penalty,
        }
    }

    /// # Errors
    /// When the temperature is negative, `top_p` is outside (0, 1] or `top_k` is 0.
    pub fn check(&self) -> Result<(), Error> {
        let valid = self.temperature >= 0.0 && self.top_p > 0.0 && self.top_p <= 1.0;
        if !valid || self.top_k == 0 {
            return Err(Error::Request(
                "sampling takes a temperature of at least 0, a top_p in (0, 1] and a top_k of at \
                 least 1"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// One reply's draws: the settings, the seeded generator, the tokens drawn so far and the scratch
/// of every draw.
pub struct Sampler {
    sampling: Sampling,
    rng: Rng,
    drawn: BTreeSet<u32>,
    /// Token ids, the likeliest first.
    order: Vec<u32>,
    probabilities: Vec<f32>,
}

impl Sampler {
    pub fn new(sampling: Sampling, seed: u64) -> Self {
        Self {
            sampling,
            rng: Rng::new(seed),
            drawn: BTreeSet::new(),
            order: Vec::new(),
            probabilities: Vec::new(),
        }
    }

    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// The reply's next token, from `logits` (penalized in place).
    pub fn draw(&mut self, logits: &mut [f32]) -> u32 {
        let Sampling {
            temperature,
            top_p,
            top_k,
            presence_penalty,
        } = self.sampling;
        for &token in &self.drawn {
            logits[token as usize] -= presence_penalty;
        }
        let likelier = |first: &u32, second: &u32| {
            let order = logits[*second as usize].total_cmp(&logits[*first as usize]);
            order.then(first.cmp(second))
        };
        let token = if temperature == 0.0 {
            let tokens = 0..logits.len() as u32;
            tokens.min_by(likelier).expect("logits are not empty")
        } else {
            let kept = top_k.min(logits.len());
            self.order.clear();
            self.order.extend(0..logits.len() as u32);
            self.order.select_nth_unstable_by(kept - 1, likelier);
            let order = &mut self.order[..kept];
            order.sort_unstable_by(likelier);
            self.probabilities.clear();
            let scaled = order
                .iter()
                .map(|&token| logits[token as usize] / temperature);
            self.probabilities.extend(scaled);
            softmax(&mut self.probabilities);
            order[self
                .rng
                .choose(&self.probabilities[..nucleus(&self.probabilities, top_p)])]
        };
        self.drawn.insert(token);
        token
    }
}

/// How many of the likeliest-first `probabilities` nucleus sampling keeps: each while those
/// before it hold less than `top_p`, at least one, and all of them at a `top_p` of 1.
fn nucleus(probabilities: &[f32], top_p: f32) -> usize {
    if top_p >= 1.0 {
        return probabilities.len();
    }
    let mut before = 0.0;
    let kept = probabilities.iter().take_while(|&&probability| {
        let within = before < top_p;
        before += probability;
        within
    });
    kept.count().max(1)
}
