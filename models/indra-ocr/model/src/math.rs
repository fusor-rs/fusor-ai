//! Math lines: the math classifier (`classifier`) flags lines that hold math; the encoder
//! reads a 64-high crop of each, and the decoder writes its text with inline LaTeX
//! (`we have $x _ { i } \leq 1$`), one token per step, greedily, until the end token or until
//! its mean token probability falls so low that the reading would be rejected anyway.

pub mod decoder;
pub mod encoder;

use kernels::numpy::{argmax, mean, sum};
use serde::Deserialize;
use team::Team;

use crate::Error;
use crate::crops::Batching;
use crate::plane::Scratch;
use decoder::{Decoder, MAX_TOKENS, Reading, TOKENS};
use encoder::Memory;

/// The classifier's probability over which a line holds math.
pub const MATH_MIN: f32 = 0.5;
/// A reading with math replaces a line at this mean token probability or more.
pub const MATH_MIN_CONFIDENCE: f64 = 0.9;
/// A line whose mean token probability is this far under `MATH_MIN_CONFIDENCE` after
/// `EARLY_STOP_TOKENS` tokens stops.
const EARLY_STOP_MARGIN: f64 = 0.1;
const EARLY_STOP_TOKENS: usize = 8;
const START: u32 = 1;
const END: u32 = 2;
/// The first token that is text.
const SPECIAL: u32 = 4;
/// Few, wide batches: each is one decoding loop.
pub const MATH_BATCHING: Batching = Batching {
    max_lines: 64,
    max_ratio: 4.0,
    min_width: 64,
    max_area: 16 * 64 * 640,
};
const MATH_SPAN: &str = "$";

/// The decoder's text tokens, as `math_meta.json` lists them after the special ones.
pub struct MathTokens {
    tokens: Vec<String>,
}

#[derive(Deserialize)]
struct StoredTokens {
    tokens: Vec<String>,
}

impl MathTokens {
    pub fn parse(json: &[u8]) -> Result<Self, Error> {
        let stored: StoredTokens = serde_json::from_slice(json)
            .map_err(|error| Error::Format(format!("math_meta.json: {error}")))?;
        if stored.tokens.len() + SPECIAL as usize != TOKENS {
            return Err(Error::Format(format!(
                "math_meta.json: {} tokens for a decoder of {TOKENS}",
                stored.tokens.len()
            )));
        }
        Ok(Self {
            tokens: stored.tokens,
        })
    }

    /// Text from tokens: inside math (between `$` tokens) joined by spaces, white space runs
    /// as one space.
    pub fn detokenize(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut math: Vec<&str> = Vec::new();
        let mut in_math = false;
        for &id in ids {
            let token = self.tokens[(id - SPECIAL) as usize].as_str();
            if token == MATH_SPAN {
                if in_math {
                    out.push_str(&format!("${}$", math.join(" ")));
                    math.clear();
                }
                in_math = !in_math;
            } else if in_math {
                math.push(token);
            } else {
                out.push_str(token);
            }
        }
        if in_math && !math.is_empty() {
            out.push_str(&format!("${}$", math.join(" ")));
        }
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// A line's math reading: its text tokens and the probability of each token chosen.
#[derive(Debug, Clone, Default)]
pub struct MathReading {
    pub ids: Vec<u32>,
    pub probabilities: Vec<f64>,
}

impl MathReading {
    /// The mean token probability, 0 without tokens.
    pub fn confidence(&self) -> f64 {
        if self.probabilities.is_empty() {
            0.0
        } else {
            mean(&self.probabilities)
        }
    }

    /// Takes the step's token and its probability; whether the line is done.
    fn take(&mut self, token: u32, probability: f64) -> bool {
        self.probabilities.push(probability);
        if token == END {
            return true;
        }
        let rejected = self.probabilities.len() >= EARLY_STOP_TOKENS
            && self.confidence() < MATH_MIN_CONFIDENCE - EARLY_STOP_MARGIN;
        if !rejected && token >= SPECIAL {
            self.ids.push(token);
        }
        rejected
    }
}

/// Each line of an encoded batch read greedily from the start token.
pub fn decode(
    decoder: &Decoder,
    team: &Team,
    scratch: &Scratch,
    memory: &Memory,
) -> Vec<MathReading> {
    let lines = memory.lengths.len();
    let mut state = decoder.start(team, scratch, memory);
    let mut readings = vec![MathReading::default(); lines];
    let mut active: Vec<Reading> = (0..lines)
        .map(|line| Reading { line, token: START })
        .collect();
    for _ in 0..MAX_TOKENS - 1 {
        if active.is_empty() {
            break;
        }
        let logits = decoder.step(team, scratch, &mut state, &active);
        let mut next = Vec::with_capacity(active.len());
        for (reading, logits) in active.iter().zip(logits.as_chunks::<TOKENS>().0) {
            let (token, probability) = best_token(logits);
            if !readings[reading.line].take(token, f64::from(probability)) {
                next.push(Reading {
                    line: reading.line,
                    token,
                });
            }
        }
        active = next;
    }
    readings
}

/// The likeliest token and its softmax probability (the reciprocal of the sum of the
/// exponentials, as the reference computes it in f32).
fn best_token(logits: &[f32]) -> (u32, f32) {
    let best = argmax(logits);
    let max = logits[best];
    let exponentials: Vec<f32> = logits.iter().map(|&logit| (logit - max).exp()).collect();
    (best as u32, 1.0 / sum(&exponentials))
}
