//! Drawing a frame's codes as the checkpoint's `generation_config.json` sets: the talker's first
//! code over a temperature of 0.9, the 50 likeliest kept, with every code already drawn
//! penalized once by 1.05 (`prepare`), and the code predictor's the same without the penalty. The
//! draws are `blocks::lm::sampling`'s; the random numbers are not torch's, so only greedy
//! decoding (`GREEDY`) matches the reference.

use std::collections::BTreeSet;

pub use blocks::lm::sampling::{Sampler, Sampling};

use crate::model::{CODEC_EOS, CODEC_VOCABULARY, CODES};

/// The model's sampling for the talker's and the code predictor's draws.
pub const SAMPLING: Sampling = Sampling {
    temperature: 0.9,
    top_p: 1.0,
    top_k: 50,
    presence_penalty: 0.0,
};

/// The likeliest code of every draw.
pub const GREEDY: Sampling = Sampling {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 1,
    presence_penalty: 0.0,
};

/// Divides the positive logit of a code already drawn, multiplies the negative one.
const REPETITION_PENALTY: f32 = 1.05;
/// The speech lasts at least this many frames.
const MIN_FRAMES: usize = 2;

/// The talker's `logits` for a first code, after `drawn` first codes: each distinct drawn code
/// penalized once, the end of speech out of reach before `MIN_FRAMES` frames, and every code
/// past `CODES` but the end of speech out of reach always.
pub fn prepare(logits: &mut [f32], drawn: &[u32]) {
    let distinct: BTreeSet<u32> = drawn.iter().copied().collect();
    for code in distinct {
        let logit = &mut logits[code as usize];
        *logit = if *logit < 0.0 {
            *logit * REPETITION_PENALTY
        } else {
            *logit / REPETITION_PENALTY
        };
    }
    if drawn.len() < MIN_FRAMES {
        logits[CODEC_EOS as usize] = f32::NEG_INFINITY;
    }
    for (code, logit) in logits[CODES..CODEC_VOCABULARY].iter_mut().enumerate() {
        if CODES + code != CODEC_EOS as usize {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// The likeliest first code after `drawn`: `prepare`d `logits`, the first of equals.
pub fn greedy(logits: &[f32], drawn: &[u32]) -> u32 {
    let mut logits = logits.to_vec();
    prepare(&mut logits, drawn);
    Sampler::new(GREEDY, 0).draw(&mut logits)
}

#[cfg(test)]
mod tests {
    use super::prepare;
    use crate::model::{CODEC_EOS, CODEC_VOCABULARY, CODES};

    /// Code 5 is drawn twice and penalized once (2.1 / 1.05), code 6's negative logit is
    /// multiplied (−2.1 · 1.05), and with fewer than two frames the end of speech goes too.
    #[test]
    fn penalizes_each_drawn_code_once_and_keeps_only_codes_and_the_end() {
        let mut logits = vec![1.0; CODEC_VOCABULARY];
        logits[5] = 2.1;
        logits[6] = -2.1;
        logits[CODEC_EOS as usize] = 3.0;
        prepare(&mut logits, &[5, 6, 5]);
        assert!((logits[5] - 2.0).abs() < 1e-6);
        assert!((logits[6] + 2.205).abs() < 1e-6);
        assert_eq!(logits[CODES - 1], 1.0);
        assert_eq!(logits[CODES], f32::NEG_INFINITY);
        assert_eq!(logits[CODEC_EOS as usize], 3.0);
        prepare(&mut logits, &[5]);
        assert_eq!(logits[CODEC_EOS as usize], f32::NEG_INFINITY);
    }
}
