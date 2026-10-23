//! Each question's answer from its options' probabilities, as the reference's `product_answer`
//! writes it, in f64 from the f32 probabilities: a choice's likeliest key with every option's
//! probability and the confidence, a noul's probability of `true`, a score's expected level
//! with its levels' probabilities, the confidence and the legend. The confidence is one minus
//! the distribution's normalized entropy.

use kernels::numpy::argmax;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::request::{Kind, Question};

/// Why a question has no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    InvalidQuestion,
    MaxLengthExceeded,
}

impl Failure {
    fn name(self) -> &'static str {
        match self {
            Failure::InvalidQuestion => "invalid_question",
            Failure::MaxLengthExceeded => "max_length_exceeded",
        }
    }
}

/// What System One returns: the model's name, each question's answer or failure under its
/// name in request order, and the input tokens read.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// A question of type `kind` (as the request gave it) that could not be answered.
pub fn failed(kind: &Value, failure: Failure, message: &str) -> Value {
    json!({
        "type": kind,
        "error": failure.name(),
        "message": message,
    })
}

/// The answer to `question` from its options' probabilities, in its options' order.
pub fn answer(question: &Question, probabilities: &[f32]) -> Value {
    let total: f64 = probabilities.iter().map(|&share| f64::from(share)).sum();
    let shares: Vec<f64> = probabilities
        .iter()
        .map(|&share| f64::from(share) / total)
        .collect();
    let by_key = |keys: &[String]| -> Map<String, Value> {
        keys.iter()
            .zip(&shares)
            .map(|(key, &share)| (key.clone(), share.into()))
            .collect()
    };
    match question.kind {
        Kind::Noul => json!({ "type": "noul", "noul": shares[1] }),
        Kind::Choice => json!({
            "type": "choice",
            "choice": question.keys[argmax(&shares)],
            "probabilities": by_key(&question.keys),
            "confidence": confidence(&shares),
        }),
        Kind::Score => {
            let expected: f64 = (0..)
                .zip(&shares)
                .map(|(level, share)| f64::from(level) * share)
                .sum();
            let legend: Map<String, Value> = question
                .keys
                .iter()
                .zip(&question.legend)
                .map(|(key, text)| (key.clone(), Value::String(text.clone())))
                .collect();
            json!({
                "type": "score",
                "score": expected,
                "probabilities": by_key(&question.keys),
                "confidence": confidence(&shares),
                "legend": legend,
            })
        }
    }
}

fn confidence(shares: &[f64]) -> f64 {
    if shares.len() < 2 {
        return 1.0;
    }
    let entropy: f64 = -shares
        .iter()
        .filter(|&&share| share > 0.0)
        .map(|&share| share * share.ln())
        .sum::<f64>();
    (1.0 - entropy / (shares.len() as f64).ln()).clamp(0.0, 1.0)
}
