//! Answers from the logits of each question's options' symbols, as the reference reads them:
//! their softmax (in f32), recalibrated by the checkpoint's temperature (in f64, `softmax(log p /
//! T)`), the most likely option the decision (the lexically first of equals).

use kernels::ops::softmax;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::request::{Kind, Question, score};

pub const MODEL: &str = "Intern-Decision-2B";
/// The calibration temperature fitted for this checkpoint (its model card's).
pub const TEMPERATURE: f64 = 2.100_509_348_278;
const CALIBRATION: &str = "temperature-scaling";
const SOURCE: &str = "local";

/// The response, in the reference's envelope.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub answers: Map<String, Value>,
    pub usage: Usage,
    pub calibration: Calibration,
    pub model: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    /// Questions answered, not tokens written: the model writes none.
    pub output_tokens: usize,
    pub decision_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    pub method: &'static str,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Answer {
    #[serde(rename = "type")]
    kind: Kind,
    probabilities: Map<String, Value>,
    confidence: f64,
    #[serde(flatten)]
    reading: Reading,
    source: &'static str,
    decision: String,
}

/// What an answer reads, by its question's kind.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum Reading {
    Choice {
        choice: String,
    },
    Noul {
        noul: f64,
    },
    Score {
        score: f64,
        legend: Map<String, Value>,
    },
}

/// The response to `questions` from each one's options' logits, after `input_tokens` rows.
pub fn respond(questions: &[Question], logits: &[Vec<f32>], input_tokens: usize) -> Response {
    let answers = questions
        .iter()
        .zip(logits)
        .map(|(question, logits)| {
            let answer = answer(question, logits);
            let answer = serde_json::to_value(answer).expect("an answer serializes");
            (question.name.clone(), answer)
        })
        .collect();
    Response {
        answers,
        usage: Usage {
            input_tokens,
            output_tokens: questions.len(),
            decision_count: questions.len(),
        },
        calibration: Calibration {
            method: CALIBRATION,
            temperature: TEMPERATURE,
        },
        model: MODEL,
    }
}

fn answer(question: &Question, logits: &[f32]) -> Answer {
    let probabilities = calibrated(logits);
    let values = question.candidates.iter().map(|candidate| &candidate.value);
    let ranked = values.clone().zip(&probabilities);
    let (decision, &confidence) = ranked
        .clone()
        .min_by(|(value, probability), (other, other_probability)| {
            other_probability
                .total_cmp(probability)
                .then_with(|| value.cmp(other))
        })
        .expect("a question has an option");
    let reading = match question.kind {
        Kind::Choice => Reading::Choice {
            choice: decision.clone(),
        },
        Kind::Noul => Reading::Noul {
            noul: probabilities[1],
        },
        Kind::Score => Reading::Score {
            score: ranked
                .clone()
                .map(|(value, probability)| {
                    let value = score(value).expect("a score's values are numbers (`questions`)");
                    value * probability
                })
                .sum(),
            legend: question
                .candidates
                .iter()
                .map(|candidate| {
                    (
                        candidate.value.clone(),
                        candidate.description.clone().into(),
                    )
                })
                .collect(),
        },
    };
    Answer {
        kind: question.kind,
        probabilities: ranked
            .map(|(value, &probability)| (value.clone(), probability.into()))
            .collect(),
        confidence,
        reading,
        source: SOURCE,
        decision: decision.clone(),
    }
}

/// The options' probabilities: the softmax of their logits, then `softmax(log p / T)`.
fn calibrated(logits: &[f32]) -> Vec<f64> {
    let mut probabilities = logits.to_vec();
    softmax(&mut probabilities);
    let logs: Vec<f64> = probabilities
        .iter()
        .map(|&probability| f64::from(probability).ln())
        .collect();
    let most = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logs
        .iter()
        .map(|log| ((log - most) / TEMPERATURE).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}
