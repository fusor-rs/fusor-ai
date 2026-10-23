//! A decision request, as the reference's `DecisionEngine.predict` takes it: a state of any JSON,
//! and named questions in order, each a choice among described options, a score on a numeric
//! scale, or a yes-or-no (`noul`) decision.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;
use crate::json;

/// The most questions a request holds.
pub const MAX_QUESTIONS: usize = 16;
/// The options' single-token symbols, in order: a question has at most one option per symbol.
pub const SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/// A `noul` question's options' descriptions when its criteria name none.
const YES: &str = "The answer is yes (affirmative, or align with the claim).";
const NO: &str = "The answer is no (negative, or disagree with the claim).";
/// The keys, in lower case, that describe a `noul` question's yes and its no.
const YES_KEYS: [&str; 3] = ["yes", "true", "1"];
const NO_KEYS: [&str; 3] = ["no", "false", "0"];

/// The request as JSON gives it, its maps in their order.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub state: Value,
    pub questions: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Choice,
    Score,
    Noul,
}

/// One option of a question: the value an answer names, and its description.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub value: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub name: String,
    pub kind: Kind,
    pub instructions: String,
    pub candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Asked {
    #[serde(rename = "type")]
    kind: Kind,
    #[serde(default)]
    instructions: Option<Value>,
    #[serde(default)]
    criteria: Option<Value>,
}

impl Request {
    /// The questions, checked as the reference checks them.
    pub fn questions(&self) -> Result<Vec<Question>, Error> {
        if !(1..=MAX_QUESTIONS).contains(&self.questions.len()) {
            return Err(invalid(format!("supply 1–{MAX_QUESTIONS} questions")));
        }
        self.questions
            .iter()
            .map(|(name, asked)| question(name, asked))
            .collect()
    }
}

fn question(name: &str, asked: &Value) -> Result<Question, Error> {
    if name.is_empty() {
        return Err(invalid("question names must be nonempty".into()));
    }
    let asked = Asked::deserialize(asked).map_err(|error| {
        invalid(format!(
            "question {name:?}: {error} (its type must be choice, score or noul)"
        ))
    })?;
    let candidates = match asked.kind {
        Kind::Choice => choices(asked.criteria.as_ref())?,
        Kind::Score => scores(asked.criteria.as_ref())?,
        Kind::Noul => yes_or_no(asked.criteria.as_ref())?,
    };
    let symbols = SYMBOLS.len();
    if !(1..=symbols).contains(&candidates.len()) {
        return Err(invalid(format!(
            "question {name:?}: supply 1–{symbols} options"
        )));
    }
    let instructions = match &asked.instructions {
        Some(instructions) => json::text(instructions)?,
        None => String::new(),
    };
    Ok(Question {
        name: name.into(),
        kind: asked.kind,
        instructions,
        candidates,
    })
}

/// A choice's options: its criteria's keys, described by their values.
fn choices(criteria: Option<&Value>) -> Result<Vec<Candidate>, Error> {
    let Some(Value::Object(criteria)) = criteria else {
        return Err(invalid("choice criteria must be an object".into()));
    };
    described(criteria)
}

/// A score's options: a list's indices, or an object's numeric keys, described by their values.
fn scores(criteria: Option<&Value>) -> Result<Vec<Candidate>, Error> {
    match criteria {
        Some(Value::Array(levels)) => levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                Ok(Candidate {
                    value: index.to_string(),
                    description: json::text(level)?,
                })
            })
            .collect(),
        Some(Value::Object(levels)) => {
            if levels.keys().any(|key| score(key).is_none()) {
                return Err(invalid("score option keys must be finite numbers".into()));
            }
            described(levels)
        }
        _ => Err(invalid("score criteria must be a list or an object".into())),
    }
}

/// A `noul` question's options, no then yes, described by the first criteria keys naming them.
fn yes_or_no(criteria: Option<&Value>) -> Result<Vec<Candidate>, Error> {
    let criteria = match criteria {
        Some(Value::Object(criteria)) => Some(criteria),
        _ => None,
    };
    let describe = |keys: [&str; 3], fallback: &str| {
        let found = criteria.and_then(|criteria| {
            criteria
                .iter()
                .find(|(key, _)| keys.contains(&key.to_lowercase().as_str()))
        });
        match found {
            Some((_, description)) => json::text(description),
            None => Ok(fallback.into()),
        }
    };
    Ok(vec![
        Candidate {
            value: "no".into(),
            description: describe(NO_KEYS, NO)?,
        },
        Candidate {
            value: "yes".into(),
            description: describe(YES_KEYS, YES)?,
        },
    ])
}

fn described(criteria: &Map<String, Value>) -> Result<Vec<Candidate>, Error> {
    criteria
        .iter()
        .map(|(value, description)| {
            Ok(Candidate {
                value: value.clone(),
                description: json::text(description)?,
            })
        })
        .collect()
}

/// A score option's value as a number, `float(value)` in the reference: `None` when it is not a
/// finite one.
pub(crate) fn score(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn invalid(message: String) -> Error {
    Error::Request(message)
}
