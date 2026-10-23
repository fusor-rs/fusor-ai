//! A request as System One takes it: a state (text or any JSON) and questions, each a `choice`
//! among described options, a yes-or-no `noul`, or a `score` over ordered levels; and each
//! question checked and turned into what its prompt lists, as the reference's
//! `normalize_question` does. A question that fails its check is answered with the error, the
//! others still answered.

use bundle::python::{self, Layout};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Error;
use crate::model::CODES;

/// Levels a score question takes, fewest and most.
const SCORE_LEVELS: (usize, usize) = (2, 10);
/// What the `false` and `true` options of a noul question say when it does not describe them.
const NOUL_DESCRIPTIONS: [(&str, &str); 2] = [("false", "No / false"), ("true", "Yes / true")];

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub state: Value,
    pub questions: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Noul => "noul",
            Kind::Score => "score",
        }
    }
}

impl From<Kind> for Value {
    fn from(kind: Kind) -> Self {
        Value::from(kind.name())
    }
}

/// A question of the request under its name, checked or with the reason it failed.
#[derive(Debug, Clone)]
pub struct Asked {
    pub name: String,
    pub question: Result<Question, Invalid>,
}

/// A question that failed its check: its type as the request gave it, and why.
#[derive(Debug, Clone)]
pub struct Invalid {
    pub kind: Value,
    pub message: String,
}

/// A checked question: its instructions and options as its prompt shows them, and the keys its
/// answer reports them under, in the same order.
#[derive(Debug, Clone)]
pub struct Question {
    pub kind: Kind,
    /// The instructions' text, or `None` when the question gives none.
    pub instructions: Option<String>,
    pub options: Vec<String>,
    pub keys: Vec<String>,
    /// A score's levels as its answer's legend shows them.
    pub legend: Vec<String>,
}

impl Request {
    /// The request's questions in order, each checked or with the reason it was not.
    ///
    /// # Errors
    /// When the request holds no question, or a question has an empty name.
    pub fn questions(&self) -> Result<Vec<Asked>, Error> {
        if self.questions.is_empty() {
            return Err(Error::Request(
                "questions must be a nonempty mapping of question IDs to questions".into(),
            ));
        }
        if self.questions.keys().any(String::is_empty) {
            return Err(Error::Request(
                "question IDs must be nonempty strings".into(),
            ));
        }
        let checked = self.questions.iter().map(|(name, question)| {
            let invalid = |message: String| Invalid {
                kind: question.get("type").cloned().unwrap_or_default(),
                message,
            };
            let checked = match question.as_object() {
                Some(question) => check(question).map_err(invalid),
                None => Err(invalid("a question is an object with a type".into())),
            };
            Asked {
                name: name.clone(),
                question: checked,
            }
        });
        Ok(checked.collect())
    }
}

/// A string as it is, any other JSON value as `json.dumps` writes it (the reference's
/// `describe`).
pub(crate) fn describe(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python::dumps(other, Layout::Inline),
    }
}

/// Whether Python reads `value` as false: `None`, `False`, zero, or an empty string, list or
/// object.
fn falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(flag) => !flag,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(entries) => entries.is_empty(),
    }
}

fn check(question: &Map<String, Value>) -> Result<Question, String> {
    let instructions = question
        .get("instructions")
        .filter(|instructions| !falsy(instructions))
        .map(describe);
    let criteria = question.get("criteria");
    let listed = match question.get("type").and_then(Value::as_str) {
        Some("choice") => choice(criteria)?,
        Some("noul") => noul(criteria)?,
        Some("score") => score(criteria)?,
        other => {
            let shown = other.map_or("none".into(), |kind| format!("{kind:?}"));
            return Err(format!(
                "unsupported question type {shown} (choice, noul or score)"
            ));
        }
    };
    Ok(Question {
        instructions,
        ..listed
    })
}

/// A question of `kind` listing `options` under `keys`, without instructions yet.
fn listed(kind: Kind, keys: Vec<String>, options: Vec<String>, legend: Vec<String>) -> Question {
    Question {
        kind,
        instructions: None,
        options,
        keys,
        legend,
    }
}

/// Each option under its key: the key alone when it has no description.
fn choice(criteria: Option<&Value>) -> Result<Question, String> {
    let criteria = criteria
        .and_then(Value::as_object)
        .filter(|criteria| (1..=CODES).contains(&criteria.len()))
        .ok_or(format!(
            "choice criteria must map 1 to {CODES} option keys to descriptions"
        ))?;
    if criteria.keys().any(String::is_empty) {
        return Err("choice option keys must be nonempty strings".into());
    }
    let options = criteria.iter().map(|(key, description)| match description {
        Value::Null => key.clone(),
        described => format!("{key}: {}", describe(described)),
    });
    let keys = criteria.keys().cloned().collect();
    Ok(listed(Kind::Choice, keys, options.collect(), Vec::new()))
}

/// `false` then `true`, each with its description or the default one.
fn noul(criteria: Option<&Value>) -> Result<Question, String> {
    let empty = Map::new();
    let given = match criteria {
        None | Some(Value::Null) => &empty,
        Some(Value::Object(given)) if given.keys().all(|key| key == "false" || key == "true") => {
            given
        }
        Some(_) => return Err(r#"noul criteria may only describe "false" and "true""#.into()),
    };
    let options = NOUL_DESCRIPTIONS.map(|(key, fallback)| match given.get(key) {
        Some(description) if !falsy(description) => describe(description),
        _ => fallback.into(),
    });
    let keys = NOUL_DESCRIPTIONS.map(|(key, _)| key.to_string());
    Ok(listed(Kind::Noul, keys.into(), options.into(), Vec::new()))
}

/// The ordered levels, each shown by its text alone, reported under its index.
fn score(criteria: Option<&Value>) -> Result<Question, String> {
    let (fewest, most) = SCORE_LEVELS;
    let levels = criteria
        .and_then(Value::as_array)
        .filter(|levels| (fewest..=most).contains(&levels.len()))
        .ok_or(format!(
            "score criteria must be an ordered list of {fewest} to {most} levels"
        ))?;
    let options: Vec<String> = levels.iter().map(describe).collect();
    let distinct = options
        .iter()
        .enumerate()
        .all(|(index, option)| !option.is_empty() && !options[..index].contains(option));
    if !distinct {
        return Err("score levels must be distinct and nonempty".into());
    }
    let keys = (0..levels.len()).map(|level| level.to_string()).collect();
    let legend = levels.iter().map(canonical).collect();
    Ok(listed(Kind::Score, keys, options, legend))
}

/// A level as its legend shows it: a string as it is, anything else as compact JSON with its
/// keys sorted (the reference's `_canonical`).
fn canonical(level: &Value) -> String {
    match level {
        Value::String(text) => text.clone(),
        other => python::dumps(&sorted(other), Layout::Compact),
    }
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort();
            let entries = keys
                .into_iter()
                .map(|key| (key.clone(), sorted(&entries[key])));
            Value::Object(entries.collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        scalar => scalar.clone(),
    }
}
