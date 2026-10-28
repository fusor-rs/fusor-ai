//! A request as d1's `system_one` takes it: a state (text, any JSON value, or none when the media
//! are the whole state) and named questions in the Decision Index's schema, each checked as the
//! reference's `check_question` checks it and turned into what its prompt asks
//! (`as_question`). A question that fails its check fails the request, naming the question.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Error;
use crate::grid::Grid;
use crate::model::PromptVersion;

/// The levels of a score question, fewest and most.
const SCORE_LEVELS: (usize, usize) = (2, 10);
/// The most values an estimate's grid has.
const ESTIMATE_VALUES: usize = 128;
/// The blocks a `boxes` question reads without `max`, and the most its `max` asks of a
/// checkpoint of the second prompt version.
pub const INSTANCES: usize = 16;
const MAX_INSTANCES: usize = 64;
const TYPES: &str = "noul, choice, score, text, number, date, box, boxes";

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub state: Value,
    pub questions: Map<String, Value>,
}

/// A checked question: its instructions and what it asks.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub instructions: String,
    pub kind: Kind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Yes or no, each perhaps described (`criteria`'s `true` and `false`).
    Noul(Option<Map<String, Value>>),
    /// One of the options, or with `multiple` each a yes or no of its own.
    Choice {
        options: Vec<Criterion>,
        multiple: bool,
    },
    /// One of 2 to 10 ordered levels, lowest first.
    Score(Vec<String>),
    /// A span read from the state.
    Read {
        value: Written,
        unit: Option<String>,
    },
    /// A number on a grid over a range.
    Estimate { grid: Grid, unit: Option<String> },
    /// Where one thing is in the picture.
    Box,
    /// Every instance of something in the picture: this many blocks read.
    Boxes(usize),
}

/// A choice's option: its name, and its description when it has one.
#[derive(Debug, Clone, PartialEq)]
pub struct Criterion {
    pub name: String,
    pub description: Option<String>,
}

/// What a read reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    Text,
    Number,
    Date,
}

impl Written {
    pub fn name(self) -> &'static str {
        match self {
            Written::Text => "text",
            Written::Number => "number",
            Written::Date => "date",
        }
    }
}

impl Kind {
    /// Whether a box's or boxes' question, which reads exactly one picture.
    pub fn boxed(&self) -> bool {
        matches!(self, Kind::Box | Kind::Boxes(_))
    }

    /// Whether the answer is read off the model raw and calibrated by its own table, not
    /// through the temperature: a read, an estimate, a box, a multiple choice.
    pub fn typed(&self) -> bool {
        !matches!(
            self,
            Kind::Noul(_)
                | Kind::Score(_)
                | Kind::Choice {
                    multiple: false,
                    ..
                }
        )
    }
}

impl Request {
    /// The request's questions in order, under their names.
    ///
    /// # Errors
    /// When the request holds no question, or a question fails its check.
    pub fn questions(&self, version: PromptVersion) -> Result<Vec<(String, Question)>, Error> {
        if self.questions.is_empty() {
            return Err(Error::Request(
                "questions are a mapping of names to questions, {\"refund\": {\"type\": \"noul\", \
                 \"instructions\": \"Is it a refund request?\"}}"
                    .into(),
            ));
        }
        let checked = self.questions.iter().map(|(name, question)| {
            let question = question
                .as_object()
                .ok_or_else(|| "a question is an object with a type".to_owned())
                .and_then(|question| check(question, version));
            question
                .map(|question| (name.clone(), question))
                .map_err(|problem| Error::Request(format!("question {name:?}: {problem}")))
        });
        checked.collect()
    }
}

fn check(question: &Map<String, Value>, version: PromptVersion) -> Result<Question, String> {
    let kind = question.get("type").and_then(Value::as_str).unwrap_or("");
    let instructions = match question.get("instructions") {
        Some(Value::String(text)) if !bundle::python::strip(text).is_empty() => text.clone(),
        Some(Value::Object(_)) => {
            return Err("spoken instructions are not read by this runtime: write them".into());
        }
        _ => return Err(r#""instructions" is the question as text"#.into()),
    };
    if question.contains_key("multiple") && kind != "choice" {
        return Err(r#""multiple" belongs to a choice"#.into());
    }
    if question.contains_key("max") && kind != "boxes" {
        return Err(r#""max" belongs to boxes"#.into());
    }
    if question.contains_key("step") && !question.contains_key("range") {
        return Err(
            r#""step" needs a "range": {"type": "number", "range": [0, 10], "step": 1}"#.into(),
        );
    }
    if question.contains_key("range") && kind != "number" {
        return Err(r#"an estimate is {"type": "number", "range": [low, high]}"#.into());
    }
    let unit = || {
        question
            .get("unit")
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let kind = match kind {
        "noul" => noul(question.get("criteria"))?,
        "choice" => choice(question)?,
        "score" => score(question.get("criteria"))?,
        "number" if question.contains_key("range") => Kind::Estimate {
            grid: estimate(question)?,
            unit: unit(),
        },
        "text" | "number" | "date" => Kind::Read {
            value: match kind {
                "text" => Written::Text,
                "number" => Written::Number,
                _ => Written::Date,
            },
            unit: unit(),
        },
        "box" => Kind::Box,
        "boxes" => Kind::Boxes(blocks(question.get("max"), version)?),
        other => return Err(format!(r#""type" is one of {TYPES}, not {other:?}"#)),
    };
    Ok(Question { instructions, kind })
}

fn noul(criteria: Option<&Value>) -> Result<Kind, String> {
    match criteria {
        None | Some(Value::Null) => Ok(Kind::Noul(None)),
        Some(Value::Object(described)) => Ok(Kind::Noul(Some(described.clone()))),
        Some(_) => Err(
            "a noul's \"criteria\", when given, is {\"true\": \"what yes means\", \"false\": \
             \"what no means\"}"
                .into(),
        ),
    }
}

fn choice(question: &Map<String, Value>) -> Result<Kind, String> {
    let invalid = || {
        "a choice's \"criteria\" names the options: {\"billing\": \"Charges and refunds\", \
         \"shipping\": \"Delivery\"}"
            .to_owned()
    };
    let criteria = question
        .get("criteria")
        .and_then(Value::as_object)
        .filter(|criteria| !criteria.is_empty())
        .ok_or_else(invalid)?;
    let options = criteria.iter().map(|(name, description)| {
        let description = match description {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            _ => return Err(invalid()),
        };
        Ok(Criterion {
            name: name.clone(),
            description,
        })
    });
    let multiple = match question.get("multiple") {
        None => false,
        Some(Value::Bool(multiple)) => *multiple,
        Some(_) => return Err(r#""multiple" is true or false"#.into()),
    };
    Ok(Kind::Choice {
        options: options.collect::<Result<_, _>>()?,
        multiple,
    })
}

fn score(criteria: Option<&Value>) -> Result<Kind, String> {
    let (fewest, most) = SCORE_LEVELS;
    let levels = criteria
        .and_then(Value::as_array)
        .filter(|levels| (fewest..=most).contains(&levels.len()))
        .and_then(|levels| {
            let texts = levels.iter().map(|level| level.as_str().map(str::to_owned));
            texts.collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            format!(
                "a score's \"criteria\" lists {fewest} to {most} levels, lowest first: \
                 [\"Can wait\", \"Today\", \"Now\"]"
            )
        })?;
    Ok(Kind::Score(levels))
}

/// An estimate's grid: `step` apart over its range, at most `ESTIMATE_VALUES` values.
fn estimate(question: &Map<String, Value>) -> Result<Grid, String> {
    let bounds = question
        .get("range")
        .and_then(Value::as_array)
        .filter(|bounds| bounds.len() == 2)
        .and_then(|bounds| Some((bounds[0].as_f64()?, bounds[1].as_f64()?)));
    let step = question.get("step").map(Value::as_f64);
    let (Some((low, high)), None | Some(Some(_))) = (bounds, step) else {
        return Err(
            r#"an estimate is {"type": "number", "range": [low, high]}, with an optional "step""#
                .into(),
        );
    };
    let step = step.flatten();
    let steps = step.map(|step| (high - low) / step);
    let divides =
        steps.is_none_or(|steps| (steps - steps.round_ties_even()).abs() <= 1e-9 * steps.abs());
    if low >= high || step.is_some_and(|step| step <= 0.0) || !divides {
        return Err(format!(
            "an estimate's range runs low to high and its step, above 0, divides it: not [{low}, \
             {high}] in steps of {step:?}"
        ));
    }
    let size = steps.map_or(ESTIMATE_VALUES, |steps| {
        ESTIMATE_VALUES.min(steps.round_ties_even() as usize + 1)
    });
    Ok(Grid { low, high, size })
}

/// The blocks a `boxes` question reads: its `max`, else `INSTANCES`.
fn blocks(max: Option<&Value>, version: PromptVersion) -> Result<usize, String> {
    let most = match version {
        PromptVersion::First => INSTANCES,
        PromptVersion::Second => MAX_INSTANCES,
    };
    match max {
        None => Ok(INSTANCES),
        Some(max) => max
            .as_u64()
            .filter(|&max| (1..=most as u64).contains(&max))
            .map(|max| max as usize)
            .ok_or_else(|| format!(r#""max" is a whole number from 1 to {most}"#)),
    }
}
