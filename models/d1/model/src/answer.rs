//! Each question's answer from what its slots read, as d1's reference writes it (`api.py`,
//! `box.py`): a noul's P(yes); a choice's likeliest option with every option's probability; a
//! multiple choice's options at P(yes) ≥ 0.5; a score's median level; an estimate's median and
//! central 80 percent interval; a read's text and span, a number's or a date's value parsed from
//! it (`parsing`); a box's four edges as fractions of the picture, its presence and confidence;
//! every instance's box in reading order. Typed answers' probabilities go through the
//! calibration's tables.

use serde_json::{Map, Value, json};

use crate::Error;
use crate::calibration::Calibration;
use crate::grid::Grid;
use crate::model::Edges;
use crate::parsing::{self, Dated};
use crate::pointer::{self, StatePositions};
use crate::request::{Criterion, Kind, Question, Written};

/// Boxes drawn from a box's edge distributions to measure its confidence.
const DRAWS: usize = 512;
/// The overlap at which a drawn box counts as the answer's.
const OVERLAP: f64 = 0.5;
/// A probability at which an answer is present.
const PRESENT: f64 = 0.5;
/// Slots of a box: its presence, then its edges.
const BOX_SLOTS: usize = 5;

/// What one slot read: its options' probabilities, or a read's logits over the state's
/// positions.
#[derive(Debug, Clone)]
pub enum Output {
    Probabilities(Vec<f64>),
    Positions(Vec<f32>),
}

impl Output {
    fn probabilities(&self) -> &[f64] {
        match self {
            Output::Probabilities(probabilities) => probabilities,
            Output::Positions(_) => unreachable!("a code slot reads probabilities"),
        }
    }

    fn positions(&self) -> &[f32] {
        match self {
            Output::Positions(logits) => logits,
            Output::Probabilities(_) => unreachable!("a read's slots read positions"),
        }
    }
}

/// How a request reads its boxes: the grid of their edges and how an edge is read.
#[derive(Debug, Clone, Copy)]
pub struct Boxing {
    pub grid: Grid,
    pub edges: Edges,
}

/// The answer to `question` from its slots' `outputs`, in order; a read's from the state's
/// `positions`.
///
/// # Errors
/// When a read's span leaves the state, or a pattern gives up on its text.
pub fn answer(
    question: &Question,
    outputs: &[Output],
    calibration: &Calibration,
    positions: Option<&StatePositions>,
    boxing: Boxing,
) -> Result<Value, Error> {
    let first = || outputs[0].probabilities();
    Ok(match &question.kind {
        Kind::Noul(_) => json!({ "type": "noul", "noul": first()[0] }),
        Kind::Choice {
            options,
            multiple: true,
        } => multiple(options, outputs, calibration),
        Kind::Choice { options, .. } => choice(options, first()),
        Kind::Score(levels) => score(levels, first()),
        Kind::Estimate { grid, unit } => {
            let level = |share| calibration.level("estimate", share);
            let quantiles = grid.quantiles(first(), level);
            json!({
                "type": "number",
                "number": quantiles.median,
                "interval": [quantiles.p10, quantiles.p90],
                "unit": unit,
            })
        }
        Kind::Read { value, unit } => {
            let positions = positions.expect("a request with a read keeps its state's positions");
            read(*value, unit.as_deref(), outputs, calibration, positions)?
        }
        Kind::Box => one_box(outputs, calibration, boxing),
        Kind::Boxes(_) => every_box(outputs, calibration, boxing),
    })
}

/// Each option's calibrated P(yes), and those at one half or more.
fn multiple(options: &[Criterion], outputs: &[Output], calibration: &Calibration) -> Value {
    let probabilities: Map<String, Value> = options
        .iter()
        .zip(outputs)
        .map(|(option, output)| {
            let yes = calibration.through("multiple", output.probabilities()[0]);
            (option.name.clone(), yes.into())
        })
        .collect();
    let chosen = probabilities
        .iter()
        .filter(|(_, yes)| yes.as_f64() >= Some(PRESENT));
    let chosen: Vec<&String> = chosen.map(|(name, _)| name).collect();
    json!({ "type": "choice", "choices": chosen, "probabilities": probabilities })
}

/// The likeliest option, its probability and every option's.
fn choice(options: &[Criterion], probabilities: &[f64]) -> Value {
    let best = likeliest(probabilities);
    let by_name: Map<String, Value> = options
        .iter()
        .zip(probabilities)
        .map(|(option, &probability)| (option.name.clone(), probability.into()))
        .collect();
    json!({
        "type": "choice",
        "choice": options[best].name,
        "confidence": probabilities[best],
        "probabilities": by_name,
    })
}

/// A read's text and span, and a number's or a date's value parsed from the text, when its
/// calibrated P(found) is one half or more.
fn read(
    value: Written,
    unit: Option<&str>,
    outputs: &[Output],
    calibration: &Calibration,
    positions: &StatePositions,
) -> Result<Value, Error> {
    let (start, end) = (outputs[0].positions(), outputs[1].positions());
    let (raw, span) = pointer::span(start, end, pointer::cap(value));
    let found = calibration.through(&format!("found:{}", value.name()), raw);
    let read = match span {
        Some((first, last)) if found >= PRESENT => Some(positions.read(first, last)?),
        _ => None,
    };
    let text = read.as_ref().map(|read| read.text.as_str());
    let context = positions.text();
    let (parsed, unit) = match (value, text) {
        (Written::Number, Some(text)) => (
            parsing::parse_number(text, &context, unit)?.into(),
            unit.or(parsing::written_unit(text)?),
        ),
        (Written::Date, Some(text)) => match parsing::parse_date(text, &context)? {
            Dated::Date(date) => (date.into(), unit),
            Dated::Ambiguous | Dated::Unread => (Value::Null, unit),
        },
        _ => (text.into(), unit),
    };
    let mut answer = Map::new();
    answer.insert("type".into(), value.name().into());
    answer.insert(value.name().into(), parsed);
    answer.insert(
        "span".into(),
        read.as_ref().map(|read| read.span.to_vec()).into(),
    );
    answer.insert("found".into(), found.into());
    if value != Written::Text {
        answer.insert("text".into(), text.into());
    }
    if let Some(path) = read.as_ref().and_then(|read| read.path.clone()) {
        answer.insert("path".into(), path.into());
    }
    if value == Written::Number {
        answer.insert("unit".into(), unit.into());
    }
    Ok(Value::Object(answer))
}

/// The first likeliest option.
fn likeliest(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > probabilities[best] {
            best = index;
        }
    }
    best
}

/// A score's median level: the first whose cumulative probability reaches one half.
fn score(levels: &[String], probabilities: &[f64]) -> Value {
    let total: f64 = probabilities.iter().sum();
    let mut mass = 0.0;
    let median = probabilities.iter().position(|probability| {
        mass += probability / total;
        mass >= PRESENT
    });
    let median = median.unwrap_or(levels.len() - 1);
    let keyed = |values: Vec<Value>| -> Map<String, Value> {
        values
            .into_iter()
            .enumerate()
            .map(|(level, value)| (level.to_string(), value))
            .collect()
    };
    json!({
        "type": "score",
        "score": median as f64,
        "confidence": probabilities[likeliest(probabilities)],
        "probabilities": keyed(probabilities.iter().map(|&share| share.into()).collect()),
        "legend": keyed(levels.iter().map(|text| text.clone().into()).collect()),
    })
}

/// A box's top, bottom, left and right, as fractions of the picture.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rectangle {
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
}

impl Rectangle {
    /// The box of edges `[top, bottom, left, right]`: crossed edges of an axis become one bin
    /// about their middle.
    fn new(edges: [f64; 4], grid: Grid) -> Self {
        let ordered = |first: f64, second: f64| {
            if first < second {
                return (first, second);
            }
            let half = grid.step() / 2.0;
            let middle = f64::midpoint(first, second).clamp(half, 1.0 - half);
            (middle - half, middle + half)
        };
        let (top, bottom) = ordered(edges[0], edges[1]);
        let (left, right) = ordered(edges[2], edges[3]);
        Self {
            top,
            bottom,
            left,
            right,
        }
    }

    fn area(&self) -> f64 {
        (self.bottom - self.top) * (self.right - self.left)
    }

    fn overlap(&self, other: &Self) -> f64 {
        let height = (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0);
        let width = (self.right.min(other.right) - self.left.max(other.left)).max(0.0);
        let shared = height * width;
        shared / (self.area() + other.area() - shared)
    }

    fn json(&self) -> Value {
        json!({ "top": self.top, "bottom": self.bottom, "left": self.left, "right": self.right })
    }
}

/// The box at its four edge distributions' medians, each its bin's value or where its CDF
/// crosses one half inside that bin.
fn medians(edges: &[Output], boxing: Boxing) -> Rectangle {
    let Boxing { grid, edges: read } = boxing;
    let at = |output: &Output| match read {
        Edges::Median => grid.quantile(output.probabilities(), 0.5),
        Edges::Interp => crossing(output.probabilities(), grid),
    };
    Rectangle::new([0, 1, 2, 3].map(|edge| at(&edges[edge])), grid)
}

/// Where a distribution's CDF crosses one half, the mass of each bin spread evenly over it.
fn crossing(probabilities: &[f64], grid: Grid) -> f64 {
    let total: f64 = probabilities.iter().sum();
    let step = grid.step();
    let mut mass = 0.0;
    for (value, probability) in grid.values().into_iter().zip(probabilities) {
        let share = probability / total;
        if share > 0.0 && mass + share >= PRESENT {
            let crossed = value - step / 2.0 + (PRESENT - mass) / share * step;
            return crossed.clamp(grid.low, grid.high);
        }
        mass += share;
    }
    grid.high
}

/// The share of `DRAWS` boxes drawn from the edge distributions (through each one's CDF, at
/// the points of a four-dimensional Halton sequence) that overlap `answer` by `OVERLAP`.
fn confidence(edges: &[Output], answer: &Rectangle, grid: Grid) -> f64 {
    let values = grid.values();
    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .map(|output| {
            let probabilities = output.probabilities();
            let total: f64 = probabilities.iter().sum();
            let mut mass = 0.0;
            let cdf = probabilities.iter().map(|probability| {
                mass += probability / total;
                mass
            });
            cdf.collect()
        })
        .collect();
    let hits = (1..=DRAWS).filter(|&draw| {
        let edge = |index: usize| {
            let point = radical_inverse(draw, [2, 3, 5, 7][index]);
            let bin = cdfs[index].partition_point(|&mass| mass < point);
            values[bin.min(grid.size - 1)]
        };
        let drawn = Rectangle::new([0, 1, 2, 3].map(edge), grid);
        drawn.overlap(answer) >= OVERLAP
    });
    hits.count() as f64 / DRAWS as f64
}

/// The `index`th point of the van der Corput sequence in `base`.
fn radical_inverse(mut index: usize, base: usize) -> f64 {
    let (mut value, mut scale) = (0.0, 1.0);
    while index > 0 {
        scale /= base as f64;
        value += (index % base) as f64 * scale;
        index /= base;
    }
    value
}

/// A box: absent below one half of its calibrated presence, else its edges and confidence.
fn one_box(outputs: &[Output], calibration: &Calibration, boxing: Boxing) -> Value {
    let found = calibration.through("found:box", outputs[0].probabilities()[0]);
    if found < PRESENT {
        return json!({ "type": "box", "box": null, "found": found, "confidence": 0.0 });
    }
    let edges = &outputs[1..BOX_SLOTS];
    let drawn = medians(edges, boxing);
    let confidence = calibration.through("box", confidence(edges, &drawn, boxing.grid));
    json!({ "type": "box", "box": drawn.json(), "found": found, "confidence": confidence })
}

/// Every instance: block `k` present when there are at least `k` (its presence kept
/// decreasing), the present boxes in reading order (left edge, then top); the count is at least
/// theirs when every block read is present.
fn every_box(outputs: &[Output], calibration: &Calibration, boxing: Boxing) -> Value {
    let mut below = 1.0_f64;
    let blocks = outputs.as_chunks::<BOX_SLOTS>().0;
    let presence: Vec<f64> = blocks
        .iter()
        .map(|block| {
            below = below.min(calibration.through("found:boxes", block[0].probabilities()[0]));
            below
        })
        .collect();
    let present = presence.iter().take_while(|&&found| found >= PRESENT);
    let mut boxes: Vec<(Rectangle, f64, f64)> = present
        .zip(blocks)
        .map(|(&found, block)| {
            let drawn = medians(&block[1..], boxing);
            let confidence = confidence(&block[1..], &drawn, boxing.grid);
            (drawn, found, calibration.through("boxes", confidence))
        })
        .collect();
    boxes.sort_by(|(first, ..), (second, ..)| {
        (first.left, first.top)
            .partial_cmp(&(second.left, second.top))
            .expect("edges are numbers")
    });
    let count = boxes.len();
    let listed: Vec<Value> = boxes
        .iter()
        .map(|(drawn, found, confidence)| {
            json!({ "box": drawn.json(), "found": found, "confidence": confidence })
        })
        .collect();
    json!({
        "type": "boxes",
        "boxes": listed,
        "count": count,
        "at_least": count > 0 && count == presence.len(),
        "presence": presence,
    })
}
