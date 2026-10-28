//! A release's `calibration.json`, as d1's reference applies it (`calibration.py`): a
//! temperature for the option questions' scores, per kind and size of question or the default,
//! and a table for each typed answer's probabilities, a monotone map from raw to calibrated
//! knots, linear between them and flat past the ends (`found:text`, `found:box`, `box`,
//! `multiple`, `estimate`, …).

use std::collections::HashMap;

use serde::Deserialize;

use crate::Error;
use crate::request::Kind;

#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    default: f64,
    temperatures: HashMap<String, f64>,
    #[serde(default)]
    tables: HashMap<String, Vec<[f64; 2]>>,
}

impl Calibration {
    /// # Errors
    /// When `json` is not a calibration.
    pub fn read(json: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(json)
            .map_err(|invalid| Error::Format(format!("calibration.json: {invalid}")))
    }

    /// The temperature of an option question's scores: its kind's and size's, else the
    /// default.
    pub fn temperature(&self, kind: &Kind) -> f64 {
        let bucket = match kind {
            Kind::Noul(_) => "noul".to_owned(),
            Kind::Score(levels) => format!("score:{}", levels.len()),
            Kind::Choice { options, .. } => match options.len() {
                0..=3 => "choice:small",
                4..=8 => "choice:mid",
                9..=26 => "choice:large",
                _ => "choice:inventory",
            }
            .to_owned(),
            _ => return 1.0,
        };
        self.temperatures
            .get(&bucket)
            .copied()
            .unwrap_or(self.default)
    }

    /// `value` through table `name`, as it is without one.
    pub fn through(&self, name: &str, value: f64) -> f64 {
        self.tables
            .get(name)
            .map_or(value, |knots| through(value, knots.iter().copied()))
    }

    /// The raw level whose calibrated share is `share`: table `name` read backwards.
    pub fn level(&self, name: &str, share: f64) -> f64 {
        let knots = |knots: &Vec<[f64; 2]>| {
            through(
                share,
                knots.iter().map(|&[raw, calibrated]| [calibrated, raw]),
            )
        };
        self.tables.get(name).map_or(share, knots)
    }
}

/// `value` through increasing `(raw, calibrated)` knots.
fn through(value: f64, knots: impl Iterator<Item = [f64; 2]>) -> f64 {
    let knots: Vec<[f64; 2]> = knots.collect();
    let index = knots.partition_point(|&[raw, _]| raw < value);
    match (
        index.checked_sub(1).map(|before| knots[before]),
        knots.get(index),
    ) {
        (Some([low, at_low]), Some([high, at_high])) => {
            at_low + (value - low) * (at_high - at_low) / (high - low)
        }
        (None, Some(&[_, first])) => first,
        (Some([_, last]), None) => last,
        (None, None) => value,
    }
}
