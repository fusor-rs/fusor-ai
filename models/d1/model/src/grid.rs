//! Ordered values read as a distribution over equally spaced bins: an estimate's, and a box
//! edge's.

/// `size` values from `low` to `high`, equally spaced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub low: f64,
    pub high: f64,
    pub size: usize,
}

/// A distribution's median and central 80 percent interval, on a grid's values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantiles {
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
}

impl Grid {
    pub fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }

    /// The values, the last exactly `high`.
    pub fn values(&self) -> Vec<f64> {
        let step = self.step();
        let inner = (0..self.size - 1).map(|index| self.low + index as f64 * step);
        inner.chain([self.high]).collect()
    }

    /// The first value at which the cumulative share of `probabilities` reaches `level`, the
    /// last when rounding leaves it short.
    pub fn quantile(&self, probabilities: &[f64], level: f64) -> f64 {
        let total: f64 = probabilities.iter().sum();
        let values = self.values();
        let mut mass = 0.0;
        for (value, probability) in values.iter().zip(probabilities) {
            mass += probability / total;
            if mass >= level {
                return *value;
            }
        }
        values[values.len() - 1]
    }

    /// The median and the 10th and 90th percentiles, each read at the level `level` gives it
    /// (a calibration's).
    pub fn quantiles(&self, probabilities: &[f64], level: impl Fn(f64) -> f64) -> Quantiles {
        let at = |share| self.quantile(probabilities, level(share));
        Quantiles {
            p10: at(0.1),
            median: at(0.5),
            p90: at(0.9),
        }
    }
}
