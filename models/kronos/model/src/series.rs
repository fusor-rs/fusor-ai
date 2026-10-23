//! A series as the reference predictor reads it: rows of a time and six values from a CSV, the
//! context's columns normalized and clipped, and a forecast's rows brought back.
//!
//! The CSV has a header naming a time column and `open`, `high`, `low` and `close`, and
//! optionally `volume` and `amount`. Like the reference, a series without volume has zero volume
//! and amount, and one without amount has `volume × mean(open, high, low, close)`.

use std::fmt;

use kernels::numpy::mean;

use crate::Error;
use crate::model::{FEATURES, Stamp};

/// The columns of a row, in order.
pub const COLUMNS: [&str; FEATURES] = ["open", "high", "low", "close", "volume", "amount"];
const PRICES: usize = 4;
/// The names a time column goes by.
const TIME_COLUMNS: [&str; 5] = ["timestamps", "timestamp", "date", "datetime", "time"];
/// Normalized values are clipped to `±CLIP`.
const CLIP: f32 = 5.0;
/// Added to each column's deviation before dividing by it.
const DEVIATION_FLOOR: f32 = 1e-5;
const SECONDS_PER_DAY: i64 = 86_400;
/// Opens a CSV that spreadsheet programs save as UTF-8.
const BYTE_ORDER_MARK: char = '\u{feff}';

/// A calendar time, to the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time {
    year: i64,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
}

impl Time {
    /// `YYYY-MM-DD`, optionally followed by ` HH:MM` or `THH:MM`, and optionally `:SS`.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let invalid = || Error::Request(format!("{text:?} is not a time like 2024-06-18 11:15:00"));
        let (date, clock) = text
            .trim()
            .split_once([' ', 'T'])
            .unwrap_or((text.trim(), ""));
        let number = |part: Option<&str>| part.and_then(|value| value.parse::<i64>().ok());
        let mut date = date.split(['-', '/']);
        let mut clock = clock.split(':');
        let fields = [date.next(), date.next(), date.next()].map(number);
        let [Some(year), Some(month @ 1..=12), Some(day @ 1..=31)] = fields else {
            return Err(invalid());
        };
        let mut part = |limit: i64| match clock.next() {
            None | Some("") => Some(0),
            value => number(value).filter(|value| (0..limit).contains(value)),
        };
        let (Some(hour), Some(minute), Some(second)) = (part(24), part(60), part(60)) else {
            return Err(invalid());
        };
        let narrow = |value: i64| u8::try_from(value).map_err(|_| invalid());
        Ok(Self {
            year,
            month: narrow(month)?,
            day: narrow(day)?,
            hour: narrow(hour)?,
            minute: narrow(minute)?,
            second: narrow(second)?,
        })
    }

    /// Days since 1970-01-01 (Howard Hinnant's `days_from_civil`).
    fn days(self) -> i64 {
        let year = self.year - i64::from(self.month <= 2);
        let era = year.div_euclid(400);
        let year_of_era = year - era * 400;
        let month = (i64::from(self.month) + 9) % 12;
        let day_of_year = (153 * month + 2) / 5 + i64::from(self.day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        era * 146_097 + day_of_era - 719_468
    }

    /// Seconds since 1970-01-01 00:00:00.
    fn seconds(self) -> i64 {
        let clock = i64::from(self.hour) * 3600 + i64::from(self.minute) * 60;
        self.days() * SECONDS_PER_DAY + clock + i64::from(self.second)
    }

    /// The time `seconds` after 1970-01-01 00:00:00 (Howard Hinnant's `civil_from_days`).
    fn from_seconds(seconds: i64) -> Self {
        let (days, clock) = (
            seconds.div_euclid(SECONDS_PER_DAY),
            seconds.rem_euclid(SECONDS_PER_DAY),
        );
        let shifted = days + 719_468;
        let era = shifted.div_euclid(146_097);
        let day_of_era = shifted - era * 146_097;
        let year_of_era =
            (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let month_index = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * month_index + 2) / 5 + 1;
        let month = if month_index < 10 {
            month_index + 3
        } else {
            month_index - 9
        };
        let narrow = |value: i64| u8::try_from(value).expect("a calendar field fits a byte");
        Self {
            year: year_of_era + era * 400 + i64::from(month <= 2),
            month: narrow(month),
            day: narrow(day),
            hour: narrow(clock / 3600),
            minute: narrow(clock / 60 % 60),
            second: narrow(clock % 60),
        }
    }

    /// The time's stamp: its minute, hour, weekday (Monday 0), day and month.
    pub fn stamp(self) -> Stamp {
        const THURSDAY: i64 = 3;
        let weekday = (self.days() + THURSDAY).rem_euclid(7);
        let weekday = u8::try_from(weekday).expect("a weekday is below 7");
        [self.minute, self.hour, weekday, self.day, self.month]
    }
}

impl fmt::Display for Time {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        } = self;
        write!(
            formatter,
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
        )
    }
}

/// A series' rows: their times and values.
#[derive(Debug, Clone, Default)]
pub struct Series {
    pub times: Vec<Time>,
    pub rows: Vec<[f64; FEATURES]>,
}

impl Series {
    pub fn parse(csv: &str) -> Result<Self, Error> {
        let csv = csv.strip_prefix(BYTE_ORDER_MARK).unwrap_or(csv);
        let mut lines = csv.lines().filter(|line| !line.trim().is_empty());
        let header: Vec<String> = lines
            .next()
            .ok_or_else(|| Error::Request("the CSV is empty".into()))?
            .split(',')
            .map(|name| name.trim().to_lowercase())
            .collect();
        let column = |name: &str| header.iter().position(|heading| heading == name);
        let time = TIME_COLUMNS
            .iter()
            .find_map(|name| column(name))
            .ok_or_else(|| {
                Error::Request(format!(
                    "the CSV has no time column: one of {TIME_COLUMNS:?}"
                ))
            })?;
        let columns = COLUMNS.map(column);
        if let Some(missing) = COLUMNS[..PRICES]
            .iter()
            .zip(&columns)
            .find(|(_, at)| at.is_none())
        {
            return Err(Error::Request(format!(
                "the CSV has no {} column",
                missing.0
            )));
        }
        let mut series = Self::default();
        for (line, text) in (1..).zip(lines) {
            let cells: Vec<&str> = text.split(',').map(str::trim).collect();
            if cells.len() < header.len() {
                return Err(Error::Request(format!(
                    "row {line} has {} cells, the header {}",
                    cells.len(),
                    header.len()
                )));
            }
            series.times.push(Time::parse(cells[time])?);
            let mut row = [0.0; FEATURES];
            for (value, at) in row.iter_mut().zip(columns) {
                let Some(at) = at else { continue };
                *value = cells[at].parse().map_err(|_| {
                    Error::Request(format!("row {line}: {:?} is not a number", cells[at]))
                })?;
            }
            let [.., volume, amount] = columns;
            if volume.is_none() || amount.is_none() {
                row[FEATURES - 1] = row[PRICES] * (row[..PRICES].iter().sum::<f64>() / 4.0);
            }
            series.rows.push(row);
        }
        Ok(series)
    }

    /// The `count` times after row `end`: the series' own, then steps of its last interval.
    pub fn times_after(&self, end: usize, count: usize) -> Vec<Time> {
        let known = self.times[end..].iter().take(count).copied();
        let mut times: Vec<Time> = known.collect();
        let interval = self.times[end - 1].seconds() - self.times[end.max(2) - 2].seconds();
        let last = *times.last().unwrap_or(&self.times[end - 1]);
        let mut step = 0;
        while times.len() < count {
            step += 1;
            times.push(Time::from_seconds(last.seconds() + step * interval));
        }
        times
    }
}

/// A context's rows normalized column by column, and each column's mean and deviation.
#[derive(Debug, Clone)]
pub struct Normalized {
    /// `rows × FEATURES`, clipped to `±CLIP`.
    pub values: Vec<f32>,
    pub mean: [f32; FEATURES],
    pub deviation: [f32; FEATURES],
}

impl Normalized {
    /// As NumPy computes the reference's `np.mean` and `np.std` of an f32 context along its
    /// rows: each column's values, then its squared deviations, summed pairwise in f32.
    pub fn new(rows: &[[f64; FEATURES]]) -> Self {
        let values: Vec<[f32; FEATURES]> = rows
            .iter()
            .map(|row| row.map(|value| value as f32))
            .collect();
        let mean = column_means(&values);
        let squares: Vec<[f32; FEATURES]> = values
            .iter()
            .map(|row| {
                std::array::from_fn(|column| {
                    let difference = row[column] - mean[column];
                    difference * difference
                })
            })
            .collect();
        let deviation = column_means(&squares).map(f32::sqrt);
        let normalized = values.iter().flat_map(|row| {
            std::array::from_fn::<f32, FEATURES, _>(|column| {
                let scaled = (row[column] - mean[column]) / (deviation[column] + DEVIATION_FLOOR);
                scaled.clamp(-CLIP, CLIP)
            })
        });
        Self {
            values: normalized.collect(),
            mean,
            deviation,
        }
    }

    /// Normalized rows, `rows × FEATURES`, in the series' units.
    pub fn restore(&self, values: &[f32]) -> Vec<[f32; FEATURES]> {
        let rows = values.as_chunks::<FEATURES>().0;
        let restore = |row: &[f32; FEATURES]| {
            std::array::from_fn(|column| {
                row[column] * (self.deviation[column] + DEVIATION_FLOOR) + self.mean[column]
            })
        };
        rows.iter().map(restore).collect()
    }
}

fn column_means(rows: &[[f32; FEATURES]]) -> [f32; FEATURES] {
    std::array::from_fn(|column| {
        let values: Vec<f32> = rows.iter().map(|row| row[column]).collect();
        mean(&values)
    })
}
