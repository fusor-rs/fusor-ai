//! Forecasting a series: `start` parses the request, normalizes its context and tokenizes it;
//! each `step` then draws one token for every sample, the first reading the context; the last
//! step decodes each sample's window, averages them and brings the forecast back to the
//! series' units. The stages also run one by one, on given inputs, for checks against the
//! reference.

use std::ops::Range;

use bundle::{Error, Store};
use kernels::rng::Rng;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::model::{FEATURES, MAX_CONTEXT, Model, Stamp};
use crate::predictor::{self, Stream, Window};
use crate::sampling::Sampling;
use crate::series::{Normalized, Series, Time};
use crate::tokenizer::{self, Token};

/// The most samples a forecast averages.
pub const MAX_SAMPLES: usize = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub csv: String,
    /// Rows of context before the forecast.
    pub lookback: usize,
    /// Rows to forecast, at most `MAX_CONTEXT`.
    pub horizon: usize,
    /// Rows of the series before the forecast: all of them unless given.
    #[serde(default)]
    pub end: Option<usize>,
    #[serde(default)]
    pub sampling: Sampling,
    /// Forecasts drawn and averaged.
    #[serde(default = "one")]
    pub samples: usize,
    #[serde(default)]
    pub seed: u64,
}

fn one() -> usize {
    1
}

impl Request {
    /// The context's rows of `series`, the `lookback` before the forecast, once the request is
    /// checked against it.
    pub fn context(&self, series: &Series) -> Result<Range<usize>, Error> {
        let end = self.end.unwrap_or(series.rows.len());
        check(self, end, series.rows.len())?;
        Ok(end - self.lookback..end)
    }
}

/// A series' rows as the page shows them: their times and values.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Rows {
    pub times: Vec<String>,
    pub values: Vec<[f32; FEATURES]>,
}

/// What a forecast reads and draws: the context, the series' own rows within the horizon (a
/// backtest's answer), and the forecast's times.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub context: Rows,
    pub actual: Rows,
    pub times: Vec<String>,
    pub samples: usize,
    /// Steps that read their whole window again, the sequence being past `MAX_CONTEXT` rows.
    pub slides: usize,
}

/// One step done: the tokens drawn so far, and the forecast after the last.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub step: usize,
    pub steps: usize,
    pub forecast: Option<Rows>,
}

pub struct Engine {
    model: Model,
    team: Team,
    tokenizer: tokenizer::Buffers,
    buffers: predictor::Buffers,
    streams: Vec<Stream>,
    job: Option<Job>,
}

struct Job {
    normalized: Normalized,
    /// Every row's stamp: the context's, then the forecast's.
    stamps: Vec<Stamp>,
    times: Vec<String>,
    /// Each sample's tokens: the context's, then those drawn.
    sequences: Vec<Vec<Token>>,
    context: usize,
    horizon: usize,
    sampling: Sampling,
    rng: Rng,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let model = Model::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        Ok(Self {
            model,
            tokenizer: tokenizer::Buffers::new(threads),
            buffers: predictor::Buffers::new(threads),
            team,
            streams: Vec::new(),
            job: None,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {
        self.job = None;
    }
}

impl Engine {
    /// Each row's code before quantization, `rows × CODE`, from normalized rows.
    pub fn codes(&mut self, rows: &[f32]) -> Vec<f32> {
        let tokenizer = &self.model.tokenizer;
        tokenizer.codes(&self.team, &mut self.tokenizer, rows)
    }

    /// The rows of `tokens`, normalized, `tokens × FEATURES`.
    pub fn decode(&mut self, tokens: &[Token]) -> Vec<f32> {
        let tokenizer = &self.model.tokenizer;
        tokenizer.decode(&self.team, &mut self.tokenizer, tokens)
    }

    /// The `s1` logits after reading `window` from its first row, then the `s2` logits for
    /// `s1`.
    pub fn logits(&mut self, window: Window, s1: u16) -> [Vec<f32>; 2] {
        let mut stream = Stream::default();
        let model = &self.model;
        model.read(&self.team, &mut self.buffers, &mut stream, window);
        let coarse = self.buffers.logits.read().to_vec();
        model.dependent(&self.team, &self.buffers, &stream, s1);
        [coarse, self.buffers.logits.read().to_vec()]
    }

    /// Starts forecasting `request`.
    pub fn start(&mut self, request: &Request) -> Result<Plan, Error> {
        self.job = None;
        let series = Series::parse(&request.csv)?;
        let context = request.context(&series)?;
        let end = context.end;
        let normalized = Normalized::new(&series.rows[context.clone()]);
        let tokens = tokenizer::tokens(&self.codes(&normalized.values));
        let future = series.times_after(end, request.horizon);
        let stamps = series.times[context.clone()]
            .iter()
            .chain(&future)
            .map(|time| time.stamp())
            .collect();
        let times: Vec<String> = future.iter().map(Time::to_string).collect();
        let actual = end..series.rows.len().min(end + request.horizon);
        self.streams.resize_with(request.samples, Stream::default);
        let plan = Plan {
            context: rows(&series, context),
            actual: rows(&series, actual),
            times: times.clone(),
            samples: request.samples,
            slides: (request.lookback + request.horizon).saturating_sub(MAX_CONTEXT + 1),
        };
        self.job = Some(Job {
            normalized,
            stamps,
            times,
            sequences: vec![tokens; request.samples],
            context: request.lookback,
            horizon: request.horizon,
            sampling: request.sampling,
            rng: Rng::new(request.seed),
        });
        Ok(plan)
    }

    /// Draws the next token of every sample; the last step also decodes the forecast.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no forecast in progress".into()))?;
        for (stream, tokens) in self.streams.iter_mut().zip(&mut job.sequences) {
            let start = tokens.len().saturating_sub(MAX_CONTEXT);
            let window = Window {
                tokens: &tokens[start..],
                stamps: &job.stamps[start..tokens.len()],
            };
            if start > 0 || tokens.len() == job.context {
                stream.restart();
            }
            let (model, team) = (&self.model, &self.team);
            model.read(team, &mut self.buffers, stream, window);
            let s1 = job
                .sampling
                .sample(self.buffers.logits.read(), &mut job.rng);
            let s1 = u16::try_from(s1).expect("a token half is below VOCABULARY");
            model.dependent(team, &self.buffers, stream, s1);
            let s2 = job
                .sampling
                .sample(self.buffers.logits.read(), &mut job.rng);
            let s2 = u16::try_from(s2).expect("a token half is below VOCABULARY");
            tokens.push(Token { s1, s2 });
        }
        let step = job.sequences[0].len() - job.context;
        let stage = if step == 1 { "read" } else { "sample" };
        if step < job.horizon {
            let progress = progress(stage, step, &job, None);
            self.job = Some(job);
            return Ok(progress);
        }
        let forecast = self.forecast(&job);
        Ok(progress("decode", step, &job, Some(forecast)))
    }

    /// Each sample's last window decoded, the forecast's rows averaged over the samples and
    /// brought back to the series' units.
    fn forecast(&mut self, job: &Job) -> Rows {
        let mut sum = vec![0.0f32; job.horizon * FEATURES];
        for tokens in &job.sequences {
            let window = &tokens[tokens.len().saturating_sub(MAX_CONTEXT)..];
            let decoded = self.decode(window);
            let forecast = &decoded[decoded.len() - sum.len()..];
            for (total, value) in sum.iter_mut().zip(forecast) {
                *total += value;
            }
        }
        let samples = job.sequences.len() as f32;
        let mean: Vec<f32> = sum.iter().map(|total| total / samples).collect();
        Rows {
            times: job.times.clone(),
            values: job.normalized.restore(&mean),
        }
    }
}

fn check(request: &Request, end: usize, rows: usize) -> Result<(), Error> {
    let Request {
        lookback,
        horizon,
        samples,
        sampling,
        ..
    } = *request;
    let problem = if !(2..=MAX_CONTEXT).contains(&lookback) {
        format!("a lookback of {lookback} rows: from 2 to {MAX_CONTEXT}")
    } else if !(lookback..=rows).contains(&end) {
        format!(
            "{end} rows before the forecast: from the lookback, {lookback}, to the series' {rows}"
        )
    } else if !(1..=MAX_CONTEXT).contains(&horizon) {
        format!("a horizon of {horizon} rows: from 1 to {MAX_CONTEXT}")
    } else if !(1..=MAX_SAMPLES).contains(&samples) {
        format!("{samples} samples: from 1 to {MAX_SAMPLES}")
    } else if sampling.temperature <= 0.0 || !(sampling.top_p > 0.0 && sampling.top_p <= 1.0) {
        "a temperature above 0 and a top-p in (0, 1]".into()
    } else {
        return Ok(());
    };
    Err(Error::Request(problem))
}

fn rows(series: &Series, range: Range<usize>) -> Rows {
    Rows {
        times: series.times[range.clone()]
            .iter()
            .map(Time::to_string)
            .collect(),
        values: series.rows[range]
            .iter()
            .map(|row| row.map(|value| value as f32))
            .collect(),
    }
}

fn progress(stage: &'static str, step: usize, job: &Job, forecast: Option<Rows>) -> Progress {
    Progress {
        stage,
        step,
        steps: job.horizon,
        forecast,
    }
}
