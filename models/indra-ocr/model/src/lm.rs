//! The token n-gram language models that break close calls on lines a recognizer is unsure of,
//! and the CTC prefix beam search that consults them: `score = log P_ctc + alpha · log P_lm +
//! beta · tokens`.
//!
//! A model is a backoff table: per history length `n`, the (history, token) keys seen in
//! training with their log-probabilities, and the history keys with their backoff weights. A
//! key packs tokens `bits` bits each (0 the line start, `t + 1` token `t`), the history length
//! in its top bits, so that one sorted table holds every length. The compiler builds the
//! `Tables` from the checkpoint's `.npz` files; the bundle stores their bytes.

use std::collections::HashMap;

use crate::Error;

/// Readings kept per step.
const BEAM: usize = 8;
/// A step whose blank is likelier than this keeps every reading as it is.
const BLANK_SKIP: f64 = 0.999;
/// A class less likely than this at a step is not tried there.
const CANDIDATE_FLOOR: f64 = 3e-3;
/// Slack for the rounding of the score bound a new reading must reach.
const SLACK: f64 = 1e-9;

/// Bits per token, unless the checkpoint says more (a larger vocabulary).
pub const DEFAULT_BITS: usize = 11;
/// The history length goes in a key's top bits, from this one.
const TAG_SHIFT: u32 = 61;
const WORD: usize = 4;
const KEY: usize = 8;

/// One history length's arrays, as the checkpoint stores them.
pub struct Level {
    pub keys: Vec<u64>,
    pub scores: Vec<f32>,
    pub histories: Vec<u64>,
    pub backoffs: Vec<f32>,
}

/// A language model: every history length's keys in two sorted tables, the length in each
/// key's top bits.
pub struct Ngram {
    order: usize,
    vocabulary: usize,
    bits: usize,
    probabilities: Table,
    backoffs: Table,
}

#[derive(Default)]
struct Table {
    keys: Vec<u64>,
    values: Vec<f32>,
}

impl Table {
    /// The levels' `(keys, values)` merged, each key tagged with its level, sorted by key.
    fn merged<'level>(levels: impl Iterator<Item = (&'level [u64], &'level [f32])>) -> Self {
        let mut pairs: Vec<(u64, f32)> = Vec::new();
        for (level, (keys, values)) in levels.enumerate() {
            let tag = (level as u64) << TAG_SHIFT;
            pairs.extend(keys.iter().map(|key| key | tag).zip(values.iter().copied()));
        }
        pairs.sort_by_key(|&(key, _)| key);
        let (keys, values) = pairs.into_iter().unzip();
        Self { keys, values }
    }

    fn find(&self, key: u64) -> Option<f64> {
        let index = self.keys.binary_search(&key).ok()?;
        Some(f64::from(self.values[index]))
    }
}

impl Ngram {
    pub fn new(vocabulary: usize, bits: usize, levels: &[Level]) -> Result<Self, Error> {
        if levels.len() * bits > TAG_SHIFT as usize {
            return Err(Error::Format(format!(
                "a language model of order {} at {bits} bits a token: keys overflow",
                levels.len()
            )));
        }
        let probabilities = Table::merged(
            levels
                .iter()
                .map(|level| (&level.keys[..], &level.scores[..])),
        );
        let backoffs = Table::merged(
            levels
                .iter()
                .map(|level| (&level.histories[..], &level.backoffs[..])),
        );
        Ok(Self {
            order: levels.len(),
            vocabulary,
            bits,
            probabilities,
            backoffs,
        })
    }

    /// Little-endian: order, vocabulary and bits (u32), then each table's length (u32), keys
    /// (u64) and values (f32), probabilities first.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for value in [self.order, self.vocabulary, self.bits] {
            out.extend_from_slice(&(value as u32).to_le_bytes());
        }
        for table in [&self.probabilities, &self.backoffs] {
            out.extend_from_slice(&(table.keys.len() as u32).to_le_bytes());
            out.extend(table.keys.iter().flat_map(|key| key.to_le_bytes()));
            out.extend(table.values.iter().flat_map(|value| value.to_le_bytes()));
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut reader = Reader { bytes };
        let order = reader.word()? as usize;
        let vocabulary = reader.word()? as usize;
        let bits = reader.word()? as usize;
        let probabilities = reader.table()?;
        let backoffs = reader.table()?;
        if !reader.bytes.is_empty() {
            return Err(corrupt());
        }
        Ok(Self {
            order,
            vocabulary,
            bits,
            probabilities,
            backoffs,
        })
    }

    /// An upper bound of every `log_probability`: the largest log-probability plus every
    /// positive backoff weight a back-off can add.
    fn ceiling(&self) -> f64 {
        let largest = self
            .probabilities
            .values
            .iter()
            .copied()
            .fold(f32::MIN, f32::max);
        let top = f64::from(largest).max(-(self.vocabulary as f64).ln());
        let mut level_max = vec![0.0f32; self.order];
        for (key, &value) in self.backoffs.keys.iter().zip(&self.backoffs.values) {
            let level = (key >> TAG_SHIFT) as usize;
            level_max[level] = level_max[level].max(value);
        }
        top + level_max.iter().map(|&max| f64::from(max)).sum::<f64>()
    }

    /// `log P(token | history)` of a key packing the last `order - 1` history tokens and then
    /// the token (`logp_keys`): from the longest seen history that saw the token, no longer than
    /// the longest seen history, plus the backoff weights of the seen histories above it; with
    /// none, every backoff weight and a uniform floor.
    fn log_probability(&self, key: u64) -> f64 {
        let bits = self.bits as u32;
        let token = key & ((1 << bits) - 1);
        let history = |level: usize| {
            let tag = (level as u64) << TAG_SHIFT;
            ((key >> bits) & ((1 << (bits * level as u32)) - 1), tag)
        };
        let backoff = |level: usize| {
            let (history, tag) = history(level);
            self.backoffs.find(history | tag)
        };
        let longest = (1..self.order)
            .take_while(|&level| backoff(level).is_some())
            .count();
        let mut weight = 0.0;
        for level in (0..=longest).rev() {
            let (history, tag) = history(level);
            if let Some(value) = self.probabilities.find((history << bits) | token | tag) {
                return weight + value;
            }
            // An unseen history adds no weight.
            weight += backoff(level).unwrap_or(0.0);
        }
        weight - (self.vocabulary as f64).ln()
    }
}

/// Reads the tables' bytes in order.
struct Reader<'bytes> {
    bytes: &'bytes [u8],
}

impl<'bytes> Reader<'bytes> {
    fn take(&mut self, len: usize) -> Result<&'bytes [u8], Error> {
        let (head, rest) = self.bytes.split_at_checked(len).ok_or_else(corrupt)?;
        self.bytes = rest;
        Ok(head)
    }

    fn word(&mut self) -> Result<u32, Error> {
        let (word, rest) = self.bytes.split_first_chunk::<WORD>().ok_or_else(corrupt)?;
        self.bytes = rest;
        Ok(u32::from_le_bytes(*word))
    }

    fn table(&mut self) -> Result<Table, Error> {
        let len = self.word()? as usize;
        let keys = self.take(len * KEY)?.as_chunks::<KEY>().0;
        let values = self.take(len * WORD)?.as_chunks::<WORD>().0;
        Ok(Table {
            keys: keys.iter().map(|key| u64::from_le_bytes(*key)).collect(),
            values: values
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect(),
        })
    }
}

fn corrupt() -> Error {
    Error::Format("a language model's tables are corrupt: recompile the bundle".into())
}

/// How much the language model weighs (`alpha`) and what each token earns (`beta`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub alpha: f64,
    pub beta: f64,
}

/// A line's recognizer outputs for the search: per step, the blank's log-probability and the
/// best classes (the blank among them) with theirs.
pub(crate) struct Steps<'line> {
    pub(crate) blank: &'line [f32],
    pub(crate) classes: &'line [u32],
    pub(crate) log_probabilities: &'line [f32],
    pub(crate) best: usize,
}

/// A reading's state: log-probabilities of ending in a blank and in its last token, its
/// language-model score, the step each token was first emitted at, and its last `order - 1`
/// tokens packed.
#[derive(Clone)]
struct Reading {
    blank: f64,
    token: f64,
    score: f64,
    steps: Vec<usize>,
    history: u64,
}

impl Reading {
    fn total(&self) -> f64 {
        add_logs(self.blank, self.token)
    }

    fn ranked(&self) -> f64 {
        self.total() + self.score
    }
}

/// The readings of one step, in the order they arrived (a dictionary's).
#[derive(Default)]
struct Readings {
    entries: Vec<(Vec<u32>, Reading)>,
    index: HashMap<Vec<u32>, usize>,
}

impl Readings {
    /// Adds `reading` under `prefix`, or merges its probabilities into the one there (whose
    /// score, steps and history stay).
    fn add(&mut self, prefix: Vec<u32>, reading: Reading) {
        if let Some(&at) = self.index.get(&prefix) {
            let entry = &mut self.entries[at].1;
            entry.blank = add_logs(entry.blank, reading.blank);
            entry.token = add_logs(entry.token, reading.token);
            return;
        }
        self.index.insert(prefix.clone(), self.entries.len());
        self.entries.push((prefix, reading));
    }

    /// The `BEAM` best, best first, ties in arrival order.
    fn best(mut self) -> Vec<(Vec<u32>, Reading)> {
        self.entries
            .sort_by(|left, right| right.1.ranked().total_cmp(&left.1.ranked()));
        self.entries.truncate(BEAM);
        self.entries
    }
}

/// One new token on a reading, to be scored by the language model.
struct Extension {
    prefix: Vec<u32>,
    class: u32,
    token: f64,
    from: usize,
}

/// The tokens of the best reading of a line and the step each was first emitted at.
pub(crate) fn beam_search(steps: &Steps, lm: &Ngram, weights: Weights) -> (Vec<u32>, Vec<usize>) {
    let mut beams = vec![(
        Vec::new(),
        Reading {
            blank: 0.0,
            token: f64::NEG_INFINITY,
            score: 0.0,
            steps: Vec::new(),
            history: 0,
        },
    )];
    let reach = weights.beta + weights.alpha * lm.ceiling();
    for step in 0..steps.blank.len() {
        let blank = f64::from(steps.blank[step]);
        if blank > BLANK_SKIP.ln() {
            for (_, reading) in &mut beams {
                *reading = Reading {
                    blank: reading.total() + blank,
                    token: f64::NEG_INFINITY,
                    ..reading.clone()
                };
            }
            continue;
        }
        let candidates = candidates(steps, step);
        let next = carry(&beams, blank, &candidates);
        beams = extend(&beams, next, &candidates, (lm, weights, reach), step).best();
    }
    let best = beams
        .into_iter()
        .reduce(|best, entry| {
            if entry.1.ranked() > best.1.ranked() {
                entry
            } else {
                best
            }
        })
        .expect("the search keeps at least one reading");
    (best.0, best.1.steps)
}

/// The step's classes other than the blank that are likely enough to try.
fn candidates(steps: &Steps, step: usize) -> Vec<(u32, f64)> {
    let row = step * steps.best..(step + 1) * steps.best;
    steps.classes[row.clone()]
        .iter()
        .zip(&steps.log_probabilities[row])
        .filter(|&(&class, &value)| class != 0 && f64::from(value) >= CANDIDATE_FLOOR.ln())
        .map(|(&class, &value)| (class, f64::from(value)))
        .collect()
}

/// Every reading carried through the step unchanged: by the blank, and by a repeat of its last
/// token that collapses into it.
fn carry(beams: &[(Vec<u32>, Reading)], blank: f64, candidates: &[(u32, f64)]) -> Readings {
    let mut next = Readings::default();
    for (prefix, reading) in beams {
        let carried = Reading {
            blank: reading.total() + blank,
            token: f64::NEG_INFINITY,
            ..reading.clone()
        };
        next.add(prefix.clone(), carried);
        for &(class, value) in candidates {
            if prefix.last() == Some(&class) {
                let repeated = Reading {
                    blank: f64::NEG_INFINITY,
                    token: reading.token + value,
                    ..reading.clone()
                };
                next.add(prefix.clone(), repeated);
            }
        }
    }
    next
}

/// `next` with every reading extended by one more token: into a reading already there, or,
/// when it can still reach the beam's floor, as a new one scored by the language model.
fn extend(
    beams: &[(Vec<u32>, Reading)],
    mut next: Readings,
    candidates: &[(u32, f64)],
    (lm, weights, reach): (&Ngram, Weights, f64),
    step: usize,
) -> Readings {
    let floor = if beams.len() >= BEAM {
        next.entries
            .iter()
            .map(|(_, reading)| reading.ranked())
            .fold(f64::INFINITY, f64::min)
    } else {
        f64::NEG_INFINITY
    };
    let mut extensions = Vec::new();
    for (from, (prefix, reading)) in beams.iter().enumerate() {
        for &(class, value) in candidates {
            let start = if prefix.last() == Some(&class) {
                reading.blank
            } else {
                reading.total()
            };
            let token = start + value;
            if token == f64::NEG_INFINITY {
                continue;
            }
            let mut longer = prefix.clone();
            longer.push(class);
            if next.index.contains_key(&longer) {
                next.add(
                    longer,
                    Reading {
                        blank: f64::NEG_INFINITY,
                        token,
                        ..reading.clone()
                    },
                );
            } else if token + reading.score + reach >= floor - SLACK {
                extensions.push(Extension {
                    prefix: longer,
                    class,
                    token,
                    from,
                });
            }
        }
    }
    let bits = lm.bits as u32;
    let mask = (1u64 << (bits * (lm.order as u32 - 1))) - 1;
    for extension in extensions {
        let reading = &beams[extension.from].1;
        let key = (reading.history << bits) | (u64::from(extension.class) + 1);
        let mut steps = reading.steps.clone();
        steps.push(step);
        next.add(
            extension.prefix,
            Reading {
                blank: f64::NEG_INFINITY,
                token: extension.token,
                score: reading.score + weights.alpha * lm.log_probability(key) + weights.beta,
                steps,
                history: key & mask,
            },
        );
    }
    next
}

/// `log(eᵃ + eᵇ)`.
fn add_logs(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let max = left.max(right);
    max + ((left - max).exp() + (right - max).exp()).ln()
}
