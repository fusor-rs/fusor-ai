//! A read's span, as d1's reference reads it (`pointer.py`): each of its two slots' final-normed
//! state against every state position's, scaled by the width's square root, the keys the
//! positions of the state's text and position 0, which reads as nothing. P(found) is one minus
//! the start's softmax at 0; a found span is the likeliest start and end within the read's cap.
//! Its text is the state's characters the tokens cover, trimmed of the space around them and,
//! in a JSON state, clipped to the one string they hold (`string_paths`), its escapes read.

use serde_json::Value;

use bundle::python;

use crate::Error;
use crate::prompt::{Prompter, state_block};
use crate::request::Written;

/// A read's longest span, in tokens.
pub fn cap(value: Written) -> usize {
    match value {
        Written::Text => 512,
        Written::Number | Written::Date => 32,
    }
}

/// The state's text, each prompt token's characters of it (none for the template's, the
/// media's and the questions'), and the spans of its JSON strings with their paths.
#[derive(Debug, Clone)]
pub struct StatePositions {
    text: Vec<char>,
    offsets: Vec<Option<(usize, usize)>>,
    paths: Vec<(usize, usize, String)>,
}

/// A read: its text and its span of the state's characters, and the path of the JSON string it
/// lies in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub span: [usize; 2],
    pub path: Option<String>,
}

impl StatePositions {
    /// The positions of the prompt `prompt` (its tokens' character `spans`), which opens with
    /// `prefix`, the state rendered in it.
    pub fn new(
        prompter: &Prompter,
        state: &Value,
        prefix: &str,
        prompt: &str,
        spans: &[(usize, usize)],
    ) -> Self {
        let prefix_chars = prefix.chars().count();
        let (text, boundaries, start) = match state {
            Value::Null => (Vec::new(), vec![0], prefix_chars),
            state => {
                let block = state_block(state);
                let (escaped, boundaries) = prompter.plain_boundaries(&block);
                let tail = escaped.len() + 1 + Prompter::header(state).chars().count();
                let mut text: Vec<char> = block.chars().collect();
                text.truncate(text.len() - 2);
                (text, boundaries, prefix_chars - tail)
            }
        };
        let full: Vec<char> = prompt.chars().collect();
        let stop = start + boundaries.partition_point(|&boundary| boundary <= text.len()) - 1;
        let blank = |chars: &[char]| python::strip(&chars.iter().collect::<String>()).is_empty();
        let offsets = spans.iter().map(|&(first, end)| {
            let (low, high) = (first.max(start), end.min(stop));
            if low >= high || !blank(&full[first..low]) || !blank(&full[high..end]) {
                return None;
            }
            let (left, right) = (boundaries[low - start], boundaries[high - start]);
            (left < right && right <= text.len() && !blank(&text[left..right]))
                .then_some((left, right))
        });
        let paths = match state {
            Value::Null | Value::String(_) => Vec::new(),
            _ => string_paths(&text),
        };
        Self {
            offsets: offsets.collect(),
            text,
            paths,
        }
    }

    /// The state's text, which a read's span indexes.
    pub fn text(&self) -> String {
        self.text.iter().collect()
    }

    /// The positions a read's pointer scores: position 0 and the state's text, up to the last.
    pub fn keys(&self) -> Vec<bool> {
        let valid = self
            .offsets
            .iter()
            .enumerate()
            .map(|(at, span)| at == 0 || span.is_some());
        let mut valid: Vec<bool> = valid.collect();
        let last = valid
            .iter()
            .rposition(|&key| key)
            .expect("position 0 is a key");
        valid.truncate(last + 1);
        valid
    }

    /// The text of tokens `first` to `last`.
    ///
    /// # Errors
    /// When either token is not of the state's text.
    pub fn read(&self, first: usize, last: usize) -> Result<Span, Error> {
        let (Some(Some(from)), Some(Some(to))) = (self.offsets.get(first), self.offsets.get(last))
        else {
            return Err(Error::Format(format!(
                "a read's span {first}..={last} leaves the state"
            )));
        };
        let (mut start, mut end) = (from.0, to.1);
        while start < end && self.text[start].is_whitespace() {
            start += 1;
        }
        while end > start && self.text[end - 1].is_whitespace() {
            end -= 1;
        }
        let mut held = self
            .paths
            .iter()
            .filter(|&&(from, to, _)| from < end && start < to);
        if let (Some((from, to, path)), None) = (held.next(), held.next()) {
            let (start, end) = (start.max(*from), end.min(*to));
            let escaped: String = self.text[start..end].iter().collect();
            return Ok(Span {
                text: json_text(escaped),
                span: [start, end],
                path: Some(path.clone()),
            });
        }
        Ok(Span {
            text: self.text[start..end].iter().collect(),
            span: [start, end],
            path: None,
        })
    }
}

/// A read's span from its start and end logits over the keys: P(found), and the likeliest
/// start and end within `cap` tokens when found, the end at or after the start.
pub fn span(start: &[f32], end: &[f32], cap: usize) -> (f64, Option<(usize, usize)>) {
    let (start, end) = (log_softmax(start), log_softmax(end));
    let found = -start[0].exp_m1();
    if found < 0.5 || start.len() == 1 {
        return (found, None);
    }
    let reach = cap.min(start.len() - 1);
    let mut best = (f64::NEG_INFINITY, (1, 1));
    for first in 1..start.len() {
        let window = &end[first..(first + reach).min(end.len())];
        let (offset, score) =
            window
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |kept, (at, &score)| {
                    if score > kept.1 { (at, score) } else { kept }
                });
        if start[first] + score > best.0 {
            best = (start[first] + score, (first, first + offset));
        }
    }
    (found, Some(best.1))
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .map(|&logit| f64::from(logit - max).exp())
        .sum();
    let log_total = total.ln();
    logits
        .iter()
        .map(|&logit| f64::from(logit - max) - log_total)
        .collect()
}

/// The spans of a rendered JSON value's strings, their quotes excluded, with their RFC 6901
/// paths.
fn string_paths(text: &[char]) -> Vec<(usize, usize, String)> {
    let mut paths = Vec::new();
    value(text, 0, "", &mut paths);
    paths
}

fn skip(text: &[char], mut at: usize) -> usize {
    while text
        .get(at)
        .is_some_and(|character| character.is_whitespace())
    {
        at += 1;
    }
    at
}

/// The value at `at` under `path`, its strings' spans into `paths`: where it ends.
fn value(text: &[char], at: usize, path: &str, paths: &mut Vec<(usize, usize, String)>) -> usize {
    let at = skip(text, at);
    match text[at] {
        '{' => {
            let mut at = skip(text, at + 1);
            while text[at] != '}' {
                let (key, end) = string(text, at);
                let name = key.replace('~', "~0").replace('/', "~1");
                let colon = skip(text, end);
                at = skip(
                    text,
                    value(text, colon + 1, &format!("{path}/{name}"), paths),
                );
                if text[at] == ',' {
                    at = skip(text, at + 1);
                }
            }
            at + 1
        }
        '[' => {
            let mut at = skip(text, at + 1);
            let mut index = 0;
            while text[at] != ']' {
                at = skip(text, value(text, at, &format!("{path}/{index}"), paths));
                index += 1;
                if text[at] == ',' {
                    at = skip(text, at + 1);
                }
            }
            at + 1
        }
        '"' => {
            let (_, end) = string(text, at);
            paths.push((at + 1, end - 1, path.to_owned()));
            end
        }
        _ => {
            let scalar = text[at..].iter().position(|&character| {
                matches!(character, ',' | ']' | '}') || character.is_whitespace()
            });
            scalar.map_or(text.len(), |length| at + length)
        }
    }
}

/// The JSON string opening at `at`: its text, and where it ends, past its closing quote.
fn string(text: &[char], at: usize) -> (String, usize) {
    let mut end = at + 1;
    while text[end] != '"' {
        end += if text[end] == '\\' { 2 } else { 1 };
    }
    let quoted: String = text[at..=end].iter().collect();
    (json_text(quoted[1..quoted.len() - 1].to_owned()), end + 1)
}

/// A slice of a rendered JSON string as the characters it encodes.
fn json_text(escaped: String) -> String {
    // A span may cut an escape: such a slice stays as it is.
    serde_json::from_str::<String>(&format!("\"{escaped}\"")).unwrap_or(escaped)
}
