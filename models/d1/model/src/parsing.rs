//! A read's written value, as d1's reference parses it (`parsing.py`): no arithmetic, unit
//! conversion or invented date components. A writing that does not settle one value reads as
//! none.

use std::collections::BTreeSet;

use fancy_regex::{Captures, Regex};

use crate::Error;

fn pattern(source: &str) -> Regex {
    Regex::new(source).expect("the reference's patterns compile")
}

fn whole(source: &str) -> Regex {
    pattern(&format!(r"\A(?:{source})\z"))
}

macro_rules! patterns {
    ($($name:ident = $source:expr;)*) => {
        $(static $name: LazyLock<Regex> = LazyLock::new(|| $source);)*
    };
}

mod date;
mod number;

pub use date::{Dated, parse_date};
pub use number::{parse_number, written_unit};

/// The alternatives longest first, escaped, as one alternation.
fn alternation<'a>(alternatives: impl Iterator<Item = &'a str>) -> String {
    let mut alternatives: Vec<&str> = alternatives.collect();
    alternatives
        .sort_by_key(|alternative| (std::cmp::Reverse(alternative.chars().count()), *alternative));
    let escaped: Vec<String> = alternatives
        .iter()
        .map(|alternative| fancy_regex::escape(alternative).into_owned())
        .collect();
    escaped.join("|")
}

fn failed(error: &fancy_regex::Error) -> Error {
    Error::Format(format!("reading a written value: {error}"))
}

fn captures<'t>(pattern: &Regex, text: &'t str) -> Result<Option<Captures<'t>>, Error> {
    pattern.captures(text).map_err(|error| failed(&error))
}

fn matches(pattern: &Regex, text: &str) -> Result<bool, Error> {
    pattern.is_match(text).map_err(|error| failed(&error))
}

fn every<'t>(pattern: &Regex, text: &'t str) -> Result<Vec<Captures<'t>>, Error> {
    pattern
        .captures_iter(text)
        .collect::<Result<_, _>>()
        .map_err(|error| failed(&error))
}

fn group<'t>(captures: &Captures<'t>, index: usize) -> &'t str {
    captures.get(index).map_or("", |found| found.as_str())
}

fn removed(pattern: &Regex, text: &str, replacement: &str) -> String {
    pattern.replace_all(text, replacement).into_owned()
}

/// The one value the evidence settles on; none when it holds none or disagrees with itself.
fn one<T: Copy>(values: &BTreeSet<T>) -> Option<T> {
    let mut values = values.iter();
    match (values.next(), values.next()) {
        (Some(&value), None) => Some(value),
        _ => None,
    }
}

/// Python's `str.casefold`, as far as the patterns read it: lower case, ß as ss.
fn casefold(text: &str) -> String {
    text.to_lowercase().replace('ß', "ss")
}
