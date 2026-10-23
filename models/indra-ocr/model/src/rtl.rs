//! Right-to-left text from recognizers that read every line left to right, in visual order:
//! `flip` turns visual order into logical order (and back: it is its own inverse). A line with
//! a right-to-left base reverses its units, each run of left-to-right text (Latin words, digits
//! and what lies between them) kept whole; a left-to-right line reverses each right-to-left run
//! in place. A unit is a base character with its combining marks; a mirrored bracket at
//! right-to-left level becomes its mirror image. A subset of UAX #9, as the reference's `rtl.py`.

use crate::text::{is_ltr, is_mark, is_rtl, reads_left_to_right};

const MIRRORED: &[(char, char)] = &[
    ('(', ')'),
    (')', '('),
    ('[', ']'),
    (']', '['),
    ('{', '}'),
    ('}', '{'),
    ('<', '>'),
    ('>', '<'),
    ('«', '»'),
    ('»', '«'),
    ('‹', '›'),
    ('›', '‹'),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    RightToLeft,
    LeftToRight,
    Neutral,
}

pub(crate) fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// Whether the line reads right to left: more right-to-left letters than left-to-right ones.
fn rtl_base(text: &str) -> bool {
    let right = text.chars().filter(|&character| is_rtl(character)).count();
    let left = text.chars().filter(|&character| is_ltr(character)).count();
    right > left
}

/// Base characters with their combining marks.
fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for character in text.chars() {
        match out.last_mut() {
            Some(last) if is_mark(character) => last.push(character),
            _ => out.push(character.to_string()),
        }
    }
    out
}

fn kind(cluster: &str) -> Kind {
    let first = base(cluster);
    if is_rtl(first) {
        Kind::RightToLeft
    } else if reads_left_to_right(first) {
        Kind::LeftToRight
    } else {
        Kind::Neutral
    }
}

/// `[start, end)` spans from a `strong` cluster to the last `strong` one before a cluster of
/// kind `stop`.
fn runs(kinds: &[Kind], strong: Kind, stop: Kind) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for (index, &kind) in kinds.iter().enumerate() {
        if kind == strong {
            open = Some((open.map_or(index, |(start, _)| start), index));
        } else if kind == stop
            && let Some((start, last)) = open.take()
        {
            spans.push((start, last + 1));
        }
    }
    spans.extend(open.map(|(start, last)| (start, last + 1)));
    spans
}

fn base(cluster: &str) -> char {
    cluster
        .chars()
        .next()
        .expect("a cluster starts with its base character")
}

fn mirror(cluster: &str) -> String {
    let first = base(cluster);
    let mirrored = MIRRORED
        .iter()
        .find(|&&(from, _)| from == first)
        .map_or(first, |&(_, to)| to);
    std::iter::once(mirrored)
        .chain(cluster.chars().skip(1))
        .collect()
}

/// Visual order to logical order, brackets mirrored. `base_rtl`: the paragraph's direction when
/// known, else decided by the line's letters.
pub(crate) fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let clusters = clusters(text);
    let kinds: Vec<Kind> = clusters.iter().map(|cluster| kind(cluster)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        let mut units = Vec::new();
        let mut next = 0;
        let spans = runs(&kinds, Kind::LeftToRight, Kind::RightToLeft);
        let end = (clusters.len(), clusters.len());
        for (start, stop) in spans.into_iter().chain([end]) {
            units.extend(clusters[next..start].iter().map(|cluster| mirror(cluster)));
            if start < stop {
                units.push(clusters[start..stop].concat());
            }
            next = stop;
        }
        return units.into_iter().rev().collect();
    }
    let mut out = clusters.clone();
    for (start, stop) in runs(&kinds, Kind::RightToLeft, Kind::LeftToRight) {
        for (slot, cluster) in out[start..stop]
            .iter_mut()
            .zip(clusters[start..stop].iter().rev())
        {
            *slot = mirror(cluster);
        }
    }
    out.concat()
}
