//! Cutting text into fragments of at most `MAX_SEMANTIC_LENGTH` semantic units
//! (`text_segment_cut.cut_text_by_semantic_length(text, 50, 5)["fragments"]`).

use std::sync::LazyLock;

use regex::Regex;

use super::{CJK, is_cjk, regex};

const MAX_SEMANTIC_LENGTH: usize = 50;
const MIN_TAIL_LENGTH: usize = 5;
/// Stands for a period that does not end a sentence (in a decimal or an acronym) while cutting.
const DOT: &str = "##DOT##";
/// The reference applies its acronym rule three times: one pass protects every other dot of a
/// run like `U.S.A.`, since its matches cannot overlap.
const ACRONYM_PASSES: usize = 3;
const SENTENCE_ENDS: &str = "。！？";
const CLAUSE_ENDS: &str = "，;；";

/// The fragments of `text`.
pub(super) fn cut(text: &str) -> Vec<String> {
    static DECIMAL: LazyLock<Regex> = LazyLock::new(|| regex(r"(\d)\.(\d)"));
    static ACRONYM: LazyLock<Regex> = LazyLock::new(|| regex(r"([A-Z])\.([A-Z])"));
    if !has_valid_content(text) {
        return Vec::new();
    }
    let protected = format!("${{1}}{DOT}${{2}}");
    let mut processed = DECIMAL.replace_all(text, &protected).into_owned();
    for _ in 0..ACRONYM_PASSES {
        processed = ACRONYM.replace_all(&processed, &protected).into_owned();
    }
    let processed = processed.replace('\n', " ").replace("。，", "。");
    if semantic_length(&processed) <= MAX_SEMANTIC_LENGTH {
        return vec![processed.replace(DOT, ".")];
    }
    let normalized = processed
        .replace('.', "。")
        .replace('!', "！")
        .replace('?', "？")
        .replace(',', "，");
    let mut fragments = Vec::new();
    for sentence in sentences(&normalized) {
        append_clauses(&mut fragments, &clauses(&sentence));
    }
    fragments
        .into_iter()
        .map(|fragment| fragment.replace(DOT, "."))
        .collect()
}

/// One CJK character or one run of Latin letters counts 1.
fn semantic_length(text: &str) -> usize {
    static WORD: LazyLock<Regex> = LazyLock::new(|| regex(r"[a-zA-Z]+"));
    text.chars().filter(|&ch| is_cjk(ch)).count() + WORD.find_iter(text).count()
}

/// Anything but ASCII punctuation and whitespace (Python's `string.punctuation + whitespace`).
fn has_valid_content(text: &str) -> bool {
    text.chars()
        .any(|ch| !(ch.is_ascii_punctuation() || " \t\n\r\x0b\x0c".contains(ch)))
}

/// The pieces of `text` that end after any of `marks`, each trimmed; empty pieces dropped.
fn split_after<'text>(text: &'text str, marks: &str) -> Vec<&'text str> {
    text.split_inclusive(|ch| marks.contains(ch))
        .map(str::trim)
        .filter(|piece| !piece.is_empty())
        .collect()
}

/// Sentences ending in `。！？` (kept); the last one gets a `。` if it lacks an end mark.
fn sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = split_after(text, SENTENCE_ENDS)
        .into_iter()
        .map(str::to_string)
        .collect();
    if let Some(last) = sentences.last_mut()
        && !last.ends_with(|ch| SENTENCE_ENDS.contains(ch))
    {
        last.push('。');
    }
    sentences
}

/// Clauses of a sentence ending in `，;；`; content-free pieces join the previous clause.
fn clauses(sentence: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    for clause in split_after(sentence, CLAUSE_ENDS) {
        match clauses.last_mut() {
            Some(last) if !has_valid_content(clause) => last.push_str(clause),
            _ if has_valid_content(clause) => clauses.push(clause.to_string()),
            _ => {}
        }
    }
    clauses
}

/// The clauses of a sentence into `fragments`: a short clause joined to the next while they
/// fit, a long one cut.
fn append_clauses(fragments: &mut Vec<String>, clauses: &[String]) {
    let mut i = 0;
    while i < clauses.len() {
        let clause = &clauses[i];
        let length = semantic_length(clause);
        if length < MIN_TAIL_LENGTH && i + 1 < clauses.len() {
            let combined = format!("{clause}{}", clauses[i + 1]);
            if semantic_length(&combined) <= MAX_SEMANTIC_LENGTH {
                append_fragment(fragments, &combined);
                i += 2;
                continue;
            }
        }
        if length > MAX_SEMANTIC_LENGTH {
            for part in split_long_fragment(clause) {
                append_fragment(fragments, &part);
            }
        } else {
            append_fragment(fragments, clause);
        }
        i += 1;
    }
}

/// `append_text_fragment`: joins `text` to the last fragment while it fits.
fn append_fragment(fragments: &mut Vec<String>, text: &str) {
    let text = text.trim_start_matches(|ch: char| "，,:;".contains(ch) || ch.is_whitespace());
    if !has_valid_content(text) {
        return;
    }
    let Some(last) = fragments.last_mut() else {
        fragments.push(text.into());
        return;
    };
    let new_length = semantic_length(text);
    if semantic_length(last) + new_length > MAX_SEMANTIC_LENGTH
        || last.ends_with(|ch| SENTENCE_ENDS.contains(ch)) && new_length < MIN_TAIL_LENGTH
    {
        fragments.push(text.into());
        return;
    }
    let starts_alphanumeric = text
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_alphanumeric());
    if !last.ends_with(' ') && starts_alphanumeric {
        last.push(' ');
    }
    last.push_str(text);
}

/// `split_long_fragment`: a clause over the limit, cut after punctuation, else at words.
fn split_long_fragment(text: &str) -> Vec<String> {
    static BOUNDARY: LazyLock<Regex> = LazyLock::new(|| regex(r"[，、；。！？,;!?]|\s+[-•·]"));
    let mut ends: Vec<usize> = Vec::new();
    for boundary in BOUNDARY.find_iter(text) {
        match ends.last_mut() {
            Some(end) if *end == boundary.start() => *end = boundary.end(),
            _ => ends.push(boundary.end()),
        }
    }
    if ends.last() != Some(&text.len()) {
        ends.push(text.len());
    }
    let starts = std::iter::once(0).chain(ends.iter().copied());
    let pieces = starts.zip(&ends).map(|(start, &end)| &text[start..end]);
    pack(pieces.collect(), split_by_word)
}

/// `split_by_word`: a piece over the limit cut into words, CJK characters and punctuation.
fn split_by_word(text: &str) -> Vec<String> {
    static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
        let cjk = format!(
            r"\x{{{:x}}}-\x{{{:x}}}",
            u32::from(*CJK.start()),
            u32::from(*CJK.end())
        );
        regex(&format!(
            r"\s+|[，、；。！？,;!?]|[{cjk}]|[a-zA-Z]+|[^a-zA-Z{cjk}]+"
        ))
    });
    let tokens = TOKEN.find_iter(text).map(|token| token.as_str()).collect();
    pack(tokens, |token| vec![token.to_string()])
}

/// Packs `pieces` greedily into fragments of at most `MAX_SEMANTIC_LENGTH`; a piece over the
/// limit closes the fragment before it and is cut by `oversized`.
fn pack(pieces: Vec<&str>, oversized: impl Fn(&str) -> Vec<String>) -> Vec<String> {
    let mut fragments = Vec::new();
    let mut current = String::new();
    let mut current_length = 0;
    for piece in pieces {
        let length = semantic_length(piece);
        if current_length + length <= MAX_SEMANTIC_LENGTH {
            current.push_str(piece);
            current_length += length;
            continue;
        }
        if !current.is_empty() {
            fragments.push(std::mem::take(&mut current));
        }
        current_length = 0;
        if length > MAX_SEMANTIC_LENGTH {
            fragments.extend(oversized(piece));
        } else {
            current = piece.to_string();
            current_length = length;
        }
    }
    if !current.is_empty() {
        fragments.push(current);
    }
    fragments
}
