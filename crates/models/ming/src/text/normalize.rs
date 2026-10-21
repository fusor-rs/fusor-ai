//! A fragment's normalization: numbers as words, then `toolkit.normalize_text_kit` (the TTS
//! whitelist and CJK punctuation), around its paralinguistic tags.

use std::sync::LazyLock;

use regex::Regex;

use super::numbers::normalize_numbers;
use super::{CHINESE, is_chinese, is_cjk, regex};

const PARALINGUISTIC_TAGS: [&str; 14] = [
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
];

/// Punctuation the whitelist keeps besides letters and digits.
const WHITELIST_PUNCTUATION: &str =
    "，。！？；：、,.!?;: \u{2019}\u{2018}'\"()（）\u{201c}\u{201d}";
/// What every character outside the whitelist becomes.
const PAUSE: char = '，';

/// The fragment normalized around its paralinguistic tags, which keep their spacing.
pub(super) fn fragment(text: &str) -> String {
    static TAG: LazyLock<Regex> =
        LazyLock::new(|| regex(&PARALINGUISTIC_TAGS.map(regex::escape).join("|")));
    if !TAG.is_match(text) {
        return normalize_fragment(text);
    }
    let mut out = String::new();
    let mut start = 0;
    for tag in TAG.find_iter(text) {
        push_segment(&text[start..tag.start()], &mut out);
        out.push_str(tag.as_str());
        start = tag.end();
    }
    push_segment(&text[start..], &mut out);
    out
}

/// The text between two tags, normalized, keeping one space where it had leading or trailing
/// whitespace.
fn push_segment(segment: &str, out: &mut String) {
    if segment.trim().is_empty() {
        out.push_str(segment);
        return;
    }
    if segment.starts_with(char::is_whitespace) {
        out.push(' ');
    }
    out.push_str(normalize_fragment(segment).trim());
    if segment.ends_with(char::is_whitespace) {
        out.push(' ');
    }
}

fn normalize_fragment(text: &str) -> String {
    static WORD_HYPHEN: LazyLock<Regex> = LazyLock::new(|| regex(r"(\w)-(\w)"));
    let text = if is_chinese(text) {
        text.to_string()
    } else {
        normalize_numbers(text)
    };
    let text = WORD_HYPHEN.replace_all(&text, "$1 $2");
    let text = normalize_text_kit(&text);
    match text.strip_prefix(PAUSE) {
        Some(rest) => rest.to_string(),
        None => text,
    }
}

/// `toolkit.normalize_text_kit`.
fn normalize_text_kit(text: &str) -> String {
    let text = replace_outside_whitelist(text);
    let text = if text.chars().any(is_cjk) {
        cjk_punctuation(&text)
    } else {
        text
    };
    clean_text_formatting(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// `fallback_replace_non_alnum`: characters outside the TTS whitelist become one `，`.
fn replace_outside_whitelist(text: &str) -> String {
    let allowed = |ch: char| {
        CHINESE.contains(&ch) || ch.is_ascii_alphanumeric() || WHITELIST_PUNCTUATION.contains(ch)
    };
    let mut out = String::new();
    for ch in text.chars() {
        let kept = if allowed(ch) { ch } else { PAUSE };
        if !(kept == PAUSE && out.ends_with(PAUSE)) {
            out.push(kept);
        }
    }
    out.trim().to_string()
}

/// `clean_text_formatting`: leading punctuation dropped.
fn clean_text_formatting(text: &str) -> String {
    text.trim_start_matches(|ch: char| "，。！？；：、…….!?-—".contains(ch))
        .trim()
        .to_string()
}

/// A URL's stand-in while the punctuation around it is rewritten.
fn url_placeholder(index: usize) -> String {
    format!("URLHOLDER{index}")
}

/// `chinese_punc_en2cn`: ASCII punctuation to CJK, quotes paired, spaces around CJK
/// punctuation removed, stacked punctuation reduced to one mark.
fn cjk_punctuation(text: &str) -> String {
    static URL: LazyLock<Regex> = LazyLock::new(|| regex(r"https?://[^\s，。！？]+"));
    static AROUND: LazyLock<Regex> = LazyLock::new(|| {
        regex(r"[ \t\x{a0}\x{3000}]*([，。！？；：“”‘’（）【】《》——……、])[ \t\x{a0}\x{3000}]*")
    });
    static SPACES: LazyLock<Regex> = LazyLock::new(|| regex(r"[ \t\x{a0}\x{3000}]+"));
    let urls: Vec<&str> = URL.find_iter(text).map(|url| url.as_str()).collect();
    let mut text = text.to_string();
    for (i, url) in urls.iter().enumerate() {
        text = text.replace(url, &url_placeholder(i));
    }
    let out = ascii_to_cjk_punctuation(&text.replace("...", "……"));
    let mut text = AROUND.replace_all(&out, "$1").into_owned();
    loop {
        let deduped = dedupe_punctuation(&text);
        if deduped == text {
            break;
        }
        text = deduped;
    }
    let mut text = SPACES.replace_all(&text, " ").into_owned();
    for (i, url) in urls.iter().enumerate() {
        text = text.replace(&url_placeholder(i), url);
    }
    text
}

/// ASCII punctuation to its CJK form, alternating opening and closing quotes; a period between
/// digits stays.
fn ascii_to_cjk_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let (mut double, mut single) = (0, 0);
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        let decimal = ch == '.'
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        let mapped = match ch {
            '.' if decimal => ".",
            ',' => "，",
            '.' => "。",
            ';' => "；",
            ':' => "：",
            '!' => "！",
            '?' => "？",
            '(' => "（",
            ')' => "）",
            '[' => "【",
            ']' => "】",
            '<' => "《",
            '>' => "》",
            '_' => "——",
            '"' => {
                double += 1;
                if double % 2 == 1 { "“" } else { "”" }
            }
            '\'' => {
                single += 1;
                if single % 2 == 1 { "‘" } else { "’" }
            }
            _ => {
                out.push(ch);
                continue;
            }
        };
        out.push_str(mapped);
    }
    out
}

/// One pass of `chinese_punc_en2cn`'s stacked-punctuation rules.
fn dedupe_punctuation(text: &str) -> String {
    const PUNCTUATION: &str = "，。！？；：“”‘’（）【】《》——……、";
    const STRONG: &str = "。！？";
    const WEAK: &str = "，；、";
    let mut out: Vec<char> = Vec::new();
    for ch in text.chars() {
        let Some(&previous) = out.last() else {
            out.push(ch);
            continue;
        };
        let replaced = if ch == previous && PUNCTUATION.contains(ch) {
            Some(previous)
        } else if WEAK.contains(previous) && STRONG.contains(ch) {
            Some(ch)
        } else if STRONG.contains(previous) && (WEAK.contains(ch) || STRONG.contains(ch)) {
            Some(previous)
        } else if (previous == '、' && "，；".contains(ch))
            || ("，；".contains(previous) && ch == '、')
        {
            Some('、')
        } else if WEAK.contains(previous) && WEAK.contains(ch) {
            Some(previous)
        } else {
            None
        };
        match replaced {
            Some(kept) => *out.last_mut().expect("checked above") = kept,
            None => out.push(ch),
        }
    }
    out.into_iter().collect()
}
