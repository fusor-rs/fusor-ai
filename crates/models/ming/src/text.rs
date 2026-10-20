//! The text frontend (`MingTTS.prepare_text` in the reference, `frontend/`): split the input
//! into fragments of at most 50 semantic units (CJK characters + Latin words), synthesized one
//! by one, and normalize each fragment.
//!
//! Per fragment: English numbers become words (`number_en.normalize_numbers`, inflect's
//! wording), then `toolkit.normalize_text_kit` (characters outside the TTS whitelist become `，`,
//! CJK text gets CJK punctuation). Paralinguistic tags (`[Laughter]` …) are routed around both.
//!
//! Not ported: the pynini FST normalizer (`TalkerTN`) that runs between those two steps in the
//! reference. In its place a hyphen inside a word (`ninety-eight`, `well-known`) becomes a space,
//! because the whitelist step would otherwise turn it into a pause (`ninety，eight`).

use std::sync::LazyLock;

use regex::{Captures, Regex};

const MAX_SEMANTIC_LENGTH: usize = 50;
const MIN_TAIL_LENGTH: usize = 5;

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

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("the patterns in this module are valid")
}

/// How a request's text becomes fragments (`MingTTS.prepare_text`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segmentation {
    /// Built-in voices, Chinese/English: cut into ≤ 50 semantic units, then normalized.
    Sentences,
    /// Voice design and free-form instructions apply to the utterance as a whole: one
    /// fragment, normalized.
    Whole,
    /// German, Spanish, French, Japanese, Korean: cut, never normalized (the normalizer reads
    /// any CJK ideograph as Chinese, and the number words are English).
    Verbatim,
}

/// The fragments to synthesize.
pub fn fragments(text: &str, segmentation: Segmentation) -> Vec<String> {
    let text = strip_markdown_code(text);
    let fragments = match segmentation {
        Segmentation::Sentences => cut_by_semantic_length(&add_period(&text)),
        Segmentation::Whole => vec![add_period(&text)],
        Segmentation::Verbatim => cut_by_semantic_length(text.trim()),
    };
    fragments
        .into_iter()
        .map(|fragment| match segmentation {
            Segmentation::Verbatim => fragment,
            _ => protect_paralinguistic_tags(&fragment),
        })
        .filter(|fragment| !fragment.trim().is_empty())
        .collect()
}

fn is_cjk(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fa5}').contains(&ch)
}

fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

/// `text_segment_cut.is_chinese` (a wider range than `is_cjk`, as in the reference).
fn is_chinese(text: &str) -> bool {
    text.chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
}

/// `toolkit.strip_markdown_code`: inline code marks removed.
fn strip_markdown_code(text: &str) -> String {
    static DOUBLE: LazyLock<Regex> = LazyLock::new(|| regex(r"``(.+?)``"));
    static SINGLE: LazyLock<Regex> = LazyLock::new(|| regex(r"`(.+?)`"));
    let text = DOUBLE.replace_all(text, "$1");
    SINGLE.replace_all(&text, "$1").replace('`', "")
}

/// `toolkit.add_period`: a final period (`。` after CJK), a weak final mark promoted to one.
fn add_period(text: &str) -> String {
    let stripped = text.trim_end();
    let Some(last) = stripped.chars().last() else {
        return ".".into();
    };
    if "，,、;；".contains(last) {
        let body = &stripped[..stripped.len() - last.len_utf8()];
        return format!("{body}{}", if is_chinese(stripped) { "。" } else { "." });
    }
    if !last.is_alphanumeric() {
        return stripped.into();
    }
    let cjk =
        ('\u{4e00}'..='\u{9fff}').contains(&last) || ('\u{3400}'..='\u{4dbf}').contains(&last);
    format!("{stripped}{}", if cjk { "。" } else { "." })
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

const DOT: &str = "##DOT##";

/// `text_segment_cut.cut_text_by_semantic_length(text, 50, 5)["fragments"]`.
fn cut_by_semantic_length(text: &str) -> Vec<String> {
    static DECIMAL: LazyLock<Regex> = LazyLock::new(|| regex(r"(\d)\.(\d)"));
    static ACRONYM: LazyLock<Regex> = LazyLock::new(|| regex(r"([A-Z])\.([A-Z])"));
    if !has_valid_content(text) {
        return Vec::new();
    }
    let mut processed = DECIMAL
        .replace_all(text, format!("${{1}}{DOT}${{2}}"))
        .into_owned();
    for _ in 0..3 {
        processed = ACRONYM
            .replace_all(&processed, format!("${{1}}{DOT}${{2}}"))
            .into_owned();
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
        let clauses = clauses(&sentence);
        let mut i = 0;
        while i < clauses.len() {
            let clause = &clauses[i];
            let length = semantic_length(clause);
            if length < MIN_TAIL_LENGTH && i + 1 < clauses.len() {
                let combined = format!("{clause}{}", clauses[i + 1]);
                if semantic_length(&combined) <= MAX_SEMANTIC_LENGTH {
                    append_fragment(&mut fragments, &combined);
                    i += 2;
                    continue;
                }
            }
            if length > MAX_SEMANTIC_LENGTH {
                for part in split_long_fragment(clause) {
                    append_fragment(&mut fragments, &part);
                }
            } else {
                append_fragment(&mut fragments, clause);
            }
            i += 1;
        }
    }
    fragments
        .into_iter()
        .map(|fragment| fragment.replace(DOT, "."))
        .collect()
}

/// Sentences ending in `。！？` (kept); the last one gets a `。` if it lacks an end mark.
fn sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if "。！？".contains(ch) {
            let sentence = current.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        let mut last = current.trim().to_string();
        if !last.ends_with(['。', '！', '？']) {
            last.push('。');
        }
        sentences.push(last);
    }
    sentences
}

/// Clauses of a sentence ending in `，;；`; content-free pieces join the previous clause.
fn clauses(sentence: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    let mut current = String::new();
    let push = |current: &str, clauses: &mut Vec<String>| {
        let clause = current.trim();
        if !clause.is_empty() && has_valid_content(clause) {
            clauses.push(clause.to_string());
        } else if !clause.is_empty()
            && let Some(last) = clauses.last_mut()
        {
            last.push_str(clause);
        }
    };
    for ch in sentence.chars() {
        current.push(ch);
        if "，;；".contains(ch) {
            push(&current, &mut clauses);
            current.clear();
        }
    }
    if !current.is_empty() {
        push(&current, &mut clauses);
    }
    clauses
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
        || last.ends_with(['。', '！', '？']) && new_length < MIN_TAIL_LENGTH
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

/// `split_long_fragment`: a clause over the limit, cut at punctuation, else at words.
fn split_long_fragment(text: &str) -> Vec<String> {
    static BOUNDARY: LazyLock<Regex> = LazyLock::new(|| regex(r"[，、；。！？,;!?]|\s+[-•·]"));
    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0;
    for boundary in BOUNDARY.find_iter(text) {
        if boundary.start() > start {
            pieces.push(text[start..boundary.start()].to_string());
        }
        match pieces.last_mut() {
            Some(last) => last.push_str(boundary.as_str()),
            None => pieces.push(boundary.as_str().to_string()),
        }
        start = boundary.end();
    }
    if start < text.len() {
        pieces.push(text[start..].to_string());
    }
    let mut fragments = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if semantic_length(&piece) > MAX_SEMANTIC_LENGTH {
            if !current.is_empty() {
                fragments.push(std::mem::take(&mut current));
            }
            fragments.extend(split_by_word(&piece));
        } else if semantic_length(&current) + semantic_length(&piece) <= MAX_SEMANTIC_LENGTH {
            current.push_str(&piece);
        } else {
            if !current.is_empty() {
                fragments.push(std::mem::take(&mut current));
            }
            current = piece;
        }
    }
    if !current.is_empty() {
        fragments.push(current);
    }
    fragments
}

fn split_by_word(text: &str) -> Vec<String> {
    static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
        regex(r"\s+|[，、；。！？,;!?]|[\x{4e00}-\x{9fa5}]|[a-zA-Z]+|[^a-zA-Z\x{4e00}-\x{9fa5}]+")
    });
    let mut fragments = Vec::new();
    let mut current = String::new();
    for token in TOKEN.find_iter(text).map(|token| token.as_str()) {
        let length = semantic_length(token);
        if semantic_length(&current) + length <= MAX_SEMANTIC_LENGTH {
            current.push_str(token);
            continue;
        }
        if !current.is_empty() {
            fragments.push(std::mem::take(&mut current));
        }
        if length > MAX_SEMANTIC_LENGTH {
            fragments.push(token.to_string());
        } else {
            current = token.to_string();
        }
    }
    if !current.is_empty() {
        fragments.push(current);
    }
    fragments
}

/// Normalizes the text around paralinguistic tags, keeping the tags and their spacing.
fn protect_paralinguistic_tags(text: &str) -> String {
    static TAG: LazyLock<Regex> =
        LazyLock::new(|| regex(&PARALINGUISTIC_TAGS.map(regex::escape).join("|")));
    if !TAG.is_match(text) {
        return normalize_fragment(text);
    }
    let mut out = String::new();
    let mut start = 0;
    let segment = |segment: &str, out: &mut String| {
        if segment.trim().is_empty() {
            out.push_str(segment);
            return;
        }
        let lead = if segment.starts_with(char::is_whitespace) {
            " "
        } else {
            ""
        };
        let trail = if segment.ends_with(char::is_whitespace) {
            " "
        } else {
            ""
        };
        out.push_str(&format!(
            "{lead}{}{trail}",
            normalize_fragment(segment).trim()
        ));
    };
    for tag in TAG.find_iter(text) {
        segment(&text[start..tag.start()], &mut out);
        out.push_str(tag.as_str());
        start = tag.end();
    }
    segment(&text[start..], &mut out);
    out
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
    text.strip_prefix('，').map_or(text.clone(), str::to_string)
}

/// `toolkit.normalize_text_kit`.
fn normalize_text_kit(text: &str) -> String {
    let text = replace_outside_whitelist(text);
    let text = if has_cjk(&text) {
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
        ('\u{4e00}'..='\u{9fff}').contains(&ch)
            || ch.is_ascii_alphanumeric()
            || "，。！？；：、,.!?;: \u{2019}\u{2018}'\"()（）\u{201c}\u{201d}".contains(ch)
    };
    let mut out = String::new();
    for ch in text.chars() {
        let kept = if allowed(ch) { ch } else { '，' };
        if !(kept == '，' && out.ends_with('，')) {
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

/// `chinese_punc_en2cn`: ASCII punctuation to CJK, quotes paired, spaces around CJK
/// punctuation removed, stacked punctuation reduced to one mark.
fn cjk_punctuation(text: &str) -> String {
    static URL: LazyLock<Regex> = LazyLock::new(|| regex(r"https?://[^\s，。！？]+"));
    static AROUND: LazyLock<Regex> = LazyLock::new(|| {
        regex(r"[ \t\x{a0}\x{3000}]*([，。！？；：“”‘’（）【】《》——……、])[ \t\x{a0}\x{3000}]*")
    });
    static SPACES: LazyLock<Regex> = LazyLock::new(|| regex(r"[ \t\x{a0}\x{3000}]+"));
    let urls: Vec<String> = URL
        .find_iter(text)
        .map(|url| url.as_str().to_string())
        .collect();
    let mut text = text.to_string();
    for (i, url) in urls.iter().enumerate() {
        text = text.replace(url, &format!("URLHOLDER{i}"));
    }
    let out = ascii_to_cjk_punctuation(&text.replace("...", "……"));
    let mut text = AROUND.replace_all(&out, "$1").into_owned();
    loop {
        let before = text.clone();
        text = dedupe_punctuation(&text);
        if text == before {
            break;
        }
    }
    let mut text = SPACES.replace_all(&text, " ").into_owned();
    for (i, url) in urls.iter().enumerate() {
        text = text.replace(&format!("URLHOLDER{i}"), url);
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

/// `number_en.normalize_numbers`.
fn normalize_numbers(text: &str) -> String {
    static COMMA_NUMBER: LazyLock<Regex> = LazyLock::new(|| regex(r"([0-9][0-9,]+[0-9])"));
    static UNIT: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r"(?i)\b(-?\d+(?:\.\d+)?)\s*(ms|s|Hz|kHz|MHz|GHz|kb|mb|gb|tb|KB|MB|GB|TB|bps|kbps|Mbps|Gbps|cm|km|kg|V|A|W|°C|°F)\b",
        )
    });
    static POUNDS: LazyLock<Regex> = LazyLock::new(|| regex(r"£(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)"));
    static DOLLARS: LazyLock<Regex> = LazyLock::new(|| regex(r"\$(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)"));
    static FRACTION: LazyLock<Regex> = LazyLock::new(|| regex(r"([0-9]+)/([0-9]+)"));
    static PERCENT: LazyLock<Regex> = LazyLock::new(|| regex(r"(-?[0-9.,]*[0-9]+)%"));
    static ORDINAL: LazyLock<Regex> = LazyLock::new(|| regex(r"\b([0-9]+)(st|nd|rd|th)\b"));
    static VERSION: LazyLock<Regex> =
        LazyLock::new(|| regex(r"\b([a-zA-Z]+)(-?)([0-9]+(?:\.[0-9]+)?)\b"));
    static NUMBER: LazyLock<Regex> = LazyLock::new(|| regex(r"\b-?[0-9]+(?:\.[0-9]+)?\b"));

    let text = COMMA_NUMBER.replace_all(text, |caps: &Captures| caps[1].replace(',', ""));
    let text = UNIT.replace_all(&text, |caps: &Captures| {
        let unit = unit_word(&caps[2]);
        quantity(&caps[1]).map_or_else(
            || format!(" {} {} ", &caps[1], &caps[2]),
            |words| format!(" {words} {unit} "),
        )
    });
    let text = POUNDS.replace_all(&text, |caps: &Captures| currency(&caps[1], "pound"));
    let text = DOLLARS.replace_all(&text, |caps: &Captures| currency(&caps[1], "dollar"));
    let text = FRACTION.replace_all(&text, |caps: &Captures| {
        fraction(&caps[1], &caps[2]).unwrap_or_else(|| caps[0].to_string())
    });
    let text = PERCENT.replace_all(&text, |caps: &Captures| {
        format!(
            " {} percent ",
            quantity(&caps[1]).unwrap_or_else(|| caps[1].to_string())
        )
    });
    let text = ORDINAL.replace_all(&text, |caps: &Captures| {
        number_words(&caps[1], AND)
            .map_or_else(|| caps[0].to_string(), |words| format!(" {words} "))
    });
    let text = VERSION.replace_all(&text, |caps: &Captures| {
        match decimal_words(&caps[3], AND) {
            Some(words) => format!("{} {words}", &caps[1]),
            None => caps[0].to_string(),
        }
    });
    let text = NUMBER.replace_all(&text, |caps: &Captures| {
        format!(
            " {} ",
            quantity(&caps[0]).unwrap_or_else(|| caps[0].to_string())
        )
    });
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// inflect's default `andword`; the reference passes `andword=""` for plain quantities.
const AND: &str = "and";

fn unit_word(unit: &str) -> String {
    let lower = unit.to_lowercase();
    let word = match lower.as_str() {
        "ms" => "milliseconds",
        "s" => "seconds",
        "hz" => "hertz",
        "khz" => "kilohertz",
        "mhz" => "megahertz",
        "ghz" => "gigahertz",
        "kb" => "kilobytes",
        "mb" => "megabytes",
        "gb" => "gigabytes",
        "tb" => "terabytes",
        "kbps" => "kilobits per second",
        "mbps" => "megabits per second",
        "gbps" => "gigabits per second",
        "bps" => "bits per second",
        "cm" => "centimeters",
        "km" => "kilometers",
        "kg" => "kilograms",
        "v" => "volts",
        "a" => "amperes",
        "w" => "watts",
        "°c" => "degrees celsius",
        "°f" => "degrees fahrenheit",
        _ => return lower,
    };
    word.into()
}

/// A signed decimal in words (`andword=""`): `-7.5` → "minus seven point five".
fn quantity(text: &str) -> Option<String> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let words = decimal_words(if digits.is_empty() { "0" } else { digits }, "")?;
    Some(if negative {
        format!("minus {words}")
    } else {
        words
    })
}

/// `12.99` → "twelve point nine nine"; the integer part "0" or empty reads "zero".
fn decimal_words(text: &str, andword: &str) -> Option<String> {
    let Some((integer, decimals)) = text.split_once('.') else {
        return number_words(text, andword);
    };
    let integer = if integer.is_empty() || integer == "0" {
        "zero".to_string()
    } else {
        number_words(integer, andword)?
    };
    if decimals.is_empty() || !decimals.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let digits: Vec<&str> = decimals
        .bytes()
        .map(|digit| UNITS[usize::from(digit - b'0')])
        .collect();
    Some(format!("{integer} point {}", digits.join(" ")))
}

fn currency(text: &str, unit: &str) -> String {
    let clean = text.trim_start_matches('-');
    let clean = if clean.is_empty() { "0" } else { clean };
    let Some(words) = quantity(text) else {
        return format!(" {clean} {unit}s ");
    };
    let one = clean.parse::<f64>().is_ok_and(|value| value.abs() == 1.0);
    format!(" {words} {unit}{} ", if one { "" } else { "s" })
}

fn fraction(numerator: &str, denominator: &str) -> Option<String> {
    let (count, parts): (u64, u64) = (numerator.parse().ok()?, denominator.parse().ok()?);
    let top = number_words(numerator, AND)?;
    Some(match (count, parts) {
        (1, 2) => " one half ".into(),
        (1, 4) => " one quarter ".into(),
        (_, 2) => format!(" {top} {} ", if count == 1 { "half" } else { "halves" }),
        (_, 4) => format!(
            " {top} {} ",
            if count == 1 { "quarter" } else { "quarters" }
        ),
        _ => format!(" {top} {} ", ordinal(&number_words(denominator, AND)?)),
    })
}

const UNITS: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];
const TEENS: [&str; 10] = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const MILLS: [&str; 7] = [
    "",
    " thousand",
    " million",
    " billion",
    " trillion",
    " quadrillion",
    " quintillion",
];

/// inflect's `number_to_words(n, andword=…)` for a string of digits: three-digit groups
/// joined by ", ", and the final ", word" read as " and word" when it is a single word.
fn number_words(digits: &str, andword: &str) -> Option<String> {
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some("zero".into());
    }
    let groups: Vec<&str> = {
        let head = digits.len() % 3;
        let mut groups = Vec::new();
        if head > 0 {
            groups.push(&digits[..head]);
        }
        groups.extend(
            digits.as_bytes()[head..]
                .chunks(3)
                .map(|chunk| std::str::from_utf8(chunk).expect("ASCII digits")),
        );
        groups
    };
    if groups.len() > MILLS.len() {
        return None;
    }
    let mut parts = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let value: usize = group.parse().expect("ASCII digits");
        let (hundreds, tens, units) = (value / 100, value / 10 % 10, value % 10);
        let mill = MILLS[groups.len() - 1 - i];
        let below_hundred = if tens == 1 {
            TEENS[units].to_string()
        } else {
            format!(
                "{}{}{}",
                TENS[tens],
                if tens > 0 && units > 0 { "-" } else { "" },
                if units > 0 { UNITS[units] } else { "" }
            )
        };
        if hundreds > 0 {
            let rest = if tens > 0 || units > 0 {
                format!(" {andword} {below_hundred}")
            } else {
                String::new()
            };
            parts.push(format!("{} hundred{rest}{mill}", UNITS[hundreds]));
        } else if value > 0 {
            parts.push(format!("{below_hundred}{mill}"));
        }
    }
    let mut words = parts.join(", ");
    if let Some((head, last)) = words.rsplit_once(", ")
        && !last.contains(' ')
    {
        words = format!("{head} {andword} {last}");
    }
    Some(words.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// inflect's `ordinal` of a number in words: its last word made ordinal.
fn ordinal(words: &str) -> String {
    let split = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(split);
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        word if word.ends_with('y') => format!("{}ieth", &word[..word.len() - 1]),
        word => format!("{word}th"),
    };
    format!("{head}{last}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_like_inflect() {
        let cases = [
            ("0", "zero"),
            ("21", "twenty-one"),
            ("101", "one hundred one"),
            ("1001", "one thousand one"),
            ("1100", "one thousand, one hundred"),
            ("1998", "one thousand, nine hundred ninety-eight"),
            ("2025", "two thousand twenty-five"),
            (
                "123456",
                "one hundred twenty-three thousand, four hundred fifty-six",
            ),
            ("1000001", "one million one"),
            (
                "2345678",
                "two million, three hundred forty-five thousand, six hundred seventy-eight",
            ),
        ];
        for (digits, words) in cases {
            assert_eq!(number_words(digits, "").as_deref(), Some(words), "{digits}");
        }
        assert_eq!(
            number_words("2025", AND).as_deref(),
            Some("two thousand and twenty-five")
        );
        assert_eq!(ordinal("twenty-one"), "twenty-first");
        assert_eq!(ordinal("twenty"), "twentieth");
    }

    #[test]
    fn normalize_numbers_matches_the_reference() {
        let cases = [
            (
                "In 1998 the company sold 2,500 units, which was a forty percent increase over the previous year.",
                "In one thousand, nine hundred ninety-eight the company sold two thousand, five hundred units, which was a forty percent increase over the previous year.",
            ),
            (
                "It costs $12.99 or £1, and 3/4 of 10ms is 7.5 ms, 25% vs -3.5%.",
                "It costs twelve point nine nine dollars or one pound , and three quarters of ten milliseconds is seven point five milliseconds , twenty-five percent vs minus three point five percent .",
            ),
            (
                "Meet me on the 21st at 3:30, version v2 and gpt-4.5 ready.",
                "Meet me on the twenty-one at three : thirty , version v two and gpt four point five ready.",
            ),
            (
                "Prices rose 1.5 percent in Q3 2024!",
                "Prices rose one point five percent in Q three two thousand twenty-four !",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_numbers(input), expected);
        }
    }

    #[test]
    fn long_text_is_cut_like_the_reference() {
        let text = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold. Somewhere below, a train whistled twice, and the lights of the village flickered on one by one. Nobody in the valley seemed to notice; they were busy with supper, with stories, and with the small business of ending a long day. It was 7.5 degrees by nightfall";
        let expected = [
            "The evening wind drifted down from the mountain pass， and the distant clouds slowly turned gold。 Somewhere below，a train whistled twice， and the lights of the village flickered on one by one。 Nobody in the valley seemed to notice; they were busy with supper，",
            "with stories，and with the small business of ending a long day。 It was 7.5 degrees by nightfall。",
        ];
        assert_eq!(cut_by_semantic_length(&add_period(text)), expected);
    }

    #[test]
    fn a_fragment_without_tags_keeps_no_trailing_space() {
        assert_eq!(
            protect_paralinguistic_tags("seven eight nine ten "),
            "seven eight nine ten"
        );
        assert_eq!(
            protect_paralinguistic_tags("Well [Laughter] fine"),
            "Well [Laughter] fine"
        );
    }

    #[test]
    fn short_text_is_one_fragment_kept_as_is() {
        let text = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";
        assert_eq!(fragments(text, Segmentation::Sentences), [text]);
    }
}
