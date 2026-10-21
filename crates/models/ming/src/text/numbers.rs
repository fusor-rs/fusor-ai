//! English numbers in words, as the reference's `number_en.normalize_numbers` writes them with
//! inflect's `number_to_words`.

use std::sync::LazyLock;

use regex::{Captures, Regex};

use super::regex;

/// `number_en.normalize_numbers`.
pub(super) fn normalize_numbers(text: &str) -> String {
    static COMMA_NUMBER: LazyLock<Regex> = LazyLock::new(|| regex(r"([0-9][0-9,]+[0-9])"));
    static UNIT: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r"(?i)\b(-?\d+(?:\.\d+)?)\s*(ms|s|hz|khz|mhz|ghz|kb|mb|gb|tb|bps|kbps|mbps|gbps|cm|km|kg|v|a|w|°c|°f)\b",
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

/// Whether `text` has a leading minus, and its digits ("0" when there are none).
fn split_sign(text: &str) -> (bool, &str) {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    (negative, if digits.is_empty() { "0" } else { digits })
}

/// A signed decimal in words (`andword=""`): `-7.5` → "minus seven point five".
fn quantity(text: &str) -> Option<String> {
    let (negative, digits) = split_sign(text);
    let words = decimal_words(digits, "")?;
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
    let (_, digits) = split_sign(text);
    let Some(words) = quantity(text) else {
        return format!(" {digits} {unit}s ");
    };
    let one = digits.parse::<f64>().is_ok_and(|value| value == 1.0);
    format!(" {words} {unit}{} ", if one { "" } else { "s" })
}

fn fraction(numerator: &str, denominator: &str) -> Option<String> {
    let (count, parts): (u64, u64) = (numerator.parse().ok()?, denominator.parse().ok()?);
    let top = number_words(numerator, AND)?;
    Some(match (count, parts) {
        (1, 2) => " one half ".into(),
        (1, 4) => " one quarter ".into(),
        (_, 2) => format!(" {top} halves "),
        (_, 4) => format!(" {top} quarters "),
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
