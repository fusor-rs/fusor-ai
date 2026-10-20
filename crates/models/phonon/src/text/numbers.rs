//! Numbers, money and digit codes.
//!
//! The model reads digits: quantities keep them and only gain scale words ("48213" is "48
//! thousand 213"); years and values under a thousand stay as written; codes (a leading zero, a
//! phone number in groups, "ending in -4098") are spelled digit by digit.

use super::{PLUS, all_digits, split_suffix, trailing_punctuation};

const DIGIT_NAMES: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];
const SCALES: [(u64, &str); 3] = [
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1000, "thousand"),
];
/// Values from here on stay as written: no scale word reads them.
const UNREAD: u64 = 1_000_000_000_000;
const YEARS: std::ops::Range<u64> = 1900..2100;
const MINUS: &str = "minus";
const POINT: &str = "point";

struct Currency {
    symbol: char,
    one: &'static str,
    many: &'static str,
}

const CURRENCIES: [Currency; 5] = [
    Currency {
        symbol: '$',
        one: "dollar",
        many: "dollars",
    },
    Currency {
        symbol: '€',
        one: "euro",
        many: "euros",
    },
    Currency {
        symbol: '£',
        one: "pound",
        many: "pounds",
    },
    Currency {
        symbol: '¥',
        one: "yen",
        many: "yen",
    },
    Currency {
        symbol: '¢',
        one: "cent",
        many: "cents",
    },
];
const CENTS: &str = "cents";
/// Scale words that may follow an amount, and the shorthands written right after one ("€3bn").
const SCALE_WORDS: [&str; 4] = ["thousand", "million", "billion", "trillion"];
const SCALE_SHORTHANDS: [(&str, &str); 14] = [
    ("thousand", "thousand"),
    ("million", "million"),
    ("billion", "billion"),
    ("trillion", "trillion"),
    ("bn", "billion"),
    ("BN", "billion"),
    ("mn", "million"),
    ("tn", "trillion"),
    ("k", "thousand"),
    ("K", "thousand"),
    ("m", "million"),
    ("M", "million"),
    ("b", "billion"),
    ("B", "billion"),
];

/// The longest group of a phone number or code written in spaced groups.
const MAX_GROUP: usize = 8;
/// Groups of a dashed code up to this long are read as numbers, longer ones digit by digit.
const MAX_DASHED_NUMBER: usize = 3;
const UNDERSCORE: &str = "underscore";

/// A number word: a code with a leading zero, a range ("10-15"), a quantity, or a quantity
/// joined to a word ("1500-word").
pub(super) fn numbers(word: &str) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    if body.len() > 1 && body.starts_with('0') && all_digits(body) {
        return Some(spell_digits(body) + suffix);
    }
    if let Some(range) = range(body) {
        return Some(range + suffix);
    }
    if let Some((spoken, suffix)) = quantity(word) {
        return Some(spoken + suffix);
    }
    let (head, tail) = word.split_at(word.find('-')?);
    if !tail[1..].starts_with(char::is_alphabetic) {
        return None;
    }
    match quantity(head)? {
        (spoken, "") => Some(spoken + tail),
        _ => None,
    }
}

fn range(body: &str) -> Option<String> {
    let (from, to) = body.split_once('-')?;
    let small =
        |part: &str| (2..=3).contains(&part.len()) && all_digits(part) && !part.starts_with('0');
    let year = |part: &str| {
        part.len() == 4 && all_digits(part) && (part.starts_with('1') || part.starts_with("20"))
    };
    let ascending = match (from.len(), to.len()) {
        _ if small(from) && small(to) => (from.len(), from) < (to.len(), to),
        (4, 4) => year(from) && year(to) && from < to,
        (4, 2) => year(from) && all_digits(to) && &from[2..] < to,
        _ => false,
    };
    ascending.then(|| format!("{from} to {to}"))
}

/// A signed, comma-grouped number with optional decimals, read with scale words, and the
/// sentence punctuation after it.
fn quantity(word: &str) -> Option<(String, &str)> {
    let (sign, unsigned) = match word.strip_prefix('-') {
        Some(rest) => (Some(MINUS), rest),
        None => (None, word),
    };
    let grouped = unsigned
        .bytes()
        .take_while(|&byte| byte.is_ascii_digit() || byte == b',')
        .count();
    let integer = unsigned[..grouped].trim_end_matches(',');
    if !integer.starts_with(|ch: char| ch.is_ascii_digit())
        || integer.len() > 1 && integer.starts_with('0')
    {
        return None;
    }
    let (decimals, rest) = decimals(&unsigned[integer.len()..]);
    let suffix = trailing_punctuation(rest)?;
    let digits: String = integer.chars().filter(char::is_ascii_digit).collect();
    // Digits past u64 are past `UNREAD` too: the number stays as written.
    let value: u64 = digits.parse().ok()?;
    if value >= UNREAD {
        return None;
    }
    let mut spoken: Vec<String> = sign.into_iter().map(str::to_string).collect();
    spoken.push(say(value));
    if let Some(decimals) = decimals {
        spoken.extend([POINT.to_string(), decimals.to_string()]);
    }
    Some((spoken.join(" "), suffix))
}

fn decimals(rest: &str) -> (Option<&str>, &str) {
    let Some(after) = rest.strip_prefix('.') else {
        return (None, rest);
    };
    match after.bytes().take_while(u8::is_ascii_digit).count() {
        0 => (None, rest),
        len => (Some(&after[..len]), &after[len..]),
    }
}

fn say(value: u64) -> String {
    if value < 1000 || YEARS.contains(&value) {
        return value.to_string();
    }
    let groups = SCALES.iter().filter_map(|&(scale, name)| {
        let count = value / scale % 1000;
        (count > 0).then(|| format!("{count} {name}"))
    });
    let mut parts: Vec<String> = groups.collect();
    if !value.is_multiple_of(1000) {
        parts.push((value % 1000).to_string());
    }
    parts.join(" ")
}

/// Each digit as its word, other characters as they are, dashes dropped.
fn spell_digits(code: &str) -> String {
    let spoken = code.chars().filter(|&ch| ch != '-').map(|ch| {
        ch.to_digit(10).map_or_else(
            || ch.to_string(),
            |digit| DIGIT_NAMES[digit as usize].to_string(),
        )
    });
    spoken.collect::<Vec<_>>().join(" ")
}

/// Spaced groups of digits read as a code, digit by digit and group by group: three groups or
/// more (not all years), or two after a `+` or with a leading zero ("07700 900123").
pub(super) fn digit_groups(words: &[&str]) -> Option<(String, usize)> {
    let plus = words.first()?.starts_with('+');
    let mut groups = Vec::new();
    let mut end = "";
    for (index, word) in words.iter().enumerate() {
        let (body, suffix) = split_suffix(word);
        let body = if index == 0 && plus { &body[1..] } else { body };
        if !is_digit_group(body) {
            break;
        }
        groups.push(body);
        if !suffix.is_empty() {
            end = suffix;
            break;
        }
    }
    let years = groups
        .iter()
        .all(|group| group.len() == 4 && group.parse().is_ok_and(|year| YEARS.contains(&year)));
    let code = match groups.len() {
        2 => plus || groups.iter().any(|group| group.starts_with('0')),
        count => count > 2 && !years,
    };
    if !code {
        return None;
    }
    let spelled: Vec<String> = groups.iter().map(|group| spell_digits(group)).collect();
    let lead = if plus {
        format!("{PLUS} ")
    } else {
        String::new()
    };
    Some((format!("{lead}{}{end}", spelled.join(", ")), groups.len()))
}

fn is_digit_group(body: &str) -> bool {
    (1..=MAX_GROUP).contains(&body.len())
        && body
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte.is_ascii_uppercase())
        && body.bytes().any(|byte| byte.is_ascii_digit())
}

/// "ending in -4098": the code after "ending in" spelled out.
pub(super) fn ending_in(previous: &str, words: &[&str]) -> Option<(String, usize)> {
    let ending = ["ending", "ends", "ended"];
    if !ending.contains(&previous.to_lowercase().as_str()) || *words.first()? != "in" {
        return None;
    }
    let (body, end) = split_suffix(words.get(1)?);
    let code = body.strip_prefix('-')?;
    let readable = |byte: u8| byte.is_ascii_digit() || byte.is_ascii_uppercase() || byte == b'-';
    if !code.bytes().any(|byte| byte.is_ascii_digit()) || !code.bytes().all(readable) {
        return None;
    }
    Some((format!("in {}{end}", spell_digits(code)), 2))
}

/// Digits with two dashes or underscores or more ("555-123-4567"), except dashed dates: long
/// groups digit by digit, underscores spoken.
pub(super) fn dashed_digits(word: &str) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    let separator = |ch: char| ch == '-' || ch == '_';
    let groups: Vec<&str> = body.split(separator).collect();
    let well_formed = groups.len() > 2 && groups.iter().all(|group| all_digits(group));
    if !well_formed || !body.contains('_') && is_date(&groups) {
        return None;
    }
    let separators = body.matches(separator).map(Some).chain([None]);
    let mut words = Vec::new();
    for (group, separator) in groups.iter().zip(separators) {
        if group.len() > MAX_DASHED_NUMBER {
            words.extend((0..group.len()).map(|at| &group[at..=at]));
        } else {
            words.push(group);
        }
        if separator == Some("_") {
            words.push(UNDERSCORE);
        }
    }
    Some(words.join(" ") + suffix)
}

/// `YYYY-MM-DD`, or day and month in either order before a 2- or 4-digit year.
fn is_date(groups: &[&str]) -> bool {
    let day_month = |first: &str, second: &str| {
        let (Ok(first), Ok(second)) = (first.parse::<u32>(), second.parse::<u32>()) else {
            return false;
        };
        let (days, months) = (1..=31, 1..=12);
        days.contains(&first) && months.contains(&second)
            || months.contains(&first) && days.contains(&second)
    };
    let short = |group: &str| group.len() <= 2;
    match groups {
        [year, month, day] if year.len() == 4 => {
            short(month) && short(day) && day_month(month, day)
        }
        [first, second, year] if matches!(year.len(), 2 | 4) => {
            short(first) && short(second) && day_month(first, second)
        }
        _ => false,
    }
}

/// An amount with a currency symbol before or after it: "$1,234.56", "5$", "-$5", "£3.2m".
pub(super) fn money(word: &str) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    let (sign, unsigned) = match body.strip_prefix('-') {
        Some(rest)
            if CURRENCIES
                .iter()
                .any(|currency| rest.starts_with(currency.symbol)) =>
        {
            (format!("{MINUS} "), rest)
        }
        _ => (String::new(), body),
    };
    let (currency, amount) = split_currency(unsigned)?;
    let spoken = currency
        .scaled(amount)
        .or_else(|| currency.with_cents(amount))
        .or_else(|| currency.whole(amount))?;
    Some(format!("{sign}{spoken}{suffix}"))
}

/// "$2 million": an amount, then a scale word.
pub(super) fn scaled_money(words: &[&str]) -> Option<(String, usize)> {
    let [word, next, ..] = words else {
        return None;
    };
    let (scale, suffix) = split_suffix(next);
    if !SCALE_WORDS.contains(&scale) {
        return None;
    }
    let (currency, amount) = split_currency(word)?;
    let spoken = currency.with_scale(amount, scale)?;
    Some((spoken + suffix, 2))
}

fn split_currency(word: &str) -> Option<(&'static Currency, &str)> {
    CURRENCIES.iter().find_map(|currency| {
        let amount = word.strip_prefix(currency.symbol);
        let amount = amount.or_else(|| word.strip_suffix(currency.symbol))?;
        Some((currency, amount))
    })
}

impl Currency {
    fn scaled(&self, amount: &str) -> Option<String> {
        SCALE_SHORTHANDS.iter().find_map(|&(shorthand, scale)| {
            let number = amount.strip_suffix(shorthand)?;
            number
                .ends_with(|ch: char| ch.is_ascii_digit())
                .then(|| self.with_scale(number, scale))?
        })
    }

    fn with_scale(&self, amount: &str, scale: &str) -> Option<String> {
        match quantity(amount)? {
            (spoken, "") => Some(format!("{spoken} {scale} {}", self.many)),
            _ => None,
        }
    }

    /// Exactly two decimals are cents: "$4.59" is "4 dollars and 59 cents". Not for the cent.
    fn with_cents(&self, amount: &str) -> Option<String> {
        let (whole, cents) = amount.rsplit_once('.')?;
        let grouped = whole.split(',').enumerate().all(|(index, group)| {
            all_digits(group) && (index == 0 && group.len() <= 3 || group.len() == 3)
        });
        let plain_cents = cents.len() == 2 && all_digits(cents);
        if self.symbol == '¢' || !plain_cents || !(all_digits(whole) || grouped) {
            return None;
        }
        let (whole, _) = quantity(whole)?;
        let name = if whole == "1" { self.one } else { self.many };
        let cents = cents.strip_prefix('0').unwrap_or(cents);
        Some(match (whole.as_str(), cents) {
            (_, "0") => format!("{whole} {name}"),
            ("0", _) => format!("{cents} {CENTS}"),
            _ => format!("{whole} {name} and {cents} {CENTS}"),
        })
    }

    /// Always the plural ("1 dollars", as the training text reads), except for one cent.
    fn whole(&self, amount: &str) -> Option<String> {
        let (spoken, _) = quantity(amount)?;
        let one_cent = self.symbol == '¢' && amount == "1";
        let name = if one_cent { self.one } else { self.many };
        Some(format!("{spoken} {name}"))
    }
}
