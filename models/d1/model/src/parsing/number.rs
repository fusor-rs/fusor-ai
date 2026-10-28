//! A number's magnitude, its currency and percent sign set apart and its scale word applied, its
//! decimal mark settled by the read and the state.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use fancy_regex::Regex;
use serde_json::Number;

use bundle::python;

use super::{alternation, captures, casefold, every, group, matches, one, pattern, removed, whole};
use crate::Error;

/// A decimal mark: "." or ",", or "" for amounts in whole units.
type Mark = &'static str;

/// Currencies by the decimal mark their amounts are written with, `None` for both in use.
const CURRENCIES: [(Option<Mark>, &str); 4] = [
    (
        Some("."),
        "$ £ ₹ USD GBP CAD AUD NZD INR Rs. Rs MXN CNY RMB 元 HKD SGD",
    ),
    (Some(","), "R$ BRL ₽ RUB zł PLN kr SEK NOK DKK ₺ TRY"),
    (Some(""), "¥ ￥ 円 ₩ ₫ Rp. Rp IDR JPY KRW VND CLP"),
    (None, "€ EUR CHF"),
];

/// Scale words, written apart or attached, by their power of ten. German "Billion" (10^12) is
/// left out: it reads as no value.
const SCALES: [(&str, u32); 8] = [
    ("thousand mil tausend тыс 千", 3),
    ("万 萬", 4),
    ("lakh lakhs", 5),
    ("crore crores", 7),
    (
        "million millions mio mln millón millones milhão milhões milione milioni млн",
        6,
    ),
    ("亿 億", 8),
    (
        "billion billions milliard milliards milliarde milliarden mrd miliardo miliardi млрд",
        9,
    ),
    ("trillion", 12),
];
const AMBIGUOUS_SCALE: &str = "Billion";
/// Attached k, M, B: a magnitude only beside money or a count.
const SUFFIXES: [(&str, u32); 4] = [("k", 3), ("m", 6), ("b", 9), ("bn", 9)];

static CURRENCY: LazyLock<String> = LazyLock::new(|| {
    let mut names: Vec<&str> = currencies().map(|(_, name)| name).collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.chars().count()));
    let alternatives: Vec<String> = names
        .iter()
        .map(|name| {
            let escaped = fancy_regex::escape(name);
            let first = name.chars().next().expect("a currency is named");
            if first.is_ascii_alphabetic() {
                format!(r"\b{escaped}(?![a-z])")
            } else {
                escaped.into_owned()
            }
        })
        .collect();
    format!("(?i:{})", alternatives.join("|"))
});

patterns! {
    ANY_CURRENCY = pattern(&CURRENCY);
    ONLY_CURRENCY = whole(&CURRENCY);
    CURRENCY_SPLIT = whole(&format!(r"({0})\s*(.*)|(.*?)\s*({0})", *CURRENCY));
    SCALE = pattern(&format!(r"(?i)\s*({})\.?$", alternation(scales().map(|(word, _)| word))));
    SUFFIX = pattern(r"(?i)(?<=\d)(k|m|bn|b)$");
    COUNTED = pattern(concat!(
        r"(?i)\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|",
        r"users|downloads)\b",
    ));
    GROUPED = pattern(concat!(
        r"(?<![\d.,])\d{1,3}(?:([,. \x{a0}\x{202f}'’])\d{3})+",
        r"(?!\1)([.,])\d+(?![\d.,])",
    ));
    SHORT = pattern(r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])");
    REPEATED = pattern(r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])");
    NUMBER_LOCALE = pattern(concat!(
        r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?"#,
        r"(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b",
    ));
    MARKUP_DECLARATION = pattern(r"<\?[^>]*\?>");
    DIGITS = whole(r"\d[\d., '’]*");
    GROUP_SPACE = pattern(r"[ '’]");
    SPACE_GROUPED = whole(r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?");
    LAKH_GROUPED = whole(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?");
    RUPEE = pattern(r"(?i)₹|\bINR\b|\bRs\b|lakh|crore");
    POINT_GROUPED = whole(r"\d{1,3}(?:,\d{3})+\.\d+");
    COMMA_GROUPED = whole(r"\d{1,3}(?:\.\d{3})+,\d+");
}

fn currencies() -> impl Iterator<Item = (Option<Mark>, &'static str)> {
    CURRENCIES
        .iter()
        .flat_map(|&(mark, names)| names.split(' ').map(move |name| (mark, name)))
}

fn scales() -> impl Iterator<Item = (&'static str, u32)> {
    SCALES
        .iter()
        .flat_map(|&(words, exponent)| words.split(' ').map(move |word| (word, exponent)))
}

/// The currency `written` names, case aside: its decimal mark and its canonical name.
fn currency(written: &str) -> (Option<Mark>, &'static str) {
    let folded = written.to_lowercase();
    currencies()
        .find(|(_, name)| name.to_lowercase() == folded)
        .expect("a currency pattern's match names a currency")
}

/// Only an explicitly written percent or currency marker; a currency sign stays a sign.
///
/// # Errors
/// When a pattern gives up on `text`.
pub fn written_unit(text: &str) -> Result<Option<&'static str>, Error> {
    if text.trim_end_matches(python::isspace).ends_with('%') {
        return Ok(Some("%"));
    }
    let markers = every(&ANY_CURRENCY, text)?;
    Ok(match markers.as_slice() {
        [marker] => Some(currency(group(marker, 0)).1),
        _ => None,
    })
}

/// The decimal marks of the currencies `text` writes; a currency of both says nothing.
fn currency_marks(text: &str) -> Result<BTreeSet<Mark>, Error> {
    let markers = every(&ANY_CURRENCY, text)?;
    Ok(markers
        .iter()
        .filter_map(|marker| currency(group(marker, 0)).0)
        .collect())
}

/// The decimal mark from the first evidence that speaks: the read's own currency, the state's
/// other numbers, the state's currencies, a declared locale. A markup declaration's numbers
/// (`<?xml version="1.0"?>`) are not the state's.
fn decimal_mark(written: &str, context: &str) -> Result<Option<Mark>, Error> {
    let context = removed(&MARKUP_DECLARATION, context, "");
    let mark = |text: &str| if text == "." { "." } else { "," };
    let mut numbers: BTreeSet<Mark> = every(&GROUPED, &context)?
        .iter()
        .map(|found| mark(group(found, 2)))
        .collect();
    numbers.extend(
        every(&SHORT, &context)?
            .iter()
            .map(|found| mark(group(found, 1))),
    );
    numbers.extend(
        every(&REPEATED, &context)?
            .iter()
            .map(|found| if group(found, 1) == "." { "," } else { "." }),
    );
    let locales: BTreeSet<Mark> = every(&NUMBER_LOCALE, &casefold(&context))?
        .iter()
        .map(|found| {
            if matches!(group(found, 1), "en" | "ja" | "zh" | "ko") {
                "."
            } else {
                ","
            }
        })
        .collect();
    let evidence = [
        currency_marks(written)?,
        numbers,
        currency_marks(&context)?,
        locales,
    ];
    Ok(evidence
        .iter()
        .find(|marks| !marks.is_empty())
        .and_then(one))
}

/// The magnitude `text` writes, percent points included; none when the writing does not settle
/// one value. `context` is the state the read is of, `unit` the question's.
///
/// # Errors
/// When a pattern gives up on the text.
pub fn parse_number(
    text: &str,
    context: &str,
    unit: Option<&str>,
) -> Result<Option<Number>, Error> {
    let mut context = context.to_owned();
    if let Some(unit) = unit
        && matches(&ONLY_CURRENCY, python::strip(unit))?
    {
        context = format!("{context}\n{unit}");
    }
    let cleaned = python::strip(text)
        .replace('\u{2212}', "-")
        .replace(['\u{a0}', '\u{202f}'], " ");
    let cleaned = cleaned
        .strip_suffix('%')
        .unwrap_or(&cleaned)
        .trim_end_matches(python::isspace);
    let bracketed = cleaned
        .strip_prefix('(')
        .and_then(|inner| inner.strip_suffix(')'));
    let amount = amount(bracketed.map_or(cleaned, python::strip))?;
    if bracketed.is_some() && !amount.sign.is_empty() {
        return Ok(None);
    }
    let Some((digits_text, exponent)) = scaled(&amount.text, text, &context)? else {
        return Ok(None);
    };
    let Some(digits) = digits(digits_text, text, &context)? else {
        return Ok(None);
    };
    let sign = if bracketed.is_some() {
        "-"
    } else {
        amount.sign
    };
    Ok(magnitude(sign, &digits, exponent))
}

/// An amount's sign and its text without its currency and sign.
struct Amount {
    sign: &'static str,
    text: String,
}

/// One currency and one sign set apart from `text`, in either order.
fn amount(text: &str) -> Result<Amount, Error> {
    let mut text = text.to_owned();
    let mut currency = false;
    let mut sign = None;
    for _ in 0..2 {
        if !currency && let Some(split) = captures(&CURRENCY_SPLIT, &text)? {
            currency = true;
            let rest = if split.get(1).is_some() {
                group(&split, 2)
            } else {
                group(&split, 3)
            };
            text = python::strip(rest).to_owned();
        }
        if sign.is_none() && (text.starts_with('-') || text.starts_with('+')) {
            sign = Some(if text.starts_with('-') { "-" } else { "+" });
            text = python::strip(&text[1..]).to_owned();
        }
    }
    Ok(Amount {
        sign: sign.unwrap_or(""),
        text,
    })
}

/// The amount without its scale word or suffix, and the power of ten it applies; none when the
/// scale settles no value.
fn scaled<'t>(
    text: &'t str,
    written: &str,
    context: &str,
) -> Result<Option<(&'t str, u32)>, Error> {
    if let Some(word) = captures(&SCALE, text)? {
        let found = group(&word, 1);
        if found == AMBIGUOUS_SCALE {
            return Ok(None);
        }
        let lowered = found.to_lowercase();
        let (_, exponent) = scales()
            .find(|&(scale, _)| scale == lowered)
            .expect("a scale pattern's match is a scale word");
        return Ok(Some((
            &text[..word.get(0).expect("a match").start()],
            exponent,
        )));
    }
    let Some(suffix) = captures(&SUFFIX, text)? else {
        return Ok(Some((text, 0)));
    };
    let money = matches(&ANY_CURRENCY, &format!("{written}\n{context}"))?;
    if !money && !matches(&COUNTED, context)? {
        return Ok(None);
    }
    let lowered = group(&suffix, 1).to_lowercase();
    let (_, exponent) = SUFFIXES
        .iter()
        .find(|(name, _)| *name == lowered)
        .expect("a suffix pattern's match is a suffix");
    Ok(Some((
        &text[..suffix.get(0).expect("a match").start()],
        *exponent,
    )))
}

/// `text`'s number as plain digits and a point: the decimal mark from the evidence, the other
/// mark a complete grouping (threes, or India's lakh grouping beside a rupee).
fn digits(text: &str, written: &str, context: &str) -> Result<Option<String>, Error> {
    if !matches(&DIGITS, text)? {
        return Ok(None);
    }
    if matches(&GROUP_SPACE, text)? {
        let Some(grouping) = captures(&SPACE_GROUPED, text)? else {
            return Ok(None);
        };
        let grouped = removed(&GROUP_SPACE, group(&grouping, 1), "");
        return digits(
            &format!("{grouped}{}", group(&grouping, 2)),
            written,
            context,
        );
    }
    if matches(&LAKH_GROUPED, text)? && matches(&RUPEE, &format!("{written}\n{context}"))? {
        return Ok(Some(text.replace(',', "")));
    }
    if let (Some(point), Some(comma)) = (text.rfind('.'), text.rfind(',')) {
        let (pattern, group) = if point > comma {
            (&POINT_GROUPED, ",")
        } else {
            (&COMMA_GROUPED, ".")
        };
        if !matches(pattern, text)? {
            return Ok(None);
        }
        let mark = if point > comma { "." } else { "," };
        return Ok(Some(text.replace(group, "").replace(mark, ".")));
    }
    let Some(mark) = [".", ","].into_iter().find(|mark| text.contains(mark)) else {
        return Ok(Some(text.to_owned()));
    };
    marked(text, mark, written, context)
}

/// A number written with one kind of mark: a decimal or a grouping, as the evidence settles.
fn marked(text: &str, mark: Mark, written: &str, context: &str) -> Result<Option<String>, Error> {
    let parts: Vec<&str> = text.split(mark).collect();
    if parts.iter().any(|part| part.is_empty()) {
        return Ok(None);
    }
    let point = decimal_mark(written, context)?;
    let lengths: Vec<usize> = parts.iter().map(|part| part.chars().count()).collect();
    if parts.len() > 2 || point.is_some_and(|point| point != mark) {
        let complete =
            (1..=3).contains(&lengths[0]) && lengths[1..].iter().all(|&length| length == 3);
        return Ok(complete.then(|| parts.concat()));
    }
    if point.is_none() && lengths[1] == 3 && lengths[0] <= 3 {
        return Ok(None);
    }
    Ok(Some(parts.join(".")))
}

/// `sign` and `digits` (with a point or not) times ten to `exponent`: whole as an integer, else
/// as the float nearest it.
fn magnitude(sign: &str, digits: &str, exponent: u32) -> Option<Number> {
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    let shift = usize::try_from(exponent).expect("a scale's exponent is small");
    let mut fraction: Vec<char> = fraction.chars().collect();
    fraction.resize(fraction.len().max(shift), '0');
    let moved: String = fraction.drain(..shift).collect();
    let rest: String = fraction.into_iter().collect();
    let whole = format!("{sign}{whole}{moved}");
    let rest = rest.trim_end_matches('0');
    if rest.is_empty()
        && let Ok(integer) = whole.parse::<i64>()
    {
        return Some(integer.into());
    }
    // Digits other than ASCII's settle no value here.
    let value: f64 = format!("{whole}.{rest}")
        .trim_end_matches('.')
        .parse()
        .ok()?;
    Number::from_f64(value)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{parse_number, written_unit};

    const INVOICE: &str = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\n\
        Pay before 03/04/2026 (day/month/year)";

    #[test]
    fn reads_numbers_as_the_reference_does() {
        let cases = [
            (
                "1 234,50 EUR",
                INVOICE,
                Some("EUR"),
                json!(1234.5),
                Some("EUR"),
            ),
            ("$1,250", "", None, json!(1250), Some("$")),
            ("1,250", "", None, Value::Null, None),
            ("1.234.567", "", None, json!(1_234_567), None),
            ("12,50", "", None, json!(12.5), None),
            ("(45.00)", "", None, json!(-45), None),
            ("-3.5%", "", None, json!(-3.5), Some("%")),
            ("2.5 million", "", None, json!(2_500_000), None),
            ("5k", "the price list", None, json!(5000), None),
            ("5k", "", None, Value::Null, None),
            ("1,00,000", "₹ in lakh", None, json!(100_000), None),
            ("3 Billion", "", None, Value::Null, None),
            ("¥1,200", "", None, json!(1200), Some("¥")),
            ("1.250", "Total 3,50 EUR", None, json!(1250), None),
            ("USD 1,234.56", "", None, json!(1234.56), Some("USD")),
            ("1'234.5", "", None, json!(1234.5), None),
            ("+7", "", None, json!(7), None),
            ("0.10", "", None, json!(0.1), None),
            ("12 千", "", None, json!(12000), None),
        ];
        for (text, context, unit, number, written) in cases {
            let parsed: Value = parse_number(text, context, unit).unwrap().into();
            assert_eq!(
                (parsed, written_unit(text).unwrap()),
                (number, written),
                "{text}"
            );
        }
    }
}
