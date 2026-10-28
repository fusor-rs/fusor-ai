//! A date as ISO 8601, its day order settled by the state.

use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::LazyLock;

use fancy_regex::{Captures, Regex};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

use bundle::python;

use super::{alternation, captures, casefold, every, group, matches, one, pattern, removed, whole};
use crate::Error;

static MONTHS: LazyLock<HashMap<String, u32>> = LazyLock::new(months);

/// Month names a date is written with (accents folded; Polish and Russian in the genitive a date
/// uses).
const MONTH_NAMES: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];
/// Abbreviations the three-letter rule misses.
const MONTH_ABBREVIATIONS: [(&str, u32); 4] = [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)];
/// The weekdays a date may lead with.
const WEEKDAYS: &str = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed \
    thu thur thurs fri sat sun lundi mardi mercredi jeudi vendredi samedi dimanche montag \
    dienstag mittwoch donnerstag freitag samstag sonntag lunes martes miercoles jueves viernes \
    sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca quarta \
    quinta sexta";
const ABBREVIATION_CHARS: usize = 3;

/// Two numbers of a date settle its order only when exactly one of them is over this.
const MONTHS_IN_YEAR: u32 = 12;
const LAST_DAY: u32 = 31;
const YEAR_DIGITS: usize = 4;
const SHORT_YEAR_DIGITS: usize = 2;
const YEARS_IN_CENTURY: u32 = 100;

patterns! {
    CJK_DATE = whole(r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]");
    NUMERIC_DATE = pattern(r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)");
    ONLY_NUMERIC_DATE = whole(r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)");
    WORD_DATE = pattern(&word_date());
    DAY_FIRST = pattern(&format!(r"\b{DAY}\s*[/.-]\s*{MONTH}\s*[/.-]\s*{YEAR}\b"));
    MONTH_FIRST = pattern(&format!(r"\b{MONTH}\s*[/.-]\s*{DAY}\s*[/.-]\s*{YEAR}\b"));
    DATE_LOCALE = pattern(concat!(
        r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?"#,
        r"(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b",
    ));
    TIMED = whole(r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)");
    CLOCK = whole(r"(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(?:(Z)|([+-])(\d{2}):(\d{2}))?");
    ORDINAL = pattern(r"(?<=\d)(?:st|nd|rd|th|er)\b");
    GENDERED_ORDINAL = pattern(r"(?<=\d)\.?(?:o|a|°)\b");
    ARTICLE = pattern(r"\b(?:de|of|the|le|el|il|den)\b");
    PUNCTUATION = pattern(r"[,.]");
}

/// A declared numeric date format's parts, in English, French (jj/mm/aaaa), German
/// (tt.mm.jjjj), Spanish or Italian (gg/mm/aaaa).
const DAY: &str = r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
const MONTH: &str = r"(?:mm?|month|mois|monat|mes|mese)";
const YEAR: &str = r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";

fn months() -> HashMap<String, u32> {
    let mut months = HashMap::new();
    for (month, names) in (1..).zip(MONTH_NAMES) {
        for name in names.split(' ') {
            months.insert(name.to_owned(), month);
        }
    }
    let mut abbreviated: HashMap<String, BTreeSet<u32>> = HashMap::new();
    for (name, &month) in &months {
        let abbreviation: String = name.chars().take(ABBREVIATION_CHARS).collect();
        abbreviated.entry(abbreviation).or_default().insert(month);
    }
    for (abbreviation, candidates) in abbreviated {
        if let Some(month) = one(&candidates) {
            months.insert(abbreviation, month);
        }
    }
    for (abbreviation, month) in MONTH_ABBREVIATIONS {
        months.insert(abbreviation.to_owned(), month);
    }
    months
}

/// A date written in words, in folded text: 20 octobre 2026, October 20th, 2026.
fn word_date() -> String {
    let month = alternation(MONTHS.keys().map(String::as_str));
    format!(
        concat!(
            r"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{month})[ ,.]+(?:de\s+)?\d{{4}}",
            r"|(?:{month})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b",
        ),
        month = month
    )
}

/// `text` case-folded, compatibility-decomposed and without its combining marks.
fn fold(text: &str) -> String {
    casefold(text)
        .nfkd()
        .filter(|&character| canonical_combining_class(character) == 0)
        .collect()
}

/// Which of a numeric date's first two numbers is its day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Order {
    DayFirst,
    MonthFirst,
}

/// The day order the state declares (a format, then a locale), else the one its other
/// unambiguous numeric dates share; conflicting evidence leaves it unsettled.
fn date_order(context: &str, folded: &str) -> Result<Option<Order>, Error> {
    let mut declared = BTreeSet::new();
    for (pattern, order) in [
        (&DAY_FIRST, Order::DayFirst),
        (&MONTH_FIRST, Order::MonthFirst),
    ] {
        if matches(pattern, folded)? {
            declared.insert(order);
        }
    }
    if !declared.is_empty() {
        return Ok(one(&declared));
    }
    let locales: BTreeSet<Order> = every(&DATE_LOCALE, folded)?
        .iter()
        .map(|found| match group(found, 1).replace('_', "-").as_str() {
            "en" | "en-us" => Order::MonthFirst,
            _ => Order::DayFirst,
        })
        .collect();
    if !locales.is_empty() {
        return Ok(one(&locales));
    }
    let mut orders = BTreeSet::new();
    for found in every(&NUMERIC_DATE, context)? {
        let (Some(first), Some(second), Some(year)) = (
            number(group(&found, 1)),
            number(group(&found, 3)),
            number(group(&found, 4)),
        ) else {
            continue;
        };
        let complete = group(&found, 4).len() == YEAR_DIGITS;
        if !complete || !(1..=LAST_DAY).contains(&first) || !(1..=LAST_DAY).contains(&second) {
            continue;
        }
        if (first > MONTHS_IN_YEAR) == (second > MONTHS_IN_YEAR) {
            continue;
        }
        let (day, month, order) = if first > MONTHS_IN_YEAR {
            (first, second, Order::DayFirst)
        } else {
            (second, first, Order::MonthFirst)
        };
        if iso_date(year, month, day).is_some() {
            orders.insert(order);
        }
    }
    Ok(one(&orders))
}

fn number(digits: &str) -> Option<u32> {
    // Digits other than ASCII's settle no date here.
    digits.parse().ok()
}

/// Year, month and day as ISO 8601, when they name a day of the proleptic Gregorian calendar.
fn iso_date(year: u32, month: u32, day: u32) -> Option<String> {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1..=12 => 31,
        _ => return None,
    };
    ((1..=9999).contains(&year) && (1..=days).contains(&day))
        .then(|| format!("{year:04}-{month:02}-{day:02}"))
}

/// A date read: its ISO value, or none and whether an ambiguity left it unsettled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dated {
    Date(String),
    Ambiguous,
    Unread,
}

/// The state a date is read in: its text, and its folded text and day order once a date needs
/// them.
struct DateContext<'a> {
    text: &'a str,
    folded: OnceCell<String>,
    order: OnceCell<Option<Order>>,
}

impl DateContext<'_> {
    fn folded(&self) -> &str {
        self.folded.get_or_init(|| fold(self.text))
    }

    fn order(&self) -> Result<Option<Order>, Error> {
        if let Some(&order) = self.order.get() {
            return Ok(order);
        }
        let order = date_order(self.text, self.folded())?;
        Ok(*self.order.get_or_init(|| order))
    }
}

/// The date `text` writes; `context` is the state the read is of.
///
/// # Errors
/// When a pattern gives up on the text.
pub fn parse_date(text: &str, context: &str) -> Result<Dated, Error> {
    let context = DateContext {
        text: context,
        folded: OnceCell::new(),
        order: OnceCell::new(),
    };
    dated(text, &context)
}

fn dated(text: &str, context: &DateContext) -> Result<Dated, Error> {
    let text = python::strip(text);
    if text.is_empty() {
        return Ok(Dated::Unread);
    }
    if let Some(timed) = captures(&TIMED, text)? {
        return match dated(group(&timed, 1), context)? {
            Dated::Date(date) => {
                Ok(clocked(&date, group(&timed, 2))?.map_or(Dated::Unread, Dated::Date))
            }
            unsettled => Ok(unsettled),
        };
    }
    if let Some(numeric) = captures(&ONLY_NUMERIC_DATE, text)? {
        return numeric_date(&numeric, context);
    }
    if let Some(cjk) = captures(&CJK_DATE, text)? {
        let date = match (
            number(group(&cjk, 1)),
            number(group(&cjk, 2)),
            number(group(&cjk, 3)),
        ) {
            (Some(year), Some(month), Some(day)) => iso_date(year, month, day),
            _ => None,
        };
        return Ok(date.map_or(Dated::Unread, Dated::Date));
    }
    worded_date(text)
}

/// A date with an explicitly written time and timezone, as Python's `datetime.isoformat` writes
/// it: no default time or timezone.
fn clocked(date: &str, clock: &str) -> Result<Option<String>, Error> {
    let clock = if clock.chars().nth(1) == Some(':') {
        format!("0{clock}")
    } else {
        clock.to_owned()
    };
    let Some(parts) = captures(&CLOCK, &clock)? else {
        return Ok(None);
    };
    let field = |index| number(group(&parts, index)).unwrap_or(0);
    let (hour, minute, second) = (field(1), field(2), field(3));
    if date.len() != "yyyy-mm-dd".len() || hour >= 24 || minute >= 60 || second >= 60 {
        return Ok(None);
    }
    let mut written = format!("{date}T{hour:02}:{minute:02}:{second:02}");
    let fraction: String = group(&parts, 4)
        .chars()
        .chain(std::iter::repeat('0'))
        .take(6)
        .collect();
    if fraction != "000000" {
        written.push('.');
        written.push_str(&fraction);
    }
    if parts.get(5).is_some() {
        written.push_str("+00:00");
    } else if parts.get(6).is_some() {
        let Some(offset) = offset(group(&parts, 6), field(7), field(8)) else {
            return Ok(None);
        };
        written.push_str(&offset);
    }
    Ok(Some(written))
}

/// A timezone offset as `datetime` writes it; none when it is a day or more.
fn offset(sign: &str, hours: u32, minutes: u32) -> Option<String> {
    let total = hours * 60 + minutes;
    if total >= 24 * 60 {
        return None;
    }
    let sign = if sign == "-" && total > 0 { "-" } else { "+" };
    Some(format!("{sign}{:02}:{:02}", total / 60, total % 60))
}

/// A numeric date: year first, year last, or a two-digit year whose century another
/// unambiguous complete date of the state establishes.
fn numeric_date(numeric: &Captures, context: &DateContext) -> Result<Dated, Error> {
    let (Some(first), Some(second), Some(last)) = (
        number(group(numeric, 1)),
        number(group(numeric, 3)),
        number(group(numeric, 4)),
    ) else {
        return Ok(Dated::Unread);
    };
    let candidates = if group(numeric, 1).len() == YEAR_DIGITS {
        vec![(first, second, last)]
    } else if group(numeric, 4).len() == YEAR_DIGITS {
        candidates(last, first, second, context.order()?)
    } else {
        if group(numeric, 4).len() != SHORT_YEAR_DIGITS {
            return Ok(Dated::Unread);
        }
        let Some(century) = century(context)? else {
            return Ok(Dated::Unread);
        };
        candidates(century + last, first, second, context.order()?)
    };
    let values: BTreeSet<String> = candidates
        .into_iter()
        .filter_map(|(year, month, day)| iso_date(year, month, day))
        .collect();
    Ok(match values.len() {
        0 => Dated::Unread,
        1 => Dated::Date(values.into_iter().next().expect("one value")),
        _ => Dated::Ambiguous,
    })
}

/// (year, month, day) for `first`/`second`/`year`: day first, month first, or the one `order`
/// settles.
fn candidates(year: u32, first: u32, second: u32, order: Option<Order>) -> Vec<(u32, u32, u32)> {
    let (day_first, month_first) = ((year, second, first), (year, first, second));
    match order {
        None => vec![day_first, month_first],
        Some(Order::DayFirst) => vec![day_first],
        Some(Order::MonthFirst) => vec![month_first],
    }
}

/// The one century the state's unambiguous complete dates are of.
fn century(context: &DateContext) -> Result<Option<u32>, Error> {
    let mut complete: Vec<String> = every(&NUMERIC_DATE, context.text)?
        .iter()
        .filter(|found| {
            group(found, 1).len() == YEAR_DIGITS || group(found, 4).len() == YEAR_DIGITS
        })
        .map(|found| group(found, 0).to_owned())
        .collect();
    complete.extend(
        every(&WORD_DATE, context.folded())?
            .iter()
            .map(|found| group(found, 0).to_owned()),
    );
    let mut centuries = BTreeSet::new();
    for written in complete {
        if let Dated::Date(date) = dated(&written, context)?
            && let Some(year) = number(&date[..YEAR_DIGITS])
        {
            centuries.insert(year / YEARS_IN_CENTURY * YEARS_IN_CENTURY);
        }
    }
    Ok(one(&centuries))
}

/// A date in words: a day, a month's name and a year in either order, after a weekday or not.
fn worded_date(text: &str) -> Result<Dated, Error> {
    let words = removed(&ORDINAL, &fold(text), "");
    let words = removed(&GENDERED_ORDINAL, &words, "");
    let words = removed(&ARTICLE, &words, " ");
    let words = removed(&PUNCTUATION, &words, " ");
    let mut words: Vec<&str> = words
        .split(python::isspace)
        .filter(|word| !word.is_empty())
        .collect();
    let weekdays: Vec<&str> = WEEKDAYS.split_whitespace().collect();
    if words
        .first()
        .is_some_and(|first| weekdays.contains(&first.strip_suffix("-feira").unwrap_or(first)))
    {
        words.remove(0);
    }
    let [first, second, year] = words.as_slice() else {
        return Ok(Dated::Unread);
    };
    let (day, month) = match (MONTHS.get(*first), MONTHS.get(*second)) {
        (Some(&month), _) => (second, month),
        (None, Some(&month)) => (first, month),
        (None, None) => return Ok(Dated::Unread),
    };
    let ascii_digits =
        |word: &str| !word.is_empty() && word.chars().all(|character| character.is_ascii_digit());
    if !ascii_digits(day) || !ascii_digits(year) || year.len() != YEAR_DIGITS {
        return Ok(Dated::Unread);
    }
    let date = number(day)
        .zip(number(year))
        .and_then(|(day, year)| iso_date(year, month, day));
    Ok(date.map_or(Dated::Unread, Dated::Date))
}

#[cfg(test)]
mod tests {
    use super::{Dated, parse_date};

    #[test]
    fn reads_dates_as_the_reference_does() {
        let date = |iso: &str| Dated::Date(iso.into());
        let cases = [
            (
                "03/04/2026",
                "Pay before 03/04/2026 (day/month/year)",
                date("2026-04-03"),
            ),
            ("03/04/2026", "", Dated::Ambiguous),
            ("13/04/2026", "", date("2026-04-13")),
            ("2026-04-03", "", date("2026-04-03")),
            ("03/04/26", "Issued 2026-01-05", Dated::Ambiguous),
            ("20 octobre 2026", "", date("2026-10-20")),
            ("Tue, 20 Oct 2026", "", date("2026-10-20")),
            ("October 20th, 2026", "", date("2026-10-20")),
            ("2026年10月20日", "", date("2026-10-20")),
            ("2026-04-03 9:30", "", date("2026-04-03T09:30:00")),
            (
                "2026-04-03T09:30:00.5Z",
                "",
                date("2026-04-03T09:30:00.500000+00:00"),
            ),
            (
                "2026-04-03 10:00+05:30",
                "",
                date("2026-04-03T10:00:00+05:30"),
            ),
            ("31/02/2026", "", Dated::Unread),
            ("04/03/2026", "lang: en", date("2026-04-03")),
            (
                "segunda-feira, 20 de outubro de 2026",
                "",
                date("2026-10-20"),
            ),
            ("3. März 2026", "", date("2026-03-03")),
            ("05/06/07", "on 20 octobre 2026", Dated::Ambiguous),
        ];
        for (text, context, expected) in cases {
            assert_eq!(parse_date(text, context).unwrap(), expected, "{text}");
        }
    }
}
