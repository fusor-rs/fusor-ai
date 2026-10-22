//! The word pass: every space-separated word, or a run of them, to the first rule that claims it.

use super::characters::Sink;
use super::{AT, PLUS, addresses, numbers, split_suffix};

/// Rules over one word, in the order they are tried.
const RULES: [fn(&str) -> Option<String>; 7] = [
    numbers::numbers,
    numbers::money,
    numbers::dashed_digits,
    addresses::email,
    addresses::url,
    abbreviation,
    elongation,
];

/// Abbreviations the model hears as their words. A form with a period only matches with it.
const ABBREVIATIONS: [(&str, &str); 54] = [
    ("Lt", "Lieutenant"),
    ("Lt.", "Lieutenant"),
    ("Col.", "Colonel"),
    ("Sgt", "Sergeant"),
    ("Sgt.", "Sergeant"),
    ("Capt.", "Captain"),
    ("Cpl.", "Corporal"),
    ("Gen.", "General"),
    ("Adm.", "Admiral"),
    ("Sen.", "Senator"),
    ("Sens.", "Senators"),
    ("Rep.", "Representative"),
    ("Reps.", "Representatives"),
    ("Gov.", "Governor"),
    ("Pres.", "President"),
    ("Det.", "Detective"),
    ("Insp.", "Inspector"),
    ("Supt.", "Superintendent"),
    ("Ft.", "Fort"),
    ("Mt.", "Mount"),
    ("Jan.", "January"),
    ("Feb.", "February"),
    ("Mar.", "March"),
    ("Apr.", "April"),
    ("Jun.", "June"),
    ("Jul.", "July"),
    ("Aug.", "August"),
    ("Sep.", "September"),
    ("Sept.", "September"),
    ("Oct.", "October"),
    ("Nov.", "November"),
    ("Dec.", "December"),
    ("Mon.", "Monday"),
    ("Tue.", "Tuesday"),
    ("Tues.", "Tuesday"),
    ("Wed.", "Wednesday"),
    ("Thu.", "Thursday"),
    ("Thur.", "Thursday"),
    ("Thurs.", "Thursday"),
    ("Fri.", "Friday"),
    ("Sat.", "Saturday"),
    ("Sun.", "Sunday"),
    ("vol.", "volume"),
    ("Vol.", "Volume"),
    ("ed.", "edition"),
    ("approx.", "approximately"),
    ("Approx.", "Approximately"),
    ("dept.", "department"),
    ("Dept.", "Department"),
    ("vs.", "versus"),
    ("w/", "with"),
    ("W/", "With"),
    ("w/o", "without"),
    ("W/o", "Without"),
];

/// Letters a word is stretched with ("sooo", "loooove"), in either case.
const STRETCHED: &str = "aeiouyhw";
/// The shortest run of one letter that is a stretch rather than spelling ("good").
const MIN_STRETCH: usize = 3;

/// Units written with a slash between letters, which keep it ("mg/kg").
const SLASH_UNITS: [&str; 30] = [
    "km", "kg", "mg", "ml", "mL", "dL", "mol", "kJ", "kW", "kWh", "Hz", "min", "hr", "hrs", "sec",
    "ft", "lb", "lbs", "oz", "mi", "mph", "kph", "yd", "cm", "mm", "yr", "wk", "mo", "kcal", "rpm",
];

pub(super) fn rewrite(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let rest = &words[index..];
        if let Some((spoken, used)) = span(out.last().map(String::as_str), rest) {
            out.push(spoken);
            index += used;
            continue;
        }
        out.push(claimed(rest[0]).unwrap_or_else(|| unclaimed(rest[0])));
        index += 1;
    }
    out.join(" ")
}

/// A rule that reads several words at once, and how many it read.
fn span(previous: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    numbers::scaled_money(words)
        .or_else(|| numbers::digit_groups(words))
        .or_else(|| numbers::ending_in(previous?, words))
        .or_else(|| lieutenant_colonel(words))
}

fn claimed(word: &str) -> Option<String> {
    RULES.iter().find_map(|rule| rule(word))
}

/// A word no rule claimed, with a slash between words read as a pause and `@` and `+` spelled
/// out; the pieces they leave get their own turn at the rules ("1500+20").
fn unclaimed(word: &str) -> String {
    if let Some(words) = slashed_words(word) {
        return each_word(&words);
    }
    if !word.contains(['@', '+']) {
        return word.to_string();
    }
    let mut sink = Sink::default();
    for ch in word.chars() {
        match ch {
            '@' => sink.push_word(AT),
            '+' => sink.push_word(PLUS),
            _ => sink.push(ch),
        }
    }
    each_word(&sink.finish())
}

fn each_word(text: &str) -> String {
    let words = text.split(' ');
    let spoken: Vec<String> = words
        .map(|word| claimed(word).unwrap_or_else(|| word.to_string()))
        .collect();
    spoken.join(" ")
}

/// "and/or" as "and or": every part a word of two letters or more and not a unit.
fn slashed_words(word: &str) -> Option<String> {
    let body = word.trim_end_matches(|ch: char| ch.is_ascii_punctuation() && ch != '/');
    let parts: Vec<&str> = body.split('/').collect();
    let wordlike = |part: &&str| {
        part.chars().count() > 1
            && part.chars().all(char::is_alphabetic)
            && !SLASH_UNITS.contains(part)
    };
    let suffix = &word[body.len()..];
    (parts.len() > 1 && parts.iter().all(wordlike)).then(|| parts.join(" ") + suffix)
}

fn abbreviation(word: &str) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    let (short, rest) = match suffix.strip_prefix('.') {
        Some(rest) => (&word[..=body.len()], rest),
        None => (body, suffix),
    };
    Some(format!("{}{rest}", long_form(short)?))
}

fn long_form(short: &str) -> Option<&'static str> {
    let (_, long) = ABBREVIATIONS
        .iter()
        .find(|(abbreviation, _)| *abbreviation == short)?;
    Some(long)
}

/// "Lt Col", which reads as "Lt" and "Col." do.
fn lieutenant_colonel(words: &[&str]) -> Option<(String, usize)> {
    let [first, second, ..] = words else {
        return None;
    };
    let (rank, suffix) = split_suffix(second);
    if *first != "Lt" || rank != "Col" {
        return None;
    }
    let spoken = format!("{} {}{suffix}", long_form("Lt")?, long_form("Col.")?);
    Some((spoken, 2))
}

/// A word with a letter stretched out ("sooo"), each stretch read as one letter.
fn elongation(word: &str) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    if !body.chars().all(|ch| ch.is_ascii_alphabetic())
        || body.chars().all(|ch| ch.is_ascii_uppercase())
    {
        return None;
    }
    let mut out = String::with_capacity(word.len());
    let mut stretched = false;
    let mut rest = body;
    while let Some(letter) = rest.chars().next() {
        let run = rest.len() - rest.trim_start_matches(letter).len();
        let stretch = run >= MIN_STRETCH && STRETCHED.contains(letter.to_ascii_lowercase());
        stretched |= stretch;
        out.extend(std::iter::repeat_n(letter, if stretch { 1 } else { run }));
        rest = &rest[run..];
    }
    stretched.then(|| out + suffix)
}
