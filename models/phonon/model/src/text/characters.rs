//! The character pass: typography folded to ASCII, pictographs dropped, symbols spelled out and
//! pauses written as commas.

use std::iter::Peekable;
use std::str::Chars;

const ASCII_FOLDS: [(&str, char); 4] = [
    ("«»‹›“”„‟″‶⹂❝❞❠〝〞〟＂", '"'),
    ("‘’‚‛′‵❛❜❟＇ʼʻʹ´`", '\''),
    ("‐‑‒―", '-'),
    ("…", '.'),
];

/// Subscripts and the characters they read as.
const SUBSCRIPTS: [(char, char); 26] = [
    ('₀', '0'),
    ('₁', '1'),
    ('₂', '2'),
    ('₃', '3'),
    ('₄', '4'),
    ('₅', '5'),
    ('₆', '6'),
    ('₇', '7'),
    ('₈', '8'),
    ('₉', '9'),
    ('₊', '+'),
    ('₋', '-'),
    ('ₐ', 'a'),
    ('ₑ', 'e'),
    ('ₒ', 'o'),
    ('ₓ', 'x'),
    ('ₕ', 'h'),
    ('ₖ', 'k'),
    ('ₗ', 'l'),
    ('ₘ', 'm'),
    ('ₙ', 'n'),
    ('ₚ', 'p'),
    ('ₛ', 's'),
    ('ₜ', 't'),
    ('ᵢ', 'i'),
    ('ⱼ', 'j'),
];
const SUPERSCRIPTS: [(char, char); 13] = [
    ('⁰', '0'),
    ('¹', '1'),
    ('²', '2'),
    ('³', '3'),
    ('⁴', '4'),
    ('⁵', '5'),
    ('⁶', '6'),
    ('⁷', '7'),
    ('⁸', '8'),
    ('⁹', '9'),
    ('⁻', '-'),
    ('⁺', '+'),
    ('ⁿ', 'n'),
];

/// Letters whose diacritic the tokenizer has no piece for, by the letter they fold to. The
/// accented letters of Latin-1 other than these the tokenizer reads as they are.
const ACCENTED: [(char, &str); 25] = [
    ('A', "ĀāĂăĄąǍǎǞǟǠǡǺǻȀȁȂȃȦȧḀḁẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ"),
    ('B', "ḂḃḄḅḆḇ"),
    ('C', "ĆćĈĉĊċČčḈḉ"),
    ('D', "ĎďḊḋḌḍḎḏḐḑḒḓ"),
    ('E', "ĒēĔĕĖėĘęĚěȄȅȆȇȨȩḔḕḖḗḘḙḚḛḜḝẸẹẺẻẼẽẾếỀềỂểỄễỆệ"),
    ('F', "Ḟḟ"),
    ('G', "ĜĝĞğĠġĢģǦǧǴǵḠḡ"),
    ('H', "ĤĥȞȟḢḣḤḥḦḧḨḩḪḫẖ"),
    ('I', "ÌìĨĩĪīĬĭĮįİǏǐȈȉȊȋḬḭḮḯỈỉỊị"),
    ('J', "Ĵĵǰ"),
    ('K', "ĶķǨǩḰḱḲḳḴḵ"),
    ('L', "ĹĺĻļĽľḶḷḸḹḺḻḼḽ"),
    ('M', "ḾḿṀṁṂṃ"),
    ('N', "ŃńŅņŇňǸǹṄṅṆṇṈṉṊṋ"),
    (
        'O',
        "ÒòŌōŎŏŐőƠơǑǒǪǫǬǭȌȍȎȏȪȫȬȭȮȯȰȱṌṍṎṏṐṑṒṓỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợ",
    ),
    ('P', "ṔṕṖṗ"),
    ('R', "ŔŕŖŗŘřȐȑȒȓṘṙṚṛṜṝṞṟ"),
    ('S', "ŚśŜŝŞşŠšȘșṠṡṢṣṤṥṦṧṨṩ"),
    ('T', "ŢţŤťȚțṪṫṬṭṮṯṰṱẗ"),
    ('U', "ŨũŪūŬŭŮůŰűŲųƯưǓǔǕǖǗǘǙǚǛǜȔȕȖȗṲṳṴṵṶṷṸṹṺṻỤụỦủỨứỪừỬửỮữỰự"),
    ('V', "ṼṽṾṿ"),
    ('W', "ŴŵẀẁẂẃẄẅẆẇẈẉẘ"),
    ('X', "ẊẋẌẍ"),
    ('Y', "ÝýÿŶŷŸȲȳẎẏẙỲỳỴỵỶỷỸỹ"),
    ('Z', "ŹźŻżŽžẐẑẒẓẔẕ"),
];

const SPOKEN: [(char, &str); 14] = [
    ('×', "times"),
    ('÷', "divided by"),
    ('≤', "less than or equal to"),
    ('≥', "greater than or equal to"),
    ('≠', "is not equal to"),
    ('≈', "approximately"),
    ('√', "square root of"),
    ('∞', "infinity"),
    ('∑', "sigma"),
    ('‰', "per mille"),
    ('±', "plus or minus"),
    ('=', "equals"),
    ('❤', "love"),
    ('♥', "love"),
];

const GREEK: [&str; 24] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega",
];
/// The code point inside each Greek alphabet that is not a letter of its own: the final sigma in
/// the lowercase one (read as sigma), an unassigned one in the uppercase one.
const LOWER_GREEK_GAP: char = 'ς';
const UPPER_GREEK_GAP: char = '\u{3a2}';

/// Characters read as a word break: dashes, brackets, bullets and arrows.
const BREAKS: &str = "–—[]{}*•‣◦·→←↑↓➡➜";
/// Pictograph blocks (from mahjong tiles to the symbols extension, misc symbols and dingbats,
/// misc symbols and arrows), and the variation selector and joiner emoji sequences carry.
const EMOJI: [std::ops::RangeInclusive<u32>; 5] = [
    0x1F000..=0x1FAFF,
    0x2600..=0x27BF,
    0x2B00..=0x2BFF,
    0xFE0F..=0xFE0F,
    0x200D..=0x200D,
];

/// The ASCII punctuation a following `.` or `,` keeps: quotes would be left unbalanced, the
/// others are read aloud. A `$` is kept after a digit ("5$.").
const KEPT_BEFORE_PAUSE: &str = "\"'@+%#&";

/// The output text, which keeps the substitutions from piling up punctuation: a `.` or `,`
/// swallows the whitespace and droppable punctuation before it, and whitespace never repeats.
#[derive(Default)]
pub(super) struct Sink {
    text: Vec<char>,
}

impl Sink {
    pub(super) fn push(&mut self, ch: char) {
        if matches!(ch, '.' | ',') {
            while self
                .text
                .last()
                .is_some_and(|&last| last.is_whitespace() || self.droppable(last))
            {
                self.text.pop();
            }
        }
        self.text.push(ch);
    }

    fn push_str(&mut self, text: &str) {
        text.chars().for_each(|ch| self.push(ch));
    }

    pub(super) fn push_word(&mut self, word: &str) {
        self.space();
        self.push_str(word);
        self.space();
    }

    fn space(&mut self) {
        if self.text.last().is_some_and(|last| !last.is_whitespace()) {
            self.text.push(' ');
        }
    }

    fn pause(&mut self) {
        self.trim_end();
        self.push(',');
        self.space();
    }

    fn trim_end(&mut self) {
        while self.text.last().is_some_and(|last| last.is_whitespace()) {
            self.text.pop();
        }
    }

    pub(super) fn finish(mut self) -> String {
        self.trim_end();
        self.text.into_iter().collect()
    }

    fn droppable(&self, last: char) -> bool {
        let amount = last == '$'
            && self
                .text
                .iter()
                .rev()
                .nth(1)
                .is_some_and(char::is_ascii_digit);
        last.is_ascii_punctuation() && !KEPT_BEFORE_PAUSE.contains(last) && !amount
    }

    /// Whether the text ends with the number 1 (not the last digit of 21 or 2.1).
    fn ends_with_lone_one(&self) -> bool {
        let end = self
            .text
            .iter()
            .rposition(|ch| !ch.is_whitespace())
            .map_or(0, |last| last + 1);
        match self.text[..end] {
            [.., before, '1'] => !(before.is_ascii_digit() || before == '.' || before == ','),
            ['1'] => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy)]
struct Around {
    previous: Option<char>,
    next: Option<char>,
}

impl Around {
    fn after_space(self) -> bool {
        self.previous.is_none_or(char::is_whitespace)
    }

    fn before_space(self) -> bool {
        self.next.is_none_or(char::is_whitespace)
    }

    fn before_digit(self) -> bool {
        self.next.is_some_and(|ch| ch.is_ascii_digit())
    }
}

pub(super) fn clean(text: &str) -> String {
    let mut sink = Sink::default();
    let mut chars = text.chars().peekable();
    let mut previous = None;
    while let Some(ch) = chars.next() {
        match ch {
            '★' | '☆' => sink.push_word(&stars(ch, &mut chars)),
            '°' => degrees(&mut sink, &mut chars),
            _ if lookup(&SUPERSCRIPTS, ch).is_some() => {
                power(&mut sink, ch, &mut chars, previous);
            }
            _ => {
                let next = chars.peek().copied();
                single(&mut sink, ch, Around { previous, next });
            }
        }
        previous = Some(ch);
    }
    sink.finish()
}

fn single(sink: &mut Sink, ch: char, around: Around) {
    if let Some(word) = spoken(ch) {
        sink.push_word(word);
        return;
    }
    match ch {
        '~' if around.before_digit() => sink.push_word("about"),
        '*' if around.after_space() && around.before_digit() => sink.push_word("star"),
        ':' if !around.after_space() && !around.before_space() => sink.push(':'),
        ';' | ':' | '(' | ')' => sink.pause(),
        _ if is_break(ch) => sink.space(),
        _ => sink.push(fold(ch).unwrap_or(ch)),
    }
}

/// A run of `★` and `☆`, with a trailing `½`: "4 out of 5 stars".
fn stars(first: char, chars: &mut Peekable<Chars>) -> String {
    let mut full = usize::from(first == '★');
    let mut empty = usize::from(first == '☆');
    while let Some(star) = chars.next_if(|&ch| matches!(ch, '★' | '☆')) {
        full += usize::from(star == '★');
        empty += usize::from(star == '☆');
    }
    let half = chars.next_if_eq(&'½').is_some();
    let rating = if half {
        format!("{full}.5")
    } else {
        full.to_string()
    };
    match empty {
        0 => format!("{rating} stars"),
        _ => format!("{rating} out of {} stars", full + empty + usize::from(half)),
    }
}

/// "degrees", with the scale when a `C` or `F` alone follows: "100°C" reads "100 degrees
/// Celsius", "45°Ceiling" keeps its word.
fn degrees(sink: &mut Sink, chars: &mut Peekable<Chars>) {
    let unit = if sink.ends_with_lone_one() {
        "degree"
    } else {
        "degrees"
    };
    sink.push_word(unit);
    let scale = match chars.peek() {
        Some('C') => "Celsius",
        Some('F') => "Fahrenheit",
        _ => return,
    };
    let mut after = chars.clone();
    after.next();
    if !after.next().is_some_and(char::is_alphanumeric) {
        chars.next();
        sink.push_str(scale);
    }
}

/// A run of superscripts: an exponent after a letter, digit or `)`, digits elsewhere ("¹³C").
fn power(sink: &mut Sink, first: char, chars: &mut Peekable<Chars>, previous: Option<char>) {
    let mut exponent: String = lookup(&SUPERSCRIPTS, first).into_iter().collect();
    while let Some(digit) = chars.peek().and_then(|&ch| lookup(&SUPERSCRIPTS, ch)) {
        exponent.push(digit);
        chars.next();
    }
    if !previous.is_some_and(|ch| ch.is_alphanumeric() || ch == ')') {
        sink.push_str(&exponent);
        return;
    }
    let spoken = match exponent.as_str() {
        "2" => "squared".to_string(),
        "3" => "cubed".to_string(),
        _ => format!("to the power of {exponent}"),
    };
    sink.push_word(&spoken);
}

fn spoken(ch: char) -> Option<&'static str> {
    let symbol = SPOKEN.iter().find(|(symbol, _)| *symbol == ch);
    symbol.map(|(_, word)| *word).or_else(|| greek(ch))
}

fn greek(ch: char) -> Option<&'static str> {
    let (first, gap) = match ch {
        'α'..='ω' => ('α', LOWER_GREEK_GAP),
        'Α'..='Ω' => ('Α', UPPER_GREEK_GAP),
        _ => return None,
    };
    let index = ch as usize - first as usize - usize::from(ch > gap);
    GREEK.get(index).copied()
}

fn is_break(ch: char) -> bool {
    let code = u32::from(ch);
    BREAKS.contains(ch)
        || ch.is_whitespace()
        || ch.is_control()
        || EMOJI.iter().any(|range| range.contains(&code))
}

fn fold(ch: char) -> Option<char> {
    let ascii = ASCII_FOLDS.iter().find(|(from, _)| from.contains(ch));
    ascii
        .map(|&(_, to)| to)
        .or_else(|| lookup(&SUBSCRIPTS, ch))
        .or_else(|| unaccented(ch))
}

fn unaccented(ch: char) -> Option<char> {
    let (base, _) = ACCENTED.iter().find(|(_, letters)| letters.contains(ch))?;
    Some(if ch.is_lowercase() {
        base.to_ascii_lowercase()
    } else {
        *base
    })
}

/// What `ch` maps to in a table of `(from, to)` characters.
fn lookup(table: &[(char, char)], ch: char) -> Option<char> {
    let pair = table.iter().find(|(from, _)| *from == ch);
    pair.map(|&(_, to)| to)
}
