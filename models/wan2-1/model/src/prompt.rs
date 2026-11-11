//! A prompt as umT5's tokens: Wan's cleaning (`ftfy`'s expansion of Latin ligatures, narrowing of
//! full-width characters and straightening of curly quotes, then runs of whitespace to one space,
//! trimmed; the rest of `ftfy` and the HTML unescaping are not repeated), its SentencePiece tokens
//! (`tokenizer.json`), the first `TEXT_TOKENS − 1` of them, then `</s>`, as
//! `HuggingfaceTokenizer(clean='whitespace')` gives them.

use bundle::Tokenizer;
use bundle::python::isspace;
use unicode_normalization::UnicodeNormalization;

use crate::Error;
use crate::model::{END, TEXT_TOKENS};

const IDEOGRAPHIC_SPACE: char = '\u{3000}';
const WIDTH_FORMS_START: char = '\u{FF01}';
const WIDTH_FORMS_END: char = '\u{FFEF}';
const MODIFIER_APOSTROPHE: char = '\u{02BC}';
const SINGLE_QUOTES_START: char = '\u{2018}';
const SINGLE_QUOTES_END: char = '\u{201B}';
const DOUBLE_QUOTES_START: char = '\u{201C}';
const DOUBLE_QUOTES_END: char = '\u{201F}';

/// `ftfy.chardata.LIGATURES`: the single characters that are several letters.
const LIGATURES: [(char, &str); 22] = [
    ('\u{0132}', "IJ"),
    ('\u{0133}', "ij"),
    ('\u{0149}', "\u{02BC}n"),
    ('\u{01F1}', "DZ"),
    ('\u{01F2}', "Dz"),
    ('\u{01F3}', "dz"),
    ('\u{01C4}', "D\u{017D}"),
    ('\u{01C5}', "D\u{017E}"),
    ('\u{01C6}', "d\u{017E}"),
    ('\u{01C7}', "LJ"),
    ('\u{01C8}', "Lj"),
    ('\u{01C9}', "lj"),
    ('\u{01CA}', "NJ"),
    ('\u{01CB}', "Nj"),
    ('\u{01CC}', "nj"),
    ('\u{FB00}', "ff"),
    ('\u{FB01}', "fi"),
    ('\u{FB02}', "fl"),
    ('\u{FB03}', "ffi"),
    ('\u{FB04}', "ffl"),
    ('\u{FB05}', "\u{017F}t"),
    ('\u{FB06}', "st"),
];

/// `ftfy.fix_latin_ligatures`.
fn expand_ligatures(text: &str) -> String {
    let mut expanded = String::with_capacity(text.len());
    for character in text.chars() {
        match LIGATURES
            .iter()
            .find(|(ligature, _)| *ligature == character)
        {
            Some((_, letters)) => expanded.push_str(letters),
            None => expanded.push(character),
        }
    }
    expanded
}

/// `ftfy.fix_character_width`: the ideographic space becomes a space and every character of the
/// full-width and half-width forms block its NFKC form (the full-width comma `，` becomes `,`).
fn narrow_widths(text: &str) -> String {
    let mut narrowed = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            IDEOGRAPHIC_SPACE => narrowed.push(' '),
            WIDTH_FORMS_START..=WIDTH_FORMS_END => narrowed.extend(character.nfkc()),
            _ => narrowed.push(character),
        }
    }
    narrowed
}

/// `ftfy.uncurl_quotes`.
fn uncurl_quotes(text: &str) -> String {
    let straight = |character| match character {
        MODIFIER_APOSTROPHE | SINGLE_QUOTES_START..=SINGLE_QUOTES_END => '\'',
        DOUBLE_QUOTES_START..=DOUBLE_QUOTES_END => '"',
        _ => character,
    };
    text.chars().map(straight).collect()
}

/// The ids of `text`'s tokens, at most `TEXT_TOKENS`, the last `END`.
pub fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    let cleaned = uncurl_quotes(&narrow_widths(&expand_ligatures(text)));
    let words: Vec<&str> = cleaned
        .split(isspace)
        .filter(|word| !word.is_empty())
        .collect();
    let encoded = tokenizer.encode(&words.join(" "))?;
    let mut ids = encoded[..encoded.len().min(TEXT_TOKENS - 1)].to_vec();
    ids.push(END);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<unk>": 0, "a": 2, "b": 3, ",": 4, "fi": 5, "'": 6, "\"": 7, "ffi": 8
            },
            "unk_token": "<unk>"
        }
    }"#;

    #[test]
    fn ftfy_cleaning_is_applied_before_tokenizing() {
        let tokenizer = Tokenizer::parse(WORDS.as_bytes()).expect("the words tokenizer parses");
        let cases = [
            ("a，b\u{3000}a", vec![2, 4, 3, 2, END]),
            ("a \u{FB01} b \u{FB03}", vec![2, 5, 3, 8, END]),
            ("a\u{2018}b\u{2019}", vec![2, 6, 3, 6, END]),
            ("\u{201C}a\u{201D}", vec![7, 2, 7, END]),
        ];
        for (text, want) in cases {
            let ids = tokenize(&tokenizer, text).expect("the prompt tokenizes");
            assert_eq!(ids, want, "{text:?}");
        }
    }
}
