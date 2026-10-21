//! The text frontend (`MingTTS.prepare_text` in the reference, `frontend/`): split the input
//! into fragments of at most 50 semantic units (CJK characters + Latin words), synthesized one
//! by one (`segment`), and normalize each fragment (`normalize`).
//!
//! Per fragment: English numbers become words (`numbers`, `number_en.normalize_numbers` with
//! inflect's wording), then `toolkit.normalize_text_kit` (characters outside the TTS whitelist
//! become `，`, CJK text gets CJK punctuation). Paralinguistic tags (`[Laughter]` …) are routed
//! around both.
//!
//! Not ported: the pynini FST normalizer (`TalkerTN`) that runs between those two steps in the
//! reference. In its place a hyphen inside a word (`ninety-eight`, `well-known`) becomes a space,
//! because the whitelist step would otherwise turn it into a pause (`ninety，eight`).

mod normalize;
mod numbers;
mod segment;

use std::ops::RangeInclusive;
use std::sync::LazyLock;

use regex::Regex;

/// The ideographs a semantic unit counts (`text_segment_cut`'s `[一-龥]`).
const CJK: RangeInclusive<char> = '\u{4e00}'..='\u{9fa5}';
/// The ideographs the reference's `is_chinese` and whitelist accept: a wider range than `CJK`.
const CHINESE: RangeInclusive<char> = '\u{4e00}'..='\u{9fff}';
/// CJK Unified Ideographs Extension A, which `add_period` also ends with `。`.
const CJK_EXTENSION_A: RangeInclusive<char> = '\u{3400}'..='\u{4dbf}';

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("the patterns in this module are valid")
}

fn is_cjk(ch: char) -> bool {
    CJK.contains(&ch)
}

/// `text_segment_cut.is_chinese`.
fn is_chinese(text: &str) -> bool {
    text.chars().any(|ch| CHINESE.contains(&ch))
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
        Segmentation::Sentences => normalized(&segment::cut(&add_period(&text))),
        Segmentation::Whole => normalized(&[add_period(&text)]),
        Segmentation::Verbatim => segment::cut(text.trim()),
    };
    fragments
        .into_iter()
        .filter(|fragment| !fragment.trim().is_empty())
        .collect()
}

fn normalized(fragments: &[String]) -> Vec<String> {
    let normalized = fragments
        .iter()
        .map(|fragment| normalize::fragment(fragment));
    normalized.collect()
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
    let cjk = CHINESE.contains(&last) || CJK_EXTENSION_A.contains(&last);
    format!("{stripped}{}", if cjk { "。" } else { "." })
}

#[cfg(test)]
mod tests {
    use super::{Segmentation, fragments};

    #[test]
    fn numbers_read_like_the_reference() {
        let cases = [
            (
                "In 1998 the company sold 2,500 units, which was a forty percent increase over the previous year.",
                "In one thousand, nine hundred ninety eight the company sold two thousand, five hundred units, which was a forty percent increase over the previous year.",
            ),
            (
                "It costs $12.99 or £1, and 3/4 of 10ms is 7.5 ms, 25% vs -3.5%.",
                "It costs twelve point nine nine dollars or one pound , and three quarters of ten milliseconds is seven point five milliseconds , twenty five percent vs minus three point five percent .",
            ),
            (
                "Meet me on the 21st at 3:30, version v2 and gpt-4.5 ready.",
                "Meet me on the twenty one at three : thirty , version v two and gpt four point five ready.",
            ),
            (
                "Prices rose 1.5 percent in Q3 2024!",
                "Prices rose one point five percent in Q three two thousand twenty four !",
            ),
            (
                "Counts 101 1001 1100 123456 1000001 2345678 0",
                "Counts one hundred one one thousand one one thousand, one hundred one hundred twenty three thousand, four hundred fifty six one million one two million, three hundred forty five thousand, six hundred seventy eight zero .",
            ),
            (
                "The 2025th visit took 1/21 and 1/20 of it.",
                "The two thousand and twenty five visit took one twenty first and one twentieth of it.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(fragments(input, Segmentation::Whole), [expected], "{input}");
        }
    }

    #[test]
    fn long_text_is_cut_like_the_reference() {
        let text = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold. Somewhere below, a train whistled twice, and the lights of the village flickered on one by one. Nobody in the valley seemed to notice; they were busy with supper, with stories, and with the small business of ending a long day. It was 7.5 degrees by nightfall";
        let expected = [
            "The evening wind drifted down from the mountain pass， and the distant clouds slowly turned gold。 Somewhere below，a train whistled twice， and the lights of the village flickered on one by one。 Nobody in the valley seemed to notice; they were busy with supper，",
            "with stories，and with the small business of ending a long day。 It was seven point five degrees by nightfall。",
        ];
        assert_eq!(fragments(text, Segmentation::Sentences), expected);
    }

    #[test]
    fn paralinguistic_tags_keep_their_spacing() {
        assert_eq!(
            fragments("Well [Laughter] fine", Segmentation::Whole),
            ["Well [Laughter] fine."]
        );
    }

    #[test]
    fn short_text_is_one_fragment_kept_as_is() {
        let text = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";
        assert_eq!(fragments(text, Segmentation::Sentences), [text]);
    }
}
