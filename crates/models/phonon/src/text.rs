//! English text normalization: rewrites text into what the checkpoint was trained to read.
//!
//! A character pass folds typographic characters to ASCII, drops emoji, brackets and bullets,
//! spells out the symbols the model cannot read and turns `;`, parentheses and a spaced `:` into
//! the comma pause. A word pass then gives each space-separated word, or a run of words such as a
//! phone number or "$2 million", to the first rule that claims it: numbers, money, dashed digits,
//! addresses, abbreviations, elongated words. `@`, `+` and `/` are spelled out in what no rule
//! claims. Times, dates and phone numbers written as one word stay as written.

mod addresses;
mod characters;
mod numbers;
mod words;

/// Sentence punctuation a word rule carries through after the part it rewrites.
const PUNCTUATION: &str = "!?.:,;";
const AT: &str = "at";
const PLUS: &str = "plus";

pub(crate) fn normalize(text: &str) -> String {
    let rewritten = words::rewrite(&characters::clean(text));
    strip_closing_quote(&rewritten).to_string()
}

fn strip_closing_quote(text: &str) -> &str {
    let text = text.trim_end_matches('"');
    match text.strip_suffix('\'') {
        Some(rest) if rest.ends_with(['.', '!', '?', ',', ';']) => rest,
        _ => text,
    }
}

fn split_suffix(word: &str) -> (&str, &str) {
    let body = word.trim_end_matches(|ch| PUNCTUATION.contains(ch));
    word.split_at(body.len())
}

/// `rest` when it is only sentence punctuation: what a rule may leave after what it read.
fn trailing_punctuation(rest: &str) -> Option<&str> {
    rest.chars()
        .all(|ch| PUNCTUATION.contains(ch))
        .then_some(rest)
}

fn all_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::normalize;

    const CASES: &[(&str, &str)] = &[
        ("Hello, world!", "Hello, world!"),
        ("", ""),
        ("“hello” world it's", "\"hello\" world it's"),
        ("«hello» ‹world›", "\"hello\" \"world"),
        ("a‐b‑c‒d―e", "a-b-c-d-e"),
        ("a–b—c", "a b c"),
        ("foo (bar) [baz] {qux} *quux*", "foo, bar, baz qux quux"),
        ("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e"),
        ("wait… a…b", "wait. a.b"),
        ("user@host @home", "user at host at home"),
        ("café résumé 日本語", "café résumé 日本語"),
        ("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun"),
        (
            "party 🎉 cup 🏆 globe 🌍 pizza 🍕 star ⭐",
            "party cup globe pizza star",
        ),
        ("sun ☀️ dev 👩‍💻 done", "sun dev done"),
        ("Fish &, chips &.", "Fish &, chips &."),
        ("a;b:c", "a, b:c"),
        ("time: 10:30", "time, 10:30"),
        ("a :b", "a, b"),
        ("note:", "note,"),
        (":start", ", start"),
        ("; leading", ", leading"),
        ("hello (world)", "hello, world,"),
        ("foo ; bar  :  baz", "foo, bar, baz"),
        ("a   b\t\tc\n\nd", "a b c d"),
        ("hello   ", "hello"),
        ("a • b • c", "a b c"),
        (
            "“Hello”; please email user@host (now)… 🚀",
            "\"Hello\", please email user at host, now.",
        ),
        (
            "Numbers: one, two, three, four, five. Special items: at sign, hash, dollar, percent.",
            "Numbers, one, two, three, four, five. Special items, at sign, hash, dollar, percent.",
        ),
        (
            "The conference will be held on Tuesday, March 15th at 3:30 PM.",
            "The conference will be held on Tuesday, March 15th at 3:30 PM.",
        ),
        (
            "Write to laurent.mazare@gmail.com!",
            "Write to laurent dot mazare at gmail dot com!",
        ),
        (
            "Mail (laurent+tag@gmail.com)",
            "Mail, laurent plus tag at gmail dot com,",
        ),
        ("a@.", "a at."),
        ("x @, y", "x at, y"),
        ("C++.", "C plus plus."),
        ("1500+20", "1 thousand 500 plus 20"),
        (
            "+33612345678",
            "plus 33 billion 612 million 345 thousand 678",
        ),
        (
            "It costs $1500 today.",
            "It costs 1 thousand 500 dollars today.",
        ),
        ("It costs 5$.", "It costs 5 dollars."),
        ("It costs -$5.", "It costs minus 5 dollars."),
        ("In $.", "In."),
        ("Only £50 left.", "Only 50 pounds left."),
        (
            "The fee is 2.9% plus 30¢.",
            "The fee is 2.9% plus 30 cents.",
        ),
        (
            "Call 555-123-4567 before 2024-05-12, it costs 12345678.",
            "Call 555 123 4 5 6 7 before 2024-05-12, it costs 12 million 345 thousand 678.",
        ),
        (
            "Visit www.kyutai.fr today.",
            "Visit W-W-W dot kyutai dot F-R today.",
        ),
        ("It's 1:06pm on 20/12/2015.", "It's 1:06pm on 20/12/2015."),
        (
            "It sold for £3.2m.",
            "It sold for 3 point 2 million pounds.",
        ),
        (
            "We raised $2 million, then €3bn.",
            "We raised 2 million dollars, then 3 billion euros.",
        ),
        (
            "Ring +44 20 7946 0958.",
            "Ring plus four four, two zero, seven nine four six, zero nine five eight.",
        ),
        (
            "Text 07700 900123 now.",
            "Text zero seven seven zero zero, nine zero zero one two three now.",
        ),
        ("In 2019 2020 2021", "In 2019 2020 2021"),
        (
            "The card ending in -4098.",
            "The card ending in four zero nine eight.",
        ),
        ("It fell to -12.", "It fell to minus 12."),
        (
            "Your order number is 48213.",
            "Your order number is 48 thousand 213.",
        ),
        (
            "Lt Col Vann and Lt. Col. Harris.",
            "Lieutenant Colonel Vann and Lieutenant Colonel Harris.",
        ),
        (
            "Rated ★★★★☆, or ★★★½.",
            "Rated 4 out of 5 stars, or 3 point 5 stars.",
        ),
        (
            "Wait ~10 minutes, ±2.",
            "Wait about 10 minutes, plus or minus 2.",
        ),
        (
            "I ❤️ NY, dial *67, it's 4.5%.",
            "I love NY, dial star 67, it's 4.5%.",
        ),
        ("Press 1, then #.", "Press 1, then #."),
        ("That was sooo good.", "That was so good."),
        ("He said, \"go home!\"", "He said, \"go home!"),
        ("It's the boys'", "It's the boys'"),
        ("'Stop it.'", "'Stop it."),
        ("It's -40° outside.", "It's minus 40 degrees outside."),
        (
            "Water boils at 100°C, or 212 °F.",
            "Water boils at 100 degrees Celsius, or 212 degrees Fahrenheit.",
        ),
        (
            "It was -3.5°C, then 1°C.",
            "It was minus 3 point 5 degrees Celsius, then 1 degree Celsius.",
        ),
        ("Turn 90°, then 1°.", "Turn 90 degrees, then 1 degree."),
        ("A 45°Ceiling", "A 45 degrees Ceiling"),
        (
            "1920×1080, 6÷2, x≤3, y≥4, a≠b, π≈3.14",
            "1920 times 1 thousand 80, 6 divided by 2, x less than or equal to 3, \
             y greater than or equal to 4, a is not equal to b, pi approximately 3 point 14",
        ),
        (
            "√2, ∞, 5‰ and ∑",
            "square root of 2, infinity, 5 per mille and sigma",
        ),
        (
            "x² + y³ = 10⁻⁶",
            "x squared plus y cubed equals 10 to the power of minus 6",
        ),
        ("C₈H₁₈ and ¹³C", "C8H18 and 13C"),
        ("ΔH, α and ω", "delta H, alpha and omega"),
        (
            "and/or, to/from, reality/the deity.",
            "and or, to from, reality the deity.",
        ),
        (
            "40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025",
            "40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025",
        ),
        (
            "bǎohù zìjǐ in İstanbul, città",
            "baohù ziji in Istanbul, città",
        ),
        ("From 10 $20 bills.", "From 10 20 dollars bills."),
        (
            "Call me at 5:30 on 12/03/2026, it costs $1,234.56 or 42%.",
            "Call me at 5:30 on 12/03/2026, it costs 1 thousand 234 dollars and 56 cents \
             or 42%.",
        ),
        (
            "Email john.doe@example.com or visit https://gradium.ai/phonon for details.",
            "Email john dot doe at example dot com or visit H-T-T-P-S colon slash slash \
             gradium dot A-I slash phonon for details.",
        ),
        (
            "The meeting is at 9am; bring 3 copies (or more) — it's \"important\".",
            "The meeting is at 9am, bring 3 copies, or more, it's \"important\".",
        ),
        (
            "Dr. Smith paid €250 for 2.5 kg of apples, i.e. about 5 lbs.",
            "Dr. Smith paid 250 euros for 2 point 5 kg of apples, i.e. about 5 lbs.",
        ),
        (
            "Temperatures hit 98.6°F (37°C) at 3:45pm on Jan. 5th.",
            "Temperatures hit 98 point 6 degrees Fahrenheit, 37 degrees Celsius, at 3:45pm \
             on January 5th.",
        ),
        (
            "I loooove it!!! ★★★★☆ 4/5 stars… 😀",
            "I love it!!! 4 out of 5 stars 4/5 stars.",
        ),
        (
            "Pi is roughly 3.14159 and e ≈ 2.718; x² + y² = r².",
            "Pi is roughly 3 point 14159 and e approximately 2 point 718, x squared plus y \
             squared equals r squared.",
        ),
        (
            "Room 1-234, phone 555-123-4567, code A1B2-C3D4.",
            "Room 1-234, phone 555 123 4 5 6 7, code A1B2-C3D4.",
        ),
        (
            "We sold 1,000,000 units in Q3 2026, up 12.5% YoY.",
            "We sold 1 million units in Q3 2026, up 12.5% YoY.",
        ),
        ("Hello world", "Hello world"),
    ];

    #[test]
    fn normalizes_like_the_reference_frontend() {
        for &(input, expected) in CASES {
            assert_eq!(normalize(input), expected, "{input:?}");
        }
    }
}
