//! Email and web addresses, read with their separators spoken.

use super::{AT, PLUS, UNDERSCORE, trailing_punctuation};

const SCHEMES: [(&str, &str); 2] = [("https://", "H-T-T-P-S"), ("http://", "H-T-T-P")];
const SCHEME_SEPARATOR: [&str; 3] = ["colon", "slash", "slash"];
const SEPARATORS: [(char, &str); 5] = [
    ('.', "dot"),
    ('-', "dash"),
    ('/', "slash"),
    ('_', UNDERSCORE),
    ('+', PLUS),
];
/// Generic top-level domains; any two letters are taken as a country's.
const GENERIC_DOMAINS: [&str; 24] = [
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "app", "dev", "xyz", "online",
    "site", "tech", "store", "shop", "blog", "news", "cloud", "page", "email", "media", "studio",
];
const COUNTRY_DOMAIN_LEN: usize = 2;
const WORLD_WIDE_WEB: &str = "www";

/// "john.doe@example.com" as "john dot doe at example dot com".
pub(super) fn email(word: &str) -> Option<String> {
    let (user, rest) = word.split_once('@')?;
    let user_char = |ch: char| ch.is_ascii_alphanumeric() || "-_.+".contains(ch);
    if user.is_empty() || !user.chars().all(user_char) {
        return None;
    }
    let (domain, rest) = domain(rest)?;
    let suffix = trailing_punctuation(rest)?;
    let mut words = speak(user);
    words.push(AT.into());
    words.extend(speak(domain));
    let top = words.pop()?;
    words.push(spell_short(&top));
    Some(words.join(" ") + suffix)
}

/// "https://gradium.ai/phonon" as "H-T-T-P-S colon slash slash gradium dot A-I slash phonon".
/// The last label must be a top-level domain in one case, unless a scheme or `www` says the word
/// is an address: "it.Now" is a sentence that lost a space.
pub(super) fn url(word: &str) -> Option<String> {
    let scheme = SCHEMES
        .iter()
        .find_map(|&(prefix, spoken)| Some((spoken, word.strip_prefix(prefix)?)));
    let rest = scheme.map_or(word, |(_, rest)| rest);
    let (domain, after) = domain(rest)?;
    let path = after
        .find(|ch: char| !(ch.is_alphanumeric() || ch == '-' || ch == '/'))
        .unwrap_or(after.len());
    let suffix = trailing_punctuation(&after[path..])?;
    let top = domain.rsplit('.').next()?;
    let mixed_case = top.chars().any(char::is_uppercase) && top.chars().any(char::is_lowercase);
    let named = scheme.is_some() || domain.to_lowercase().starts_with(WORLD_WIDE_WEB);
    if !is_top_level(top) || mixed_case && !named {
        return None;
    }
    let mut words: Vec<String> = scheme
        .into_iter()
        .flat_map(|(spoken, _)| [spoken].into_iter().chain(SCHEME_SEPARATOR))
        .map(str::to_string)
        .collect();
    let address = &rest[..domain.len() + path];
    words.extend(speak(address).iter().map(|piece| spell_short(piece)));
    Some(words.join(" ") + suffix)
}

/// Two labels or more of letters, digits and dashes, separated by dots, and what follows them.
fn domain(text: &str) -> Option<(&str, &str)> {
    let mut end = label_len(text)?;
    while let Some(next) = text[end..].strip_prefix('.').and_then(label_len) {
        end += 1 + next;
    }
    let (domain, rest) = text.split_at(end);
    domain.contains('.').then_some((domain, rest))
}

fn label_len(text: &str) -> Option<usize> {
    let len = text
        .find(|ch: char| !(ch.is_alphanumeric() || ch == '-'))
        .unwrap_or(text.len());
    (len > 0).then_some(len)
}

fn is_top_level(label: &str) -> bool {
    let country =
        label.len() == COUNTRY_DOMAIN_LEN && label.chars().all(|ch| ch.is_ascii_alphabetic());
    country
        || GENERIC_DOMAINS
            .iter()
            .any(|generic| generic.eq_ignore_ascii_case(label))
}

/// The pieces of an address part: runs of letters, each digit, and each separator's word.
fn speak(part: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut letters = String::new();
    for ch in part.chars() {
        let spoken = SEPARATORS.iter().find(|(separator, _)| *separator == ch);
        let piece = match spoken {
            Some((_, word)) => word.to_string(),
            None if ch.is_ascii_digit() => ch.to_string(),
            None => {
                letters.push(ch);
                continue;
            }
        };
        if !letters.is_empty() {
            pieces.push(std::mem::take(&mut letters));
        }
        pieces.push(piece);
    }
    if !letters.is_empty() {
        pieces.push(letters);
    }
    pieces
}

/// Two letters spelled out as an abbreviation ("fr" is "F-R"), and "www" as "W-W-W".
fn spell_short(piece: &str) -> String {
    let short = piece.len() == COUNTRY_DOMAIN_LEN || piece.eq_ignore_ascii_case(WORLD_WIDE_WEB);
    if !short || !piece.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return piece.to_string();
    }
    let letters: Vec<String> = piece
        .chars()
        .map(|ch| ch.to_ascii_uppercase().to_string())
        .collect();
    letters.join("-")
}
