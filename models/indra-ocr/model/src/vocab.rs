//! A recognizer's vocabulary: class 0 is the CTC blank, class `i + 1` its token `i`, a character
//! or, in a vocabulary with LaTeX commands, a command such as `\alpha`.

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::Error;
use crate::text::is_alpha;

/// The first Hangul jamo: a vocabulary that holds it reads syllables as jamo.
const JAMO: &str = "\u{1100}";

#[derive(Deserialize)]
struct Stored {
    blank: usize,
    chars: Vec<String>,
}

pub struct Vocab {
    tokens: Vec<String>,
    commands: bool,
    jamo: bool,
}

impl Vocab {
    /// The vocabulary as `meta.json` stores it: `{"blank": 0, "chars": [...]}`.
    pub fn parse(json: &str) -> Result<Self, Error> {
        let stored: Stored = serde_json::from_str(json)
            .map_err(|error| Error::Format(format!("a vocabulary: {error}")))?;
        if stored.blank != 0 {
            return Err(Error::Format(
                "a vocabulary whose blank is not class 0".into(),
            ));
        }
        Ok(Self {
            commands: stored.chars.iter().any(|token| token.chars().count() > 1),
            jamo: stored.chars.iter().any(|token| token == JAMO),
            tokens: stored.chars,
        })
    }

    /// Classes, the blank included.
    pub fn classes(&self) -> usize {
        self.tokens.len() + 1
    }

    /// The token of class `id` (not the blank).
    pub fn token(&self, id: u32) -> &str {
        &self.tokens[id as usize - 1]
    }

    /// The class of the space.
    pub fn space(&self) -> Option<u32> {
        let index = self.tokens.iter().position(|token| token == " ")?;
        Some(index as u32 + 1)
    }

    /// The text of classes `ids`: jamo composed into syllables, and a space after a command that
    /// a letter follows (`\alpha x`, not `\alphax`).
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<&str> = ids.iter().map(|&id| self.token(id)).collect();
        if self.jamo {
            return tokens.concat().nfc().collect();
        }
        if !self.commands {
            return tokens.concat();
        }
        let mut out = String::new();
        for (index, token) in tokens.iter().enumerate() {
            out.push_str(token);
            let next = tokens.get(index + 1).and_then(|next| next.chars().next());
            if is_command(token) && next.is_some_and(is_alpha) {
                out.push(' ');
            }
        }
        out
    }
}

/// A LaTeX command of letters, such as `\alpha`.
fn is_command(token: &str) -> bool {
    let mut characters = token.chars();
    characters.next() == Some('\\') && token.chars().count() > 2 && characters.all(is_alpha)
}
