//! Text from the transducer's steps, as the reference writes it: the tokens' pieces joined, each
//! `▁` a space except in the first piece, where it is dropped; words timed by the frames their
//! tokens start and end on, punctuation joining the word before it; sentences of words.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::decoder::Step;
use crate::model::FRAME_SECONDS;

/// The word separator of the pieces.
const SPACE: char = '▁';
/// A sentence ends after a word ending with one of these, or with the words (a segment's, at
/// most `segment::SEGMENT_SECONDS` long).
const SENTENCE_ENDS: [char; 3] = ['.', '?', '!'];
/// The characters of the vocabulary that are neither letters, digits nor Unicode punctuation
/// (category P): currency and math symbols.
const SYMBOLS: [char; 6] = ['$', '<', '>', '|', '£', '€'];

/// The pieces of the tokens, by id; special tokens (unknown, padding, the blank) have none.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct Vocabulary {
    pieces: Vec<Option<String>>,
}

/// A stretch of text and when it was said, in seconds from the start of the recording.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Timed {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl Vocabulary {
    pub fn parse(json: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(json).map_err(|error| Error::Format(format!("vocabulary: {error}")))
    }

    /// The text of the steps' tokens.
    pub fn text(&self, steps: &[Step]) -> String {
        let tokens: Vec<u32> = steps.iter().map(|step| step.token).collect();
        self.decode(&tokens)
    }

    /// The words of the steps, timed by `FRAME_SECONDS` per frame.
    pub fn words(&self, steps: &[Step]) -> Vec<Timed> {
        let mut words = Words::default();
        let mut frame = 0;
        for step in steps {
            let start = frame as f64 * FRAME_SECONDS;
            frame += step.frames;
            let end = frame as f64 * FRAME_SECONDS;
            let Some(piece) = self.piece(step.token) else {
                continue;
            };
            if piece.starts_with(SPACE) {
                words.flush(self);
            }
            if is_punctuation(&self.decode(&[step.token])) {
                words.punctuate(self, step.token, end);
            } else {
                words.extend(step.token, start, end);
            }
        }
        words.flush(self);
        words.done
    }

    fn piece(&self, token: u32) -> Option<&str> {
        self.pieces.get(token as usize)?.as_deref()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let pieces = tokens.iter().filter_map(|&token| self.piece(token));
        let mut text = String::new();
        for (index, piece) in pieces.enumerate() {
            let space = if index == 0 { "" } else { " " };
            text.push_str(&piece.replace(SPACE, space));
        }
        text
    }
}

/// The words of a decoding so far, and the tokens of the one being read.
#[derive(Default)]
struct Words {
    done: Vec<Timed>,
    tokens: Vec<u32>,
    start: f64,
    end: f64,
}

impl Words {
    fn flush(&mut self, vocabulary: &Vocabulary) {
        let text = vocabulary.decode(&self.tokens);
        let text = text.trim();
        if !text.is_empty() {
            self.done.push(Timed {
                text: text.into(),
                start: self.start,
                end: self.end,
            });
        }
        self.tokens.clear();
    }

    fn extend(&mut self, token: u32, start: f64, end: f64) {
        if self.tokens.is_empty() {
            self.start = start;
        }
        self.tokens.push(token);
        self.end = end;
    }

    /// A punctuation token joins the word being read, else the last word read.
    fn punctuate(&mut self, vocabulary: &Vocabulary, token: u32, end: f64) {
        if !self.tokens.is_empty() {
            self.tokens.push(token);
            self.end = end;
        } else if let Some(last) = self.done.last_mut() {
            last.text.push_str(&vocabulary.decode(&[token]));
            last.end = end;
        }
    }
}

/// The words grouped into sentences.
pub fn sentences(words: &[Timed]) -> Vec<Timed> {
    let mut sentences = Vec::new();
    let mut first = 0;
    for (index, word) in words.iter().enumerate() {
        let ends = word.text.trim_end().ends_with(SENTENCE_ENDS);
        if ends || index + 1 == words.len() {
            let group = &words[first..=index];
            let text: Vec<&str> = group.iter().map(|word| word.text.as_str()).collect();
            sentences.push(Timed {
                text: text.join(" "),
                start: group[0].start,
                end: word.end,
            });
            first = index + 1;
        }
    }
    sentences
}

/// Whether `text` is punctuation alone (Unicode category P), as the reference tests a token.
fn is_punctuation(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty()
        && text.chars().all(|character| {
            !character.is_alphanumeric()
                && !character.is_whitespace()
                && !SYMBOLS.contains(&character)
        })
}
