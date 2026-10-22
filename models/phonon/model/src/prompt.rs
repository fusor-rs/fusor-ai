//! What the model reads: the text normalized (`text`), cut into chunks of whole sentences, each
//! prepared as the checkpoint was trained (single spaces, a capital first letter, a final period)
//! and tokenized; and the voices shipped in the bundle.

use bundle::Tokenizer;

use crate::Error;
use crate::text;

/// `(id, label)` of every voice the compiler puts in the bundle.
pub const VOICES: [(&str, &str); 7] = [
    ("default", "Default — English"),
    ("harper", "Harper — US English, female"),
    ("miles", "Miles — US English, male"),
    ("delia", "Delia — US English, female, low"),
    ("preston", "Preston — US English, male, measured"),
    ("maeve", "Maeve — British English, female"),
    ("edwin", "Edwin — British English, male"),
];

/// Sentences are packed into chunks of up to this many tokens.
const CHUNK_TOKENS: usize = 50;
/// A longer sentence is cut at words until every piece fits.
pub const MAX_CHUNK_TOKENS: usize = 200;
const FRAME_RATE: f64 = 12.5;
/// A chunk may take 2 s plus a second per 3 tokens.
const LEAD_SECONDS: f64 = 2.0;
const TOKENS_PER_SECOND: f64 = 3.0;
/// Frames kept after the end-of-speech frame: more for texts of at most `SHORT_TEXT_WORDS`
/// words.
const SHORT_TEXT_WORDS: usize = 4;
const FRAMES_AFTER_EOS: usize = 1;
const FRAMES_AFTER_EOS_SHORT: usize = 3;
const SENTENCE_ENDS: [char; 3] = ['.', '!', '?'];
const CLAUSE_ENDS: [char; 3] = [',', ';', ':'];
const CLOSERS: [char; 7] = ['"', '\'', ')', ']', '”', '’', '»'];
/// Words ending in a period that stay inside their sentence (`text` expands the others, such as
/// "Mt." and "vs.").
const SENTENCE_INTERNAL: [&str; 10] = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Jr.", "Sr.", "e.g.", "i.e.",
];

/// One chunk of a request: synthesized on its own from the voice's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub tokens: Vec<u32>,
    pub frames_after_eos: usize,
    /// The most frames the chunk may take.
    pub frame_budget: usize,
}

pub struct Prompter {
    tokenizer: Tokenizer,
}

impl Prompter {
    pub(crate) fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// The chunks of `text`, in order.
    pub fn chunks(&self, text: &str) -> Result<Vec<Chunk>, Error> {
        let prepared = prepare(&text::normalize(text));
        let mut pieces = Vec::new();
        for sentence in sentences(&prepared) {
            self.fit(sentence, &mut pieces)?;
        }
        let mut chunks: Vec<(String, usize)> = Vec::new();
        for (piece, tokens) in pieces {
            match chunks.last_mut() {
                Some((chunk, total)) if *total + tokens <= CHUNK_TOKENS => {
                    chunk.push(' ');
                    chunk.push_str(&piece);
                    *total += tokens;
                }
                _ => chunks.push((piece, tokens)),
            }
        }
        chunks
            .into_iter()
            .map(|(chunk, _)| self.chunk(chunk))
            .collect()
    }

    /// `piece` and its token count once prepared, cut at words (a single word at its middle
    /// character) until every piece has at most `MAX_CHUNK_TOKENS`.
    fn fit(&self, piece: String, out: &mut Vec<(String, usize)>) -> Result<(), Error> {
        let tokens = self.tokenizer.encode(&prepare(&piece))?.len();
        let halves = if tokens <= MAX_CHUNK_TOKENS {
            None
        } else {
            halves(&piece)
        };
        let Some((first, second)) = halves else {
            out.push((piece, tokens));
            return Ok(());
        };
        self.fit(first, out)?;
        self.fit(second, out)
    }

    fn chunk(&self, text: String) -> Result<Chunk, Error> {
        let tokens = self.tokenizer.encode(&prepare(&text))?;
        if tokens.len() > MAX_CHUNK_TOKENS {
            return Err(Error::Request(format!(
                "a piece of the text is {} tokens and cannot be cut below {MAX_CHUNK_TOKENS}",
                tokens.len()
            )));
        }
        Ok(Chunk {
            frames_after_eos: frames_after_eos(&text),
            text,
            frame_budget: frame_budget(tokens.len()),
            tokens,
        })
    }
}

/// The most frames `tokens` text tokens may take.
pub(crate) fn frame_budget(tokens: usize) -> usize {
    ((tokens as f64 / TOKENS_PER_SECOND + LEAD_SECONDS) * FRAME_RATE).ceil() as usize
}

/// The frames to keep after the end-of-speech frame of `text`.
fn frames_after_eos(text: &str) -> usize {
    if text.split_whitespace().count() > SHORT_TEXT_WORDS {
        FRAMES_AFTER_EOS
    } else {
        FRAMES_AFTER_EOS_SHORT
    }
}

/// Single spaces, a capital first letter and a period after a final letter or digit.
fn prepare(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut prepared = words.join(" ");
    if let Some(first) = prepared.chars().next() {
        prepared.replace_range(..first.len_utf8(), &first.to_uppercase().to_string());
    }
    if prepared.chars().last().is_some_and(char::is_alphanumeric) {
        prepared.push('.');
    }
    prepared
}

/// The sentences of single-spaced text: each ends at a word ending in `.`, `!` or `?` (before
/// any closing quotes or brackets) that is not one of `SENTENCE_INTERNAL`.
fn sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for word in text.split(' ').filter(|word| !word.is_empty()) {
        current.push(word);
        if ends_with(word, &SENTENCE_ENDS) && !SENTENCE_INTERNAL.contains(&word) {
            sentences.push(current.join(" "));
            current.clear();
        }
    }
    if !current.is_empty() {
        sentences.push(current.join(" "));
    }
    sentences
}

fn ends_with(word: &str, marks: &[char]) -> bool {
    word.trim_end_matches(CLOSERS).ends_with(marks)
}

/// A piece that is too long cut in two: after the clause end nearest its middle, else at the
/// middle word, else (one word) at the middle character; `None` for a single character.
fn halves(piece: &str) -> Option<(String, String)> {
    let words: Vec<&str> = piece.split(' ').collect();
    if words.len() < 2 {
        let middle = piece.char_indices().nth(piece.chars().count() / 2)?.0;
        return (middle > 0).then(|| (piece[..middle].to_string(), piece[middle..].to_string()));
    }
    let middle = words.len() / 2;
    let cut = (1..words.len())
        .filter(|&end| ends_with(words[end - 1], &CLAUSE_ENDS))
        .min_by_key(|&end| end.abs_diff(middle))
        .unwrap_or(middle);
    Some((words[..cut].join(" "), words[cut..].join(" ")))
}

#[cfg(test)]
mod tests {
    use bundle::Tokenizer;

    use super::{Chunk, Prompter};

    /// One token per space-separated word.
    const WORD_TOKENIZER: &str = r#"{
        "version": "1.0",
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
    }"#;

    /// One token per character.
    const CHARACTER_TOKENIZER: &str = r#"{
        "version": "1.0",
        "pre_tokenizer": {
            "type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false
        },
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
    }"#;

    fn sentence(words: usize) -> String {
        let mut text = vec!["word"; words].join(" ");
        text.push('.');
        text
    }

    fn summary(chunks: &[Chunk]) -> Vec<(usize, usize, usize)> {
        chunks
            .iter()
            .map(|chunk| {
                (
                    chunk.tokens.len(),
                    chunk.frames_after_eos,
                    chunk.frame_budget,
                )
            })
            .collect()
    }

    #[test]
    fn chunks_pack_whole_sentences_and_cut_long_ones() {
        let prompter = Prompter::new(Tokenizer::parse(WORD_TOKENIZER.as_bytes()).unwrap());
        let titles = prompter
            .chunks(&format!("{} Dr. Smith paid", sentence(48)))
            .unwrap();
        assert_eq!(titles[1].text, "Dr. Smith paid.");
        assert_eq!(summary(&titles), [(48, 1, 225), (3, 3, 38)]);
        let three = [sentence(20), sentence(20), sentence(20)].join(" ");
        assert_eq!(
            summary(&prompter.chunks(&three).unwrap()),
            [(40, 1, 192), (20, 1, 109)]
        );
        let mut long = vec!["word"; 300];
        long[119] = "word,";
        let pieces = prompter.chunks(&long.join(" ")).unwrap();
        assert_eq!(summary(&pieces), [(120, 1, 525), (180, 1, 775)]);
    }

    #[test]
    fn a_word_too_long_for_one_chunk_is_cut() {
        let prompter = Prompter::new(Tokenizer::parse(CHARACTER_TOKENIZER.as_bytes()).unwrap());
        let chunks = prompter.chunks(&"語".repeat(450)).unwrap();
        let tokens: Vec<usize> = chunks.iter().map(|chunk| chunk.tokens.len()).collect();
        assert_eq!(tokens, [113, 114, 114, 113]);
    }
}
