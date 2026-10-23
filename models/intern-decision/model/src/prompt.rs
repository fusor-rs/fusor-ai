//! The prompt a request is decided with, as the reference compiles it: the system prompt, the
//! user's turn (each image, then the state and the decision schema, every option under its
//! symbol), and the assistant's turn already written — the empty thinking block, then a JSON
//! object with a `<decision>` marker for each question. The logits of the row just before each
//! marker decide its question.

use std::ops::Range;

use blocks::rotary::Position;
use blocks::vit::Grid;
use bundle::Tokenizer;
use serde_json::{Map, Value};

use crate::Error;
use crate::json;
use crate::model::MERGE;
use crate::request::{Question, SYMBOLS};

const SYSTEM: &str = "You are a careful decision assistant. Use the state and decision schema in \
the user message to make the requested decisions. For every field, choose exactly one answer \
symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each \
field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include \
explanations, Markdown, or extra text.";
const INSTRUCTION: &str = "Return one answer for every field using the supplied answer symbols.";
/// What a question's answer is written as in the assistant's turn, and its token.
const DECISION: &str = "<decision>";
pub const DECISION_TOKEN: u32 = 248_077;
/// The token each of an image's language-model tokens holds the place of, and its text.
const IMAGE_TOKEN: u32 = 248_056;
const IMAGE_PAD: &str = "<|image_pad|>";
/// The indentation of the state's JSON and of the assistant's answer object.
const STATE_INDENT: usize = 2;
const ANSWER_INDENT: usize = 4;

pub struct Prompt {
    pub ids: Vec<u32>,
    /// The rows of each image's tokens, in order, with the image's grid of merged patches.
    pub images: Vec<(Range<usize>, Grid)>,
    /// For each question, in order, the row whose logits decide it.
    pub decisions: Vec<usize>,
}

impl Prompt {
    /// The prompt for `questions` about `state`, with images of `grids` patches. The chat
    /// template trims each turn as Python's `str.strip` does.
    pub fn new(
        tokenizer: &Tokenizer,
        state: &Value,
        questions: &[Question],
        grids: &[Grid],
    ) -> Result<Self, Error> {
        let schema = schema(questions);
        let user = format!(
            "{INSTRUCTION}\n\n## State\n{}\n## Decision schema\n{schema}",
            json::dumps(state, STATE_INDENT)
        );
        if user.contains(DECISION) {
            return Err(Error::Request(format!(
                "{DECISION} is reserved: the request may not contain it"
            )));
        }
        let merged: Vec<Grid> = grids.iter().map(|&grid| merged(grid)).collect();
        let pictures = merged.iter().enumerate().map(|(index, grid)| {
            let pads = IMAGE_PAD.repeat(grid.patches());
            format!(
                "Picture {}: <|vision_start|>{pads}<|vision_end|>",
                index + 1
            )
        });
        let user = format!("{}{user}", pictures.collect::<String>());
        let answers: Map<String, Value> = questions
            .iter()
            .map(|question| (question.name.clone(), DECISION.into()))
            .collect();
        let answer = json::dumps(&Value::Object(answers), ANSWER_INDENT);
        let text = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n",
            strip(&user),
            strip(&answer)
        );
        let ids = tokenizer.encode(&text)?;
        let images = image_rows(&ids, &merged)?;
        let decisions: Vec<usize> = (1..ids.len())
            .filter(|&row| ids[row] == DECISION_TOKEN)
            .map(|row| row - 1)
            .collect();
        if decisions.len() != questions.len() {
            return Err(Error::Request(format!(
                "{} decision markers for {} questions: a question's name holds a marker",
                decisions.len(),
                questions.len()
            )));
        }
        Ok(Self {
            ids,
            images,
            decisions,
        })
    }

    /// Every row's rotary position: a text row's on every axis, continuing from the last; an
    /// image token's at the image's start, down its row and across its column; the text after
    /// an image from past its longer side.
    pub fn positions(&self) -> Vec<Position> {
        let mut positions = Vec::with_capacity(self.ids.len());
        let mut images = self.images.iter().peekable();
        let mut next = 0;
        while positions.len() < self.ids.len() {
            let row = positions.len();
            let Some((rows, grid)) = images.next_if(|(rows, _)| rows.start == row) else {
                positions.push(Position::uniform(next));
                next += 1;
                continue;
            };
            positions.extend((0..rows.len()).map(|token| Position {
                temporal: next,
                height: next + token / grid.columns,
                width: next + token % grid.columns,
            }));
            next += grid.rows.max(grid.columns);
        }
        positions
    }
}

/// The decision schema: each question's instructions, then its options under their symbols.
fn schema(questions: &[Question]) -> String {
    let mut lines = Vec::new();
    for question in questions {
        lines.push(format!("{}: {}", question.name, question.instructions));
        for (symbol, candidate) in SYMBOLS.chars().zip(&question.candidates) {
            lines.push(format!(
                "    {symbol} = {}: {}",
                candidate.value, candidate.description
            ));
        }
    }
    lines.join("\n")
}

/// An image's grid of language-model tokens.
fn merged(grid: Grid) -> Grid {
    Grid {
        rows: grid.rows / MERGE,
        columns: grid.columns / MERGE,
    }
}

/// The rows of each image's tokens: its run of image tokens, in order.
fn image_rows(ids: &[u32], merged: &[Grid]) -> Result<Vec<(Range<usize>, Grid)>, Error> {
    let mut rows = Vec::new();
    let mut row = 0;
    while row < ids.len() {
        let length = ids[row..]
            .iter()
            .take_while(|&&id| id == IMAGE_TOKEN)
            .count();
        if length > 0 {
            rows.push(row..row + length);
        }
        row += length.max(1);
    }
    let counts = merged.iter().map(|grid| grid.patches());
    if rows.len() != merged.len() || rows.iter().map(Range::len).ne(counts) {
        return Err(Error::Request(
            "the request's text holds an image token".into(),
        ));
    }
    Ok(rows.into_iter().zip(merged.iter().copied()).collect())
}

/// `text` without the leading and trailing characters Python's `str.isspace` matches: Unicode's
/// whitespace and the information separators U+001C–U+001F.
fn strip(text: &str) -> &str {
    text.trim_matches(|character: char| {
        character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
    })
}
