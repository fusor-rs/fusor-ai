//! The prompt an image is read with: the chat template's user turn, holding the image's tokens
//! and the task's instruction, then the assistant's turn to answer in; and the rotary position
//! of every row, the prompt's and the answer's.

use std::ops::Range;

use blocks::rotary::Position;
use blocks::vit::Grid;
use bundle::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::model::MERGE;

/// The token the answer ends with, `</s>`.
pub const END: u32 = 2;
/// The token each merged patch of the image takes the place of, `<|IMAGE_PLACEHOLDER|>`.
const IMAGE: u32 = 100_295;
const BEFORE_IMAGE: &str = "<|begin_of_sentence|>User: <|IMAGE_START|>";
const AFTER_IMAGE: &str = "<|IMAGE_END|>";
const ASSISTANT: &str = "\nAssistant:\n";

/// What to read off the image: each task is an instruction the model was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// The text, in reading order.
    Ocr,
    /// A table, in OTSL cells (`<fcel>`, `<ecel>`, `<lcel>`, `<ucel>`, `<xcel>`, `<nl>`).
    Table,
    /// A formula, in LaTeX.
    Formula,
    /// A chart, as a table of its data.
    Chart,
    /// The text of a seal or stamp.
    Seal,
}

impl Task {
    pub const ALL: [Task; 5] = [
        Task::Ocr,
        Task::Table,
        Task::Formula,
        Task::Chart,
        Task::Seal,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Task::Ocr => "ocr",
            Task::Table => "table",
            Task::Formula => "formula",
            Task::Chart => "chart",
            Task::Seal => "seal",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Task::Ocr => "OCR:",
            Task::Table => "Table Recognition:",
            Task::Formula => "Formula Recognition:",
            Task::Chart => "Chart Recognition:",
            Task::Seal => "Seal Recognition:",
        }
    }
}

pub struct Prompt {
    pub ids: Vec<u32>,
    /// The rows of the image's merged patches.
    pub image: Range<usize>,
    /// The image's merged-patch grid.
    merged: Grid,
}

impl Prompt {
    pub fn new(tokenizer: &Tokenizer, task: Task, grid: Grid) -> Result<Self, Error> {
        let merged = Grid {
            rows: grid.rows / MERGE,
            columns: grid.columns / MERGE,
        };
        let mut ids = tokenizer.encode(BEFORE_IMAGE)?;
        let start = ids.len();
        ids.resize(start + merged.patches(), IMAGE);
        let rest = format!("{AFTER_IMAGE}{}{ASSISTANT}", task.instruction());
        ids.extend(tokenizer.encode(&rest)?);
        Ok(Self {
            ids,
            image: start..start + merged.patches(),
            merged,
        })
    }

    /// The rotary position of the row at sequence position `index`, in the prompt or after it:
    /// a text row's on every axis; a merged patch's at the image's start, down its row and
    /// across its column; the text after the image continues from past the image's furthest.
    pub fn position(&self, index: usize) -> Position {
        let Range { start, end } = self.image;
        if index < start {
            return Position::uniform(index);
        }
        if index < end {
            let patch = index - start;
            let columns = self.merged.columns;
            return Position {
                temporal: start,
                height: start + patch / columns,
                width: start + patch % columns,
            };
        }
        let resumed = start + self.merged.rows.max(self.merged.columns) + index - end;
        Position::uniform(resumed)
    }

    pub fn positions(&self) -> Vec<Position> {
        (0..self.ids.len())
            .map(|index| self.position(index))
            .collect()
    }
}
