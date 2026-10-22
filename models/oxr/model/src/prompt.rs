//! The prompt an image is read with: the chat template's user turn, holding the image's tokens
//! between its indicators and then the task's instruction, and the assistant's turn to answer
//! in.

use std::ops::Range;

use blocks::reading;
use blocks::rotary::Position;
use blocks::vit::Grid;
use bundle::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::model::MERGE;

/// The tokens an answer ends with: `<|endoftext|>` and `<|im_end|>`.
pub const ENDS: [u32; 2] = [151_643, 151_645];
/// The ids the image's rows hold in `Prompt::ids` (vLLM's: `<|vision_start|>`, `<|image_pad|>`
/// per token, `<|vision_end|>`); their embeddings are the visual ones, never these tokens'.
const IMAGE_START: u32 = 151_652;
const IMAGE_TOKEN: u32 = 151_655;
const IMAGE_END: u32 = 151_653;
const BEFORE_IMAGE: &str = "<|im_start|>user\n";
const AFTER_INSTRUCTION: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// What to read off the image: each task is an instruction SoMark's pipeline sends the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// A whole page, in reading order: Markdown, formulas in LaTeX, tables in HTML.
    Page,
    /// A region's text.
    Text,
    /// A table, in HTML.
    Table,
    /// A formula, in LaTeX.
    Formula,
    /// Code, as a Markdown code block.
    Code,
    /// A chemical structure, in SMILES.
    Chemistry,
}

impl Task {
    pub const ALL: [Task; 6] = [
        Task::Page,
        Task::Text,
        Task::Table,
        Task::Formula,
        Task::Code,
        Task::Chemistry,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Task::Page => "page",
            Task::Text => "text",
            Task::Table => "table",
            Task::Formula => "formula",
            Task::Code => "code",
            Task::Chemistry => "chemistry",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Task::Page => "Recognize the whole page content of the image.",
            Task::Text => "Recognize the content of the image.",
            Task::Table => "Recognize the table of the image into html format.",
            Task::Formula => "Recognize the formula of the image into KaTex format.",
            Task::Code => "Recognize the code of the image into markdown code format.",
            Task::Chemistry => "Recognize the chemical structure of the image into SMILES.",
        }
    }
}

pub struct Prompt {
    pub ids: Vec<u32>,
    /// The rows of the image's tokens; its start and end indicators are the rows just before
    /// and just after them.
    pub image: Range<usize>,
}

impl Prompt {
    /// The prompt for an image of `grid` patches. The text before and after the image is
    /// tokenized apart, as the reference splits it at the image.
    pub fn new(tokenizer: &Tokenizer, task: Task, grid: Grid) -> Result<Self, Error> {
        let tokens = grid.patches() / (MERGE * MERGE);
        let mut ids = tokenizer.encode(BEFORE_IMAGE)?;
        ids.push(IMAGE_START);
        let start = ids.len();
        ids.resize(start + tokens, IMAGE_TOKEN);
        ids.push(IMAGE_END);
        let rest = format!("{}{AFTER_INSTRUCTION}", task.instruction());
        ids.extend(tokenizer.encode(&rest)?);
        Ok(Self {
            ids,
            image: start..start + tokens,
        })
    }
}

impl reading::Prompt for Prompt {
    fn ids(&self) -> &[u32] {
        &self.ids
    }

    fn image(&self) -> Range<usize> {
        self.image.clone()
    }

    /// Qwen3's rows turn by their sequence positions.
    fn rotary(&self, _rows: Range<usize>) -> Option<Vec<Position>> {
        None
    }
}
