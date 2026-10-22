//! Reading an image (`blocks::reading`): `start` cuts it into patches and builds the task's
//! prompt; then each `step` does one stage — the vision tower and the visual vocabulary, the
//! prompt through the language model, or one more token of the answer. The stages also run one
//! by one, on given inputs, for checks against the reference.

use blocks::lm::{self, Lm};
use blocks::reading::{self, VisionLanguage};
pub use blocks::reading::{Plan, Prediction, Progress};
use blocks::vit::Grid;
use blocks::vit::preprocessing::{Patches, Preprocessing};
use blocks::vit::tower::{self, Tower};
use bundle::Store;
use media::image::Pixels;
use team::{Ctx, SharedBuf};

use crate::Error;
use crate::image::{self, PREPROCESSING};
use crate::model::{HIDDEN, IMAGE_END, IMAGE_START, LANGUAGE, Model, TOWER};
use crate::pipeline::Stage;
use crate::prompt::{ENDS, Prompt, Task};
use crate::vision;

/// The longest answer, in tokens.
pub const MAX_TOKENS: usize = 4096;
/// Prompt rows besides the image's tokens: the chat template, the image's indicators and the
/// longest instruction.
const TEXT_ROWS: usize = 32;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 128;

pub type Engine = reading::Engine<Model>;
pub type Request = reading::Request<Task>;

impl VisionLanguage for Model {
    type Task = Task;
    type Prompt = Prompt;
    type Stage = Stage;
    type Buffers = vision::Buffers;

    const SEE: Stage = Stage::Tokenize;
    const PREFILL: Stage = Stage::Prefill;
    const DECODE: Stage = Stage::Decode;
    const ENDS: &'static [u32] = &ENDS;
    const MAX_TOKENS: usize = MAX_TOKENS;
    const TEXT_ROWS: usize = TEXT_ROWS;
    const PREFILL_ROWS: usize = PREFILL_ROWS;
    const PREPROCESSING: Preprocessing = PREPROCESSING;
    const TOWER: tower::Shape = TOWER;
    const LANGUAGE: lm::Shape = LANGUAGE;

    fn load(store: &mut Store) -> Result<Self, Error> {
        Model::load(store)
    }

    fn buffers() -> vision::Buffers {
        vision::Buffers::new()
    }

    fn tower(&self) -> &Tower {
        &self.vision
    }

    fn language(&self) -> &Lm {
        &self.language
    }

    fn compose(&self, task: Task, grid: Grid) -> Result<Prompt, Error> {
        Prompt::new(&self.tokenizer, task, grid)
    }

    fn text(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer.decode(ids)
    }

    fn patches(image: &Pixels) -> Result<Patches, Error> {
        image::patches(image)
    }

    /// The visual vocabulary over the tower's output.
    fn tokens(&self, ctx: &Ctx, tower: &tower::Buffers, buffers: &vision::Buffers, grid: Grid) {
        self.visual.forward(ctx, tower, buffers, grid);
    }

    fn image_tokens(buffers: &vision::Buffers) -> &SharedBuf {
        &buffers.tokens
    }

    /// The image's indicators in the rows just before and after its tokens.
    fn frame_image(&self, prompt: &Prompt, embeddings: &mut [f32]) {
        let indicators = self.visual.indicators.matrix();
        let rows = [
            (prompt.image.start - 1, IMAGE_START),
            (prompt.image.end, IMAGE_END),
        ];
        for (row, indicator) in rows {
            let at = row * HIDDEN..(row + 1) * HIDDEN;
            indicators.dequantize_row(indicator, &mut embeddings[at]);
        }
    }
}
