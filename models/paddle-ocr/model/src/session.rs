//! Reading an image (`blocks::reading`): `start` cuts it into patches and builds the task's
//! prompt; then each `step` does one stage — the vision tower and projector, the prompt through
//! the language model, or one more token of the answer. The stages also run one by one, on given
//! inputs, for checks against the reference.

use blocks::lm::{self, Lm};
use blocks::reading::{self, VisionLanguage};
pub use blocks::reading::{Plan, Prediction, Progress};
use blocks::vit::Grid;
use blocks::vit::merger;
use blocks::vit::preprocessing::{Patches, Preprocessing};
use blocks::vit::tower::{self, Tower};
use bundle::Store;
use media::image::Pixels;
use team::{Ctx, SharedBuf};

use crate::Error;
use crate::image::{self, PREPROCESSING};
use crate::model::{LANGUAGE, Model, PROJECTOR, TOWER};
use crate::pipeline::Stage;
use crate::prompt::{END, Prompt, Task};

/// The longest answer, in tokens.
pub const MAX_TOKENS: usize = 2048;
/// Prompt rows besides the image's: the chat template and the longest instruction.
const TEXT_ROWS: usize = 32;
/// Prompt rows per call of the language model's stack.
const PREFILL_ROWS: usize = 128;

pub type Engine = reading::Engine<Model>;
pub type Request = reading::Request<Task>;

impl VisionLanguage for Model {
    type Task = Task;
    type Prompt = Prompt;
    type Stage = Stage;
    type Buffers = merger::Buffers;

    const SEE: Stage = Stage::Encode;
    const PREFILL: Stage = Stage::Prefill;
    const DECODE: Stage = Stage::Decode;
    const ENDS: &'static [u32] = &[END];
    const MAX_TOKENS: usize = MAX_TOKENS;
    const TEXT_ROWS: usize = TEXT_ROWS;
    const PREFILL_ROWS: usize = PREFILL_ROWS;
    const PREPROCESSING: Preprocessing = PREPROCESSING;
    const TOWER: tower::Shape = TOWER;
    const LANGUAGE: lm::Shape = LANGUAGE;

    fn load(store: &mut Store) -> Result<Self, Error> {
        Model::load(store)
    }

    fn buffers() -> merger::Buffers {
        merger::Buffers::new(&PROJECTOR, PREPROCESSING.max_patches())
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

    /// The projector over the tower's output.
    fn tokens(&self, ctx: &Ctx, tower: &tower::Buffers, buffers: &merger::Buffers, grid: Grid) {
        self.projector.forward(ctx, tower, buffers, grid);
    }

    fn image_tokens(buffers: &merger::Buffers) -> &SharedBuf {
        &buffers.tokens
    }
}
