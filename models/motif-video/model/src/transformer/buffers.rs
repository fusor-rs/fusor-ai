//! The activations of one transformer call, and the context rows its blocks read.

use blocks::mmdit::{self, Capacity};
use kernels::Linear;
use kernels::conv::{Patches, project_rows};
use team::{Ctx, PerThread, SharedBuf};

use crate::model::{
    DIT_SHAPE, DUAL_BLOCKS, IMAGE_HIDDEN, IMAGE_TOKENS, INPUT_CHANNELS, LATENT_CHANNELS, PATCH,
    SINGLE_BLOCKS, TEXT_TOKENS, TIME_FREQUENCIES,
};
use crate::session::Branch;

/// The rows a call reads beside the video's: the image's and the text's.
pub(super) const CONTEXT_ROWS: usize = IMAGE_TOKENS + TEXT_TOKENS;
/// The branches of guidance: the prompt's and the negative prompt's.
pub(crate) const BRANCHES: usize = 2;
/// Values of a patch of the transformer's input, and of its output.
pub(super) const PATCHED_INPUT: usize = PATCH * PATCH * INPUT_CHANNELS;
pub(super) const PATCHED_OUTPUT: usize = PATCH * PATCH * LATENT_CHANNELS;

/// Activations for videos of up to a number of tokens.
pub(crate) struct Buffers {
    pub(super) tokens: usize,
    /// Each latent pixel's transformer input (`INPUT_CHANNELS`): in.
    pub(crate) input: SharedBuf,
    /// Each latent pixel's predicted velocity (`LATENT_CHANNELS`): out.
    pub(crate) velocity: SharedBuf,
    pub(super) patched: SharedBuf,
    /// The patch embeddings, which the decoder blocks read again.
    pub(super) stream_input: SharedBuf,
    pub(super) sinusoid: SharedBuf,
    pub(super) time_hidden: SharedBuf,
    pub(super) time: SharedBuf,
    /// Each block's modulations at the timestep: the dual-stream blocks', then the single's.
    pub(super) modulations: Vec<SharedBuf>,
    pub(super) head_modulation: SharedBuf,
    pub(super) text_hidden: SharedBuf,
    pub(super) image_normed: SharedBuf,
    pub(super) image_inner: SharedBuf,
    pub(super) head_output: SharedBuf,
    pub(super) blocks: mmdit::Buffers,
    patches: PerThread<Patches>,
    pub(super) row: PerThread<Vec<f32>>,
}

impl Buffers {
    /// Activations for videos of up to `tokens` tokens.
    pub(crate) fn new(tokens: usize, threads: usize) -> Self {
        let new = SharedBuf::new;
        let hidden = DIT_SHAPE.hidden;
        let pixels = tokens * PATCH * PATCH;
        let capacity = Capacity {
            rows: (tokens + CONTEXT_ROWS).max(2 * tokens),
            stream: tokens,
            text: TEXT_TOKENS,
        };
        let dual = (0..DUAL_BLOCKS).map(|_| new(DIT_SHAPE.dual_modulations()));
        let single = (0..SINGLE_BLOCKS).map(|_| new(DIT_SHAPE.single_modulations()));
        Self {
            tokens,
            input: new(pixels * INPUT_CHANNELS),
            velocity: new(pixels * LATENT_CHANNELS),
            patched: new(tokens * PATCHED_INPUT),
            stream_input: new(tokens * hidden),
            sinusoid: new(TIME_FREQUENCIES),
            time_hidden: new(hidden),
            time: new(hidden),
            modulations: dual.chain(single).collect(),
            head_modulation: new(2 * hidden),
            text_hidden: new(TEXT_TOKENS * hidden),
            image_normed: new(IMAGE_TOKENS * IMAGE_HIDDEN),
            image_inner: new(IMAGE_TOKENS * IMAGE_HIDDEN),
            head_output: new(tokens * PATCHED_OUTPUT),
            blocks: DIT_SHAPE.buffers(capacity, threads),
            patches: PerThread::new(threads, Patches::default),
            row: PerThread::new(threads, Vec::new),
        }
    }

    /// `store(row, col, value)` of `linear` over the first `rows` rows of `input`, then a barrier.
    pub(super) fn project(
        &self,
        ctx: &Ctx,
        linear: &Linear,
        (input, rows): (&[f32], usize),
        store: impl FnMut(usize, usize, f32),
    ) {
        project_rows(ctx, linear, (input, rows), self.patches.get(ctx), store);
        ctx.barrier();
    }
}

/// The context rows a forward reads: the image's projection (when there is a first image) and
/// each branch's projected text.
pub(crate) struct Contexts {
    pub(super) image: SharedBuf,
    image_rows: usize,
    pub(super) text: [SharedBuf; BRANCHES],
    text_rows: [usize; BRANCHES],
}

impl Contexts {
    pub(crate) fn new() -> Self {
        let hidden = DIT_SHAPE.hidden;
        Self {
            image: SharedBuf::new(IMAGE_TOKENS * hidden),
            image_rows: 0,
            text: std::array::from_fn(|_| SharedBuf::new(TEXT_TOKENS * hidden)),
            text_rows: [0; BRANCHES],
        }
    }

    pub(crate) fn set_image_rows(&mut self, rows: usize) {
        self.image_rows = rows;
    }

    pub(crate) fn set_text_rows(&mut self, branch: Branch, rows: usize) {
        self.text_rows[branch as usize] = rows;
    }

    pub(super) fn image_rows(&self) -> usize {
        self.image_rows
    }

    pub(super) fn text_rows(&self, branch: Branch) -> usize {
        self.text_rows[branch as usize]
    }
}
