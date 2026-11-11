//! The activations of one Wan DiT call, and the context its cross-attentions read.

use team::{PerThread, SharedBuf};

use super::{BLOCK_CHUNKS, HEAD_CHUNKS, Shape};
use crate::attention;
use crate::row_work::RowWork;
use crate::time_embedding::TimeBuffers;

/// Activations for videos of up to `tokens` tokens.
pub struct Buffers {
    pub(super) tokens: usize,
    /// The latents in, laid out as `Video` says.
    pub latents: SharedBuf,
    /// The velocity out, laid out as `Video` says.
    pub velocity: SharedBuf,
    pub(super) hidden: SharedBuf,
    pub(super) normed: SharedBuf,
    pub(super) attention: SharedBuf,
    pub(super) qkv: SharedBuf,
    /// The MLP's inner activations, then the head's output.
    pub(super) inner: SharedBuf,
    pub(super) patched: SharedBuf,
    pub(super) time: TimeBuffers,
    /// The six modulations the time projects to, before each block's own are added.
    pub(super) projected: SharedBuf,
    pub(super) modulations: SharedBuf,
    pub(super) head_modulation: SharedBuf,
    pub(super) text_input: SharedBuf,
    pub(super) text_hidden: SharedBuf,
    pub(super) text_embedded: SharedBuf,
    pub(super) scratch: PerThread<attention::Scratch>,
    pub(super) rows: RowWork,
}

/// Every block's cross-attention keys and values of one branch's text.
pub struct Context {
    pub(super) layers: Vec<SharedBuf>,
}

impl Shape {
    /// Activations for videos of up to `tokens` tokens.
    pub fn buffers(&self, tokens: usize, threads: usize) -> Buffers {
        let new = SharedBuf::new;
        let patch_values = self.patch_values();
        let text_values = self.text_tokens * self.hidden;
        Buffers {
            tokens,
            latents: new(tokens * patch_values),
            velocity: new(tokens * patch_values),
            hidden: new(tokens * self.hidden),
            normed: new(tokens * self.hidden),
            attention: new(tokens * self.hidden),
            qkv: new(tokens * 3 * self.hidden),
            inner: new(tokens * self.feed_forward.max(patch_values)),
            patched: new(tokens * patch_values),
            time: TimeBuffers::new(self.frequencies, self.hidden),
            projected: new(BLOCK_CHUNKS * self.hidden),
            modulations: new(self.depth * BLOCK_CHUNKS * self.hidden),
            head_modulation: new(HEAD_CHUNKS * self.hidden),
            text_input: new(self.text_tokens * self.text),
            text_hidden: new(text_values),
            text_embedded: new(text_values),
            scratch: PerThread::new(threads, attention::Scratch::default),
            rows: RowWork::new(threads),
        }
    }

    /// One branch's cross-attention keys and values: `depth` buffers of `text_tokens × 2·hidden`.
    pub fn context(&self) -> Context {
        let layer = || SharedBuf::new(self.text_tokens * 2 * self.hidden);
        Context {
            layers: (0..self.depth).map(|_| layer()).collect(),
        }
    }
}
