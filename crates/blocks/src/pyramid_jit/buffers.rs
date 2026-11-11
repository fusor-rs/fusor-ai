//! The activations of one P-JiT call, and the text one branch conditions on.

use team::{PerThread, SharedBuf};

use super::{BLOCK_CHUNKS, HEAD_CHUNKS, Shape};
use crate::attention;
use crate::row_work::RowWork;
use crate::time_embedding::TimeBuffers;

/// Activations for images of up to `tokens` patches, with up to `text_tokens` text rows after
/// them.
pub struct Buffers {
    pub(super) tokens: usize,
    /// The noisy image in, pixel-major, channels last.
    pub image: SharedBuf,
    /// The clean image predicted, laid out as `image`.
    pub prediction: SharedBuf,
    /// The residual stream: the image's tokens, then the text's.
    pub(super) hidden: SharedBuf,
    /// The image's tokens after the image refiners.
    pub(super) refined: SharedBuf,
    pub(super) normed: SharedBuf,
    pub(super) qkv: SharedBuf,
    pub(super) attention: SharedBuf,
    /// The feed-forward's gates and values, and its gated values.
    pub(super) inner: SharedBuf,
    pub(super) gated: SharedBuf,
    /// The image's patches, then the head's output.
    pub(super) patched: SharedBuf,
    pub(super) bottleneck: SharedBuf,
    pub(super) text_input: SharedBuf,
    pub(super) time: TimeBuffers,
    pub(super) roots: [SharedBuf; 2],
    pub(super) modulations: SharedBuf,
    pub(super) head_modulation: SharedBuf,
    pub(super) scratch: PerThread<attention::Scratch>,
    pub(super) rows: RowWork,
}

/// One branch's text: its states through the text projection, `rows` of `hidden`.
pub struct Text {
    pub(super) states: SharedBuf,
    pub(super) rows: usize,
}

impl Shape {
    /// Activations for images of up to `tokens` patches.
    pub fn buffers(&self, tokens: usize, threads: usize) -> Buffers {
        let new = SharedBuf::new;
        let patch_values = self.patch_values();
        let rows = tokens + self.text_tokens;
        let feed_forward = self.feed_forward.max(self.refiner_feed_forward);
        let blocks = 2 * self.refiners + self.depth;
        Buffers {
            tokens,
            image: new(tokens * patch_values),
            prediction: new(tokens * patch_values),
            hidden: new(rows * self.hidden),
            refined: new(tokens * self.hidden),
            normed: new(rows * self.hidden),
            qkv: new(rows * self.qkv_gate()),
            attention: new(rows * self.hidden),
            inner: new(rows * 2 * feed_forward),
            gated: new(rows * feed_forward),
            patched: new(tokens * patch_values),
            bottleneck: new(tokens * self.bottleneck),
            text_input: new(self.text_tokens * self.text),
            time: TimeBuffers::new(self.frequencies, self.hidden),
            roots: [new(self.rank), new(self.rank)],
            modulations: new(blocks * BLOCK_CHUNKS * self.hidden),
            head_modulation: new(HEAD_CHUNKS * self.hidden),
            scratch: PerThread::new(threads, attention::Scratch::default),
            rows: RowWork::new(threads),
        }
    }

    /// A branch's text, with room for `text_tokens` rows.
    pub fn text(&self) -> Text {
        Text {
            states: SharedBuf::new(self.text_tokens * self.hidden),
            rows: 0,
        }
    }
}
