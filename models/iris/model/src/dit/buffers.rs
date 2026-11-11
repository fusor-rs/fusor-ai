//! The activations of one transformer call, and each thread's scratch.

use blocks::attention;
use kernels::conv::{Patches, project_rows};
use kernels::{Linear, QuantizedRows};
use team::{Ctx, PerThread, SharedBuf};

use super::pixel::{PIXEL_ATTENTION, PIXEL_QKV};
use super::{BLOCKS, MODULATION, PAIRS, PATCH_PIXELS, QKV, SINUSOID_PAIRS, TIME_FREQUENCIES};
use crate::model::{
    FEED_FORWARD, HIDDEN, IMAGE_CHANNELS, MAX_PIXELS, MAX_SIDE, PIXEL_HIDDEN, TEXT_ROWS,
};

/// Activations for images of up to `MAX_PIXELS` pixels.
pub(crate) struct Buffers {
    /// Each pixel's RGB (or depth's input), pixel-major: in.
    pub(crate) image: SharedBuf,
    /// Each pixel's predicted velocity, pixel-major: out.
    pub(crate) velocity: SharedBuf,
    /// The image cut into patches, a patch's pixels row-major, a pixel's channels contiguous.
    pub(super) grouped: SharedBuf,
    pub(super) hidden: SharedBuf,
    pub(super) normed: SharedBuf,
    pub(super) gate: SharedBuf,
    pub(super) attention: SharedBuf,
    pub(super) branch: SharedBuf,
    pub(super) qkv: SharedBuf,
    pub(super) inner: SharedBuf,
    pub(super) sinusoid: SharedBuf,
    pub(super) time_hidden: SharedBuf,
    pub(super) time_embedding: SharedBuf,
    pub(super) condition: SharedBuf,
    /// The image core's modulation of the timestep, then the text core's.
    pub(super) cores: SharedBuf,
    /// Each block's two streams' modulations at the timestep.
    pub(super) modulations: SharedBuf,
    pub(super) image_cos: SharedBuf,
    pub(super) image_sin: SharedBuf,
    pub(super) pixels: SharedBuf,
    pub(super) pixels_normed: SharedBuf,
    pub(super) compact: SharedBuf,
    pub(super) pixel_attention: SharedBuf,
    pub(super) pixel_qkv: SharedBuf,
    /// The pixel embedding's sine and cosine tables along the columns, then along the rows.
    pub(super) columns: SharedBuf,
    pub(super) rows: SharedBuf,
    pub(super) patches: PerThread<Patches>,
    pub(super) scratch: PerThread<attention::Scratch>,
    /// A thread's row on its way through the heads' norms.
    pub(super) heads: PerThread<Vec<f32>>,
    pub(super) tail: PerThread<TailScratch>,
}

/// A thread's patches on their way through the pixel head's tail.
#[derive(Default)]
pub(super) struct TailScratch {
    pub(super) quant: QuantizedRows,
    pub(super) modulations: Vec<f32>,
    pub(super) expanded: Vec<f32>,
    pub(super) normed: Vec<f32>,
    pub(super) inner: Vec<f32>,
    pub(super) out: Vec<f32>,
}

impl Buffers {
    pub(crate) fn new(threads: usize) -> Self {
        let new = SharedBuf::new;
        let patches = MAX_PIXELS / PATCH_PIXELS;
        let rows = TEXT_ROWS + patches;
        let values = MAX_PIXELS * IMAGE_CHANNELS;
        Self {
            image: new(values),
            velocity: new(values),
            grouped: new(values),
            hidden: new(rows * HIDDEN),
            normed: new(rows * HIDDEN),
            gate: new(rows * HIDDEN),
            attention: new(rows * HIDDEN),
            branch: new(rows * HIDDEN),
            qkv: new(rows * QKV),
            inner: new(rows * FEED_FORWARD),
            sinusoid: new(TIME_FREQUENCIES),
            time_hidden: new(HIDDEN),
            time_embedding: new(HIDDEN),
            condition: new(HIDDEN),
            cores: new(2 * MODULATION),
            modulations: new(BLOCKS * 2 * MODULATION),
            image_cos: new(patches * PAIRS),
            image_sin: new(patches * PAIRS),
            pixels: new(MAX_PIXELS * PIXEL_HIDDEN),
            pixels_normed: new(MAX_PIXELS * PIXEL_HIDDEN),
            compact: new(patches * PIXEL_ATTENTION),
            pixel_attention: new(patches * PIXEL_ATTENTION),
            pixel_qkv: new(patches * PIXEL_QKV),
            columns: new(MAX_SIDE * 2 * SINUSOID_PAIRS),
            rows: new(MAX_SIDE * 2 * SINUSOID_PAIRS),
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, attention::Scratch::default),
            heads: PerThread::new(threads, Vec::new),
            tail: PerThread::new(threads, TailScratch::default),
        }
    }

    /// `store(row, col, value)` of `linear` over the first `rows` rows of `input`, then a
    /// barrier.
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
