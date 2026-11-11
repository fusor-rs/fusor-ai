//! Iris's diffusion transformer, on pixels: an image's 16 × 16 patches (`PATCH`) and the adapted
//! text's 300 rows in one residual stream, text first; the timestep's sinusoids (period 10)
//! through an MLP, two modulation cores (image, text) and each block's own bias into shifts,
//! scales and gates; 8 dual-stream blocks (text and image rows with their own weights) then 16
//! single-stream blocks (one set), each a modulated RMSNorm, joint attention over every row
//! (grouped-query, 20 query heads over 5, per-head RMSNorm of queries and keys, the text's 1-D and
//! the image's isotropic 2-D rotary embeddings, a sigmoid gate on the output), a SwiGLU MLP, and
//! RMSNorms on both branches (sandwich); then the image rows, fused with the timestep, condition a
//! pixel head of 4 blocks over 16 channels per pixel whose attention runs once per patch on the
//! patch's pixels compressed to 1280; a last RMSNorm and linear give each pixel's velocity.
//! Attention runs on the int8 kernels (`blocks::attention`).
//!
//! Tensors under `dit`: `patch`, `time.{0,1}`, `modulation.{image,text}`, `positions`,
//! `adapter.*` (`adapter`), `blocks.{i}.{text,image}.*` (`i < 8`) and `blocks.{i}.shared.*` (each
//! stream's `norm1`, `qkv`, `q_norm`, `k_norm`, `gate`, `out`, `post_attention`, `norm2`,
//! `ff_gate`, `ff_up`, `ff_down`, `post_ff`, `bias`), `pixel.embed`, `pixel.blocks.{j}.{norm1,
//! compress, qkv, q_norm, k_norm, out, expand, modulation, norm2, ff1, ff2}`, `pixel.final_norm`,
//! `pixel.final`.
//!
//! Every method runs inside a team run. `time` computes a timestep's modulations once; `forward`
//! then runs the transformer for one branch's adapted text.

mod block;
mod buffers;
mod embedding;
mod pixel;
mod rope;

use std::ops::Range;

use blocks::mmdit::normalize_heads;
use blocks::timestep_sinusoid;
use blocks::video::{Patching, Video};
use bundle::{Scope, Store, load_pair};
use kernels::conv::project_rows;
use kernels::ops::{rope_interleaved, silu};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use self::block::{Block, Streams};
pub(crate) use self::buffers::Buffers;
use self::pixel::PixelHead;
use self::rope::TextAngles;
use crate::adapter::Adapter;
use crate::model::{
    DIT, DUAL_BLOCKS, HIDDEN, IMAGE_CHANNELS, NORM_EPS, PATCH, PATCH_PIXELS, SINGLE_BLOCKS,
    TEXT_ROWS, TIMESTEP_PERIOD, Variant,
};
use crate::{Error, Size};

const HEADS: usize = 20;
const KV_HEADS: usize = 5;
const HEAD_DIM: usize = 128;
/// The width of the queries.
const QUERIES: usize = HEADS * HEAD_DIM;
/// Columns of the fused projection: the queries, then the keys, then the values.
const QKV: usize = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
/// Rotary angles per row.
const PAIRS: usize = HEAD_DIM / 2;
const ROPE_BASE: f32 = 10_000.0;
/// The image's rotary positions span `0..ROPE_SPAN` along their longer side.
const ROPE_SPAN: f32 = 16.0;
const BLOCKS: usize = DUAL_BLOCKS + SINGLE_BLOCKS;
const TIME_FREQUENCIES: usize = 256;
/// Chunks of a stream's modulation: a shift, a scale and a gate for each of the attention and
/// the MLP.
const MODULATION_CHUNKS: usize = 6;
const MODULATION: usize = MODULATION_CHUNKS * HIDDEN;
/// Frequencies of the pixel embedding's table along one axis.
const SINUSOID_PAIRS: usize = 4;

pub(crate) struct Dit {
    patch: Linear,
    time: [Linear; 2],
    image_core: Linear,
    text_core: Linear,
    pub(crate) adapter: Adapter,
    blocks: Vec<Block>,
    pixel: PixelHead,
    text_angles: TextAngles,
}

/// What one forward shares among its phases.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    patching: Patching,
    text_angles: &'call TextAngles,
}

impl Pass<'_> {
    fn patches(&self) -> usize {
        self.patching.tokens()
    }

    /// The text rows, then the patches'.
    fn rows(&self) -> usize {
        TEXT_ROWS + self.patches()
    }

    /// `store(row, col, value)` of `linear` over `range`'s rows of `input`, counting rows from
    /// the range's first; no barrier.
    fn product(
        &self,
        linear: &Linear,
        (input, range): (&SharedBuf, Range<usize>),
        mut store: impl FnMut(usize, usize, f32),
    ) {
        let width = linear.matrix().cols;
        let rows = (&input.read()[range.start * width..], range.len());
        let patches = self.buffers.patches.get(self.ctx);
        project_rows(self.ctx, linear, rows, patches, |row, col, value| {
            store(range.start + row, col, value);
        });
    }
}

impl Dit {
    pub(crate) fn load(store: &mut Store, variant: Variant) -> Result<Self, Error> {
        let time = load_pair(
            store,
            &format!("{DIT}.time"),
            [[HIDDEN, TIME_FREQUENCIES], [HIDDEN; 2]],
        )?;
        let blocks = (0..BLOCKS)
            .map(|index| {
                let streams = if index < DUAL_BLOCKS {
                    Streams::Dual
                } else {
                    Streams::Single
                };
                Block::load(store, (index, streams))
            })
            .collect::<Result<_, _>>()?;
        let adapter = Adapter::load(
            store,
            &format!("{DIT}.adapter"),
            &format!("{DIT}.positions"),
        )?;
        let pixel = PixelHead::load(store, variant.outputs())?;
        let mut scope = Scope::new(store, DIT);
        Ok(Self {
            patch: scope.load("patch", [HIDDEN, PATCH_PIXELS * IMAGE_CHANNELS])?,
            image_core: scope.load("modulation.image", [MODULATION, HIDDEN])?,
            text_core: scope.load("modulation.text", [MODULATION, HIDDEN])?,
            text_angles: TextAngles::new(),
            time,
            adapter,
            blocks,
            pixel,
        })
    }

    /// The timestep's embedding and every block's modulations at model time `timestep`, into
    /// `buffers`.
    pub(crate) fn time(&self, ctx: &Ctx, buffers: &Buffers, timestep: f32) {
        if ctx.is_first() {
            let frequencies = buffers.sinusoid.part(0..TIME_FREQUENCIES);
            timestep_sinusoid(timestep, TIMESTEP_PERIOD, frequencies);
        }
        ctx.barrier();
        let sinusoid = (buffers.sinusoid.read(), 1);
        buffers.project(ctx, &self.time[0], sinusoid, |_, col, value| {
            buffers.time_hidden.set(col, silu(value));
        });
        let hidden = (buffers.time_hidden.read(), 1);
        buffers.project(ctx, &self.time[1], hidden, |_, col, value| {
            buffers.time_embedding.set(col, value);
            buffers.condition.set(col, silu(value));
        });
        let condition = (buffers.condition.read(), 1);
        buffers.project(ctx, &self.image_core, condition, |_, col, value| {
            buffers.cores.set(col, value);
        });
        buffers.project(ctx, &self.text_core, condition, |_, col, value| {
            buffers.cores.set(MODULATION + col, value);
        });
        for block in &self.blocks {
            block.modulate(ctx, buffers.cores.read(), &buffers.modulations);
        }
        ctx.barrier();
    }

    /// `buffers.velocity` (pixel-major, `outputs` per pixel) = the prediction for
    /// `buffers.image` (pixel-major RGB of `size`) under `context` (`TEXT_ROWS × HIDDEN`) at the
    /// last `time`.
    pub(crate) fn forward(&self, ctx: &Ctx, buffers: &Buffers, context: &SharedBuf, size: Size) {
        let pass = Pass {
            ctx,
            buffers,
            patching: Patching {
                video: Video {
                    frames: 1,
                    height: size.height,
                    width: size.width,
                },
                patch: PATCH,
            },
            text_angles: &self.text_angles,
        };
        embedding::group(&pass);
        self.embed_tokens(&pass, context);
        for block in &self.blocks {
            block.forward(&pass);
        }
        condition_pixels(&pass);
        self.pixel.forward(&pass);
    }

    /// `hidden` = the text rows, then the patches' embeddings; then a barrier.
    fn embed_tokens(&self, pass: &Pass, context: &SharedBuf) {
        let Pass { ctx, buffers, .. } = pass;
        for row in ctx.split(TEXT_ROWS, 1) {
            let range = row * HIDDEN..(row + 1) * HIDDEN;
            let text = buffers.hidden.part(range.clone());
            text.copy_from_slice(&context.read()[range]);
        }
        let grouped = (buffers.grouped.read(), pass.patches());
        buffers.project(ctx, &self.patch, grouped, |row, col, value| {
            buffers.hidden.set((TEXT_ROWS + row) * HIDDEN + col, value);
        });
    }
}

/// `normed` rows `0..patches` = the pixel head's condition: the patches' trunk states fused with
/// the timestep's embedding, `silu(t_emb + s)`; then a barrier.
fn condition_pixels(pass: &Pass) {
    let buffers = pass.buffers;
    let time = buffers.time_embedding.read();
    for patch in pass.ctx.split(pass.patches(), 1) {
        let state = &buffers.hidden.read()[(TEXT_ROWS + patch) * HIDDEN..][..HIDDEN];
        let condition = buffers.normed.part(patch * HIDDEN..(patch + 1) * HIDDEN);
        for ((value, time), state) in condition.iter_mut().zip(time).zip(state) {
            *value = silu(time + state);
        }
    }
    pass.ctx.barrier();
}

/// Which half of a block a norm, a modulation or a residual belongs to.
#[derive(Clone, Copy)]
enum Half {
    Attention,
    FeedForward,
}

/// A row's query and key heads (`heads`: the `queries` values, then the keys) RMS-normalized by
/// the queries' and the keys' weights and turned by the row's rotary angles, through `scratch`.
fn turn_heads(
    heads: &mut [f32],
    queries: usize,
    (query_norm, key_norm): (&Vector, &Vector),
    (cos, sin): (&[f32], &[f32]),
    scratch: &mut Vec<f32>,
) {
    let (query_heads, key_heads) = heads.split_at_mut(queries);
    normalize_heads(query_heads, (query_norm, NORM_EPS), scratch);
    normalize_heads(key_heads, (key_norm, NORM_EPS), scratch);
    for head in heads.as_chunks_mut::<HEAD_DIM>().0 {
        rope_interleaved(head, cos, sin);
    }
}
