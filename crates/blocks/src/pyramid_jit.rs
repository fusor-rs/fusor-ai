//! P-JiT (Linum's Pyramid-JiT: P-JiT 21K and the pyramid-jit release), a pixel-space diffusion
//! transformer predicting the clean image: an image's `patch × patch`-pixel patches (the image
//! padded with zeros to whole patches) through two linears around a bottleneck and `refiners`
//! blocks of their own; the text's states through two bias-free linears around a GELU-tanh and
//! `refiners` blocks of their own; then `depth` blocks over the image's tokens followed by the
//! text's; a modulated LayerNorm and a linear back to each patch's pixels, cropped to the image.
//!
//! A block: RMSNorm scaled by `1 + s₁`; attention whose queries and keys are RMS-normalized over
//! the whole width, the image's tokens turned by 3-D rotary embeddings (`Rotation::Axes`,
//! interleaved pairs) at their patch row and column times `rope_spatial_scale`, the text's left at
//! the identity, each head's output gated by the sigmoid of a logit of its own; RMSNorm, then
//! `x += tanh(g₁) · …`; the same around a SiLU-gated feed-forward with `s₂`, `g₂`. The scales and
//! gates come from the timestep: its sinusoids through an MLP (`e`), a SiLU and a linear to `rank`
//! values (one root for the image's blocks, one for the text refiners), then each block's own
//! linear. Attention runs on the int8 kernels (`attention`). Training's extra pyramid readouts are
//! not part of sampling.
//!
//! Tensors under a prefix `p`: the linears `p.patch.{0,1}`, `p.time.{0,1}`, `p.roots.{image,text}`,
//! `p.text.{0,1}`, `p.head.linear`, the LayerNorm `p.head.norm`, the vector `p.head.modulation`,
//! and per block `p.image_refiner.{i}`, `p.text_refiner.{i}`, `p.blocks.{i}` the vectors `norm1`,
//! `q_norm`, `k_norm`, `post_norm1`, `norm2`, `post_norm2` and the linears `qkv_gate` (queries,
//! keys, values, then one gate logit per head), `out`, `ff1` (the gates, then the values the gates
//! multiply), `ff2`, `modulation`.
//!
//! `condition` runs its own team run; every other method runs inside one. A request's text
//! conditions a `Text` once per branch; each step computes the timestep's modulations (`time`) and
//! the image's refined tokens (`embed_image`) once, then `forward` once per branch.

mod block;
mod buffers;
mod pixels;

use bundle::{Load, Scope, Store, load_each, load_pair};
use kernels::ops::gelu_tanh;
use kernels::{Linear, Vector};
use media::image::Size;
use team::{Ctx, Team};

use crate::norm::LayerNorm;
use crate::rotary::{Rotary, Rotation, check_axes};
use crate::scaled;
use crate::video::{Patching, Video};
use block::Block;
pub use buffers::{Buffers, Text};

/// The modulations of a block, `hidden` values each: the attention's norm scale and residual
/// gate, then the feed-forward's.
#[derive(Clone, Copy)]
enum Chunk {
    AttentionScale,
    AttentionGate,
    FeedForwardScale,
    FeedForwardGate,
}

const BLOCK_CHUNKS: usize = Chunk::FeedForwardGate as usize + 1;
/// The chunks that scale a norm (`1 +` them is its weight); the others gate a residual (`tanh`).
const BLOCK_SCALES: [usize; 2] = [
    Chunk::AttentionScale as usize,
    Chunk::FeedForwardScale as usize,
];

/// The modulations of the head, `hidden` values each.
#[derive(Clone, Copy)]
enum HeadChunk {
    Shift,
    Scale,
}

const HEAD_CHUNKS: usize = HeadChunk::Scale as usize + 1;
const HEAD_SCALES: [usize; 1] = [HeadChunk::Scale as usize];
/// The head's LayerNorm epsilon: PyTorch's default, which P-JiT's head keeps.
const HEAD_EPS: f32 = 1e-5;

/// The dimensions of one transformer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    /// Blocks over the image's tokens and the text's together.
    pub depth: usize,
    /// Width inside those blocks' SiLU-gated feed-forwards.
    pub feed_forward: usize,
    /// Blocks over the image's tokens alone, and as many over the text's alone, before them.
    pub refiners: usize,
    pub refiner_feed_forward: usize,
    /// Width of the timestep's conditioning each block's modulations are projected from.
    pub rank: usize,
    /// Width of a text state.
    pub text: usize,
    /// Most text states a branch holds.
    pub text_tokens: usize,
    /// Channels of a pixel, in and out.
    pub channels: usize,
    /// Pixels per patch along each side.
    pub patch: usize,
    /// Width between the patch embedding's two linears.
    pub bottleneck: usize,
    /// Width of the timestep's sinusoids.
    pub frequencies: usize,
    /// Rotary frequencies turned by a patch's frame (0 for an image), row and column.
    pub rope_sections: [usize; 3],
    pub rope_base: f32,
    /// What a patch's row and column are multiplied by before they turn the rotary frequencies.
    pub rope_spatial_scale: f32,
    pub eps: f32,
}

impl Shape {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// The tokens of `image`: its patches, each side divided by `patch` rounded up.
    pub fn tokens(&self, image: Size) -> usize {
        self.patching(image).tokens()
    }

    /// `image` as a one-frame video cut into patches.
    fn patching(&self, image: Size) -> Patching {
        Patching {
            video: Video {
                frames: 1,
                height: image.height,
                width: image.width,
            },
            patch: self.patch,
        }
    }

    /// Values of a patch: `patch²` pixels of `channels`.
    fn patch_values(&self) -> usize {
        self.patch * self.patch * self.channels
    }

    /// Outputs of a block's fused queries, keys, values and one gate logit per head.
    fn qkv_gate(&self) -> usize {
        3 * self.hidden + self.heads
    }
}

pub struct Dit {
    shape: Shape,
    patch: [Linear; 2],
    time: [Linear; 2],
    /// The timestep's conditioning of the image's blocks, then of the text refiners.
    roots: [Linear; 2],
    text: [Linear; 2],
    image_refiners: Vec<Block>,
    text_refiners: Vec<Block>,
    blocks: Vec<Block>,
    head_norm: LayerNorm,
    head_modulation: Vector,
    head: Linear,
    rotary: Rotary,
}

/// Which root a block's modulations are projected from.
#[derive(Clone, Copy)]
enum Root {
    Image,
    Text,
}

/// What one call shares among its blocks.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    shape: &'call Shape,
    rotary: &'call Rotary,
    patching: Patching,
}

impl Pass<'_> {
    fn tokens(&self) -> usize {
        self.patching.tokens()
    }
}

impl Dit {
    /// What a call on `image` shares among its blocks.
    ///
    /// # Panics
    /// When the buffers hold fewer patches than `image` has.
    fn pass<'call>(
        &'call self,
        ctx: &'call Ctx<'call>,
        buffers: &'call Buffers,
        image: Size,
    ) -> Pass<'call> {
        let pass = Pass {
            ctx,
            buffers,
            shape: &self.shape,
            rotary: &self.rotary,
            patching: self.shape.patching(image),
        };
        assert!(
            pass.tokens() <= buffers.tokens,
            "the buffers hold the image's patches"
        );
        pass
    }

    /// Every block in the order of its modulations, with the root it reads.
    fn every_block(&self) -> impl Iterator<Item = (&Block, Root)> {
        let image_refiners = self.image_refiners.iter().map(|block| (block, Root::Image));
        let text_refiners = self.text_refiners.iter().map(|block| (block, Root::Text));
        let blocks = self.blocks.iter().map(|block| (block, Root::Image));
        image_refiners.chain(text_refiners).chain(blocks)
    }

    /// `text` = the projection of `states` (rows of `shape.text` values, one per token), in a team
    /// run of its own.
    ///
    /// # Panics
    /// When `states` holds more than `text_tokens` rows.
    pub fn condition(&self, team: &Team, buffers: &Buffers, states: &[f32], text: &mut Text) {
        let Shape {
            hidden,
            text: width,
            text_tokens,
            ..
        } = self.shape;
        let rows = states.len() / width;
        assert!(rows <= text_tokens, "the text fits its buffers");
        text.rows = rows;
        let target = &text.states;
        team.run(|ctx| {
            for row in ctx.split(rows, 1) {
                let source = &states[row * width..(row + 1) * width];
                buffers
                    .text_input
                    .part(row * width..(row + 1) * width)
                    .copy_from_slice(source);
            }
            ctx.barrier();
            let input = (&buffers.text_input, rows);
            buffers
                .rows
                .project_synced(ctx, &self.text[0], input, |row, col, value| {
                    buffers.normed.set(row * hidden + col, gelu_tanh(value));
                });
            let activated = (&buffers.normed, rows);
            buffers
                .rows
                .project_synced(ctx, &self.text[1], activated, |row, col, value| {
                    target.set(row * hidden + col, value);
                });
        });
    }

    /// The modulations of every block and of the head at `timestep`, into `buffers`.
    pub fn time(&self, ctx: &Ctx, buffers: &Buffers, timestep: f32) {
        buffers.time.embed(ctx, &buffers.rows, &self.time, timestep);
        let time_silu = (&buffers.time.activated, 1);
        for (root, linear) in self.roots.iter().enumerate() {
            buffers
                .rows
                .project_synced(ctx, linear, time_silu, |_, col, value| {
                    buffers.roots[root].set(col, value);
                });
        }
        self.modulate(ctx, buffers);
        ctx.barrier();
    }

    /// Each block's modulations from its root, and the head's own plus the time embedding.
    fn modulate(&self, ctx: &Ctx, buffers: &Buffers) {
        let hidden = self.shape.hidden;
        for (index, (block, root)) in self.every_block().enumerate() {
            let root = (&buffers.roots[root as usize], 1);
            buffers
                .rows
                .project_synced(ctx, &block.modulation, root, |_, col, value| {
                    let at = index * BLOCK_CHUNKS * hidden + col;
                    buffers.modulations.set(at, modulated(col / hidden, value));
                });
        }
        let time = buffers.time.embedding.read();
        for at in ctx.split(HEAD_CHUNKS * hidden, 1) {
            let value = self.head_modulation.values()[at] + time[at % hidden];
            let chunk = at / hidden;
            buffers
                .head_modulation
                .set(at, scaled(chunk, &HEAD_SCALES, value));
        }
    }

    /// `buffers.refined` = the tokens of `buffers.image` (`image`) through the patch embedding
    /// and the image refiners, at the timestep last given to `time`.
    ///
    /// # Panics
    /// When the buffers hold fewer patches than `image` has.
    pub fn embed_image(&self, ctx: &Ctx, buffers: &Buffers, image: Size) {
        let pass = self.pass(ctx, buffers, image);
        let Shape {
            hidden, bottleneck, ..
        } = self.shape;
        let tokens = pass.tokens();
        pass.patchify();
        let patched = (&buffers.patched, tokens);
        buffers
            .rows
            .project_synced(ctx, &self.patch[0], patched, |token, col, value| {
                buffers.bottleneck.set(token * bottleneck + col, value);
            });
        let narrowed = (&buffers.bottleneck, tokens);
        buffers
            .rows
            .project_synced(ctx, &self.patch[1], narrowed, |token, col, value| {
                buffers.hidden.set(token * hidden + col, value);
            });
        for (index, block) in self.image_refiners.iter().enumerate() {
            block.forward(&pass, 0..tokens, index);
        }
        for token in ctx.split(tokens, 1) {
            let row = token * hidden..(token + 1) * hidden;
            buffers
                .refined
                .part(row.clone())
                .copy_from_slice(&buffers.hidden.read()[row]);
        }
        ctx.barrier();
    }

    /// `buffers.prediction` = the clean image the DiT predicts for the image `embed_image` last
    /// refined, under `text`.
    ///
    /// # Panics
    /// When the buffers hold fewer patches than `image` has.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, text: &Text, image: Size) {
        let pass = self.pass(ctx, buffers, image);
        let (hidden, refiners) = (self.shape.hidden, self.shape.refiners);
        let tokens = pass.tokens();
        let rows = tokens + text.rows;
        for row in ctx.split(rows, 1) {
            let (source, first) = if row < tokens {
                (&buffers.refined, row)
            } else {
                (&text.states, row - tokens)
            };
            buffers
                .hidden
                .part(row * hidden..(row + 1) * hidden)
                .copy_from_slice(&source.read()[first * hidden..(first + 1) * hidden]);
        }
        ctx.barrier();
        for (index, block) in self.text_refiners.iter().enumerate() {
            block.forward(&pass, tokens..rows, refiners + index);
        }
        for (index, block) in self.blocks.iter().enumerate() {
            block.forward(&pass, 0..rows, 2 * refiners + index);
        }
        self.finish(&pass);
    }
}

/// A block's modulation of `chunk`: `1 +` a scale, `tanh` of a gate.
fn modulated(chunk: usize, value: f32) -> f32 {
    if BLOCK_SCALES.contains(&chunk) {
        1.0 + value
    } else {
        value.tanh()
    }
}

impl Shape {
    fn rotary(&self) -> Rotary {
        let rotation = Rotation::Axes(self.rope_sections);
        Rotary::new(rotation, self.rope_base, self.head_dim())
            .spatially_scaled(self.rope_spatial_scale)
    }
}

/// The blocks `prefix.{name}.{i}` for `i < count`, their feed-forwards `feed_forward` wide.
fn load_blocks(
    store: &mut Store,
    prefix: &str,
    shape: Shape,
    (name, count, feed_forward): (&str, usize, usize),
) -> Result<Vec<Block>, bundle::Error> {
    load_each(store, &format!("{prefix}.{name}"), count, |store, path| {
        Block::load(store, path, (shape, feed_forward))
    })
}

impl Load for Dit {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        check_axes(prefix, (shape.hidden, shape.heads), shape.rope_sections)?;
        let Shape {
            hidden,
            rank,
            bottleneck,
            ..
        } = shape;
        let roots_prefix = format!("{prefix}.roots");
        let mut roots = Scope::new(store, &roots_prefix);
        let roots = [
            roots.load("image", [rank, hidden])?,
            roots.load("text", [rank, hidden])?,
        ];
        let head_prefix = format!("{prefix}.head");
        let mut head = Scope::new(store, &head_prefix);
        let head_norm = head.load("norm", hidden)?;
        let head_modulation = head.load("modulation", HEAD_CHUNKS * hidden)?;
        let head_linear = head.load("linear", [shape.patch_values(), hidden])?;
        Ok(Self {
            shape,
            patch: load_pair(
                store,
                &format!("{prefix}.patch"),
                [[bottleneck, shape.patch_values()], [hidden, bottleneck]],
            )?,
            time: load_pair(
                store,
                &format!("{prefix}.time"),
                [[hidden, shape.frequencies], [hidden; 2]],
            )?,
            roots,
            text: load_pair(
                store,
                &format!("{prefix}.text"),
                [[hidden, shape.text], [hidden; 2]],
            )?,
            image_refiners: load_blocks(
                store,
                prefix,
                shape,
                ("image_refiner", shape.refiners, shape.refiner_feed_forward),
            )?,
            text_refiners: load_blocks(
                store,
                prefix,
                shape,
                ("text_refiner", shape.refiners, shape.refiner_feed_forward),
            )?,
            blocks: load_blocks(
                store,
                prefix,
                shape,
                ("blocks", shape.depth, shape.feed_forward),
            )?,
            head_norm,
            head_modulation,
            head: head_linear,
            rotary: shape.rotary(),
        })
    }
}
