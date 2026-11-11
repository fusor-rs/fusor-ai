//! Wan2.1's diffusion transformer (its 1.3B and 14B text-to-video models): a video's latents in
//! patches of one frame by `patch × patch` pixels, one token each; the timestep's sinusoids through
//! an MLP into six modulations per block (shifts, scales and gates around a parameter-free
//! LayerNorm); per block a self-attention with the queries and keys RMS-normalized over the whole
//! width and turned by 3-D rotary embeddings (`Rotation::Axes`, interleaved pairs), a
//! cross-attention to the text's `text_tokens` context rows (the prompt's states, then zero rows,
//! through the text MLP; every row attended), and a GELU-tanh MLP; a modulated head and the
//! patches back into latents. Attention runs on the int8 kernels (`attention`).
//!
//! Tensors under a prefix `p`: the linears `p.patch`, `p.text.{0,1}`, `p.time.{0,1}`,
//! `p.modulation`, `p.head.linear`, the vector `p.head.modulation`, and per block
//! `p.blocks.{i}.modulation` (a vector) and the linears `qkv`, `out`, `cross_query`,
//! `cross_context` (keys, then values), `cross_out`, `ff1`, `ff2`, the vectors `q_norm`, `k_norm`,
//! `cross_q_norm`, `cross_k_norm` and the LayerNorm `norm3`.
//!
//! Every method runs inside a team run. A request's text conditions a `Context` once
//! (`condition`); each step computes the timestep's modulations once (`time`), then `forward` for
//! each branch of classifier-free guidance.

mod block;
mod buffers;
mod embedding;

use bundle::{Load, Scope, Store, load_each, load_pair};
use kernels::ops::{gelu_tanh, layer_norm};
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use crate::rotary::{Rotary, Rotation, check_axes};
use crate::scaled;
use crate::video::{Patching, Video};
use block::Block;
pub use buffers::{Buffers, Context};

/// The modulations of a block, `hidden` values each: a shift, a scale and a gate for the
/// attention, then for the MLP.
#[derive(Clone, Copy)]
enum Chunk {
    AttentionShift,
    AttentionScale,
    AttentionGate,
    MlpShift,
    MlpScale,
    MlpGate,
}

const BLOCK_CHUNKS: usize = Chunk::MlpGate as usize + 1;
/// The chunks of a block's modulations that scale a norm: `1 +` them is its weight.
const BLOCK_SCALES: [usize; 2] = [Chunk::AttentionScale as usize, Chunk::MlpScale as usize];

/// The modulations of the head, `hidden` values each.
#[derive(Clone, Copy)]
enum HeadChunk {
    Shift,
    Scale,
}

const HEAD_CHUNKS: usize = HeadChunk::Scale as usize + 1;
const HEAD_SCALES: [usize; 1] = [HeadChunk::Scale as usize];

/// The dimensions of one transformer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub feed_forward: usize,
    pub depth: usize,
    /// Channels of a latent pixel, in and out.
    pub latent: usize,
    /// Latent pixels per patch along each side; a patch is one frame deep.
    pub patch: usize,
    /// Width of a text state.
    pub text: usize,
    /// Rows of the context: the prompt's states, then zero rows.
    pub text_tokens: usize,
    /// Width of the timestep's sinusoids.
    pub frequencies: usize,
    /// Rotary frequencies turned by a patch's frame, row and column.
    pub rope_sections: [usize; 3],
    pub rope_base: f32,
    pub eps: f32,
}

impl Shape {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Values of a patch: `patch²` pixels of `latent` channels.
    fn patch_values(&self) -> usize {
        self.patch * self.patch * self.latent
    }
}

pub struct Dit {
    shape: Shape,
    patch: Linear,
    text: [Linear; 2],
    time: [Linear; 2],
    modulation: Linear,
    head_modulation: Vector,
    head: Linear,
    blocks: Vec<Block>,
    rotary: Rotary,
}

/// What one forward call shares among its blocks.
struct Pass<'call> {
    ctx: &'call Ctx<'call>,
    buffers: &'call Buffers,
    shape: &'call Shape,
    patching: Patching,
    rotary: &'call Rotary,
}

impl Pass<'_> {
    fn tokens(&self) -> usize {
        self.patching.tokens()
    }

    /// `normed` = `hidden` normalized, then scaled by `scales` and shifted by `shifts`; every
    /// thread's tokens, then a barrier.
    fn modulated_norm(&self, shifts: &[f32], scales: &[f32]) {
        let hidden = self.shape.hidden;
        for token in self.ctx.split(self.tokens(), 1) {
            let row = token * hidden..(token + 1) * hidden;
            let input = &self.buffers.hidden.read()[row.clone()];
            let output = self.buffers.normed.part(row);
            layer_norm(input, scales, shifts, self.shape.eps, output);
        }
        self.ctx.barrier();
    }

    /// `hidden += gate ⊙ linear(input)` (`gate` of 1 without one), `input` the call's tokens.
    fn residual(&self, linear: &Linear, input: &SharedBuf, gate: Option<&[f32]>) {
        let width = self.shape.hidden;
        let hidden = &self.buffers.hidden;
        let tokens = (input, self.tokens());
        self.buffers
            .rows
            .project_synced(self.ctx, linear, tokens, |token, col, value| {
                let at = token * width + col;
                let gated = gate.map_or(value, |gate| gate[col] * value);
                hidden.set(at, hidden.get(at) + gated);
            });
    }
}

impl Dit {
    /// Every block's cross-attention keys and values of `text` (rows of `text` values, at most
    /// `text_tokens`; the rows after them zeros) into `context`.
    pub fn condition(&self, ctx: &Ctx, buffers: &Buffers, text: &[f32], context: &Context) {
        let Shape {
            hidden,
            text: width,
            text_tokens,
            ..
        } = self.shape;
        let rows = text.len() / width;
        for row in ctx.split(text_tokens, 1) {
            let target = buffers.text_input.part(row * width..(row + 1) * width);
            if row < rows {
                target.copy_from_slice(&text[row * width..(row + 1) * width]);
            } else {
                target.fill(0.0);
            }
        }
        ctx.barrier();
        let input = (&buffers.text_input, text_tokens);
        buffers
            .rows
            .project_synced(ctx, &self.text[0], input, |row, col, value| {
                buffers
                    .text_hidden
                    .set(row * hidden + col, gelu_tanh(value));
            });
        let hidden_states = (&buffers.text_hidden, text_tokens);
        buffers
            .rows
            .project_synced(ctx, &self.text[1], hidden_states, |row, col, value| {
                buffers.text_embedded.set(row * hidden + col, value);
            });
        for (block, keys) in self.blocks.iter().zip(&context.layers) {
            block.condition(ctx, buffers, &self.shape, keys);
        }
    }

    /// The modulations of every block and of the head at `timestep`, into `buffers`.
    pub fn time(&self, ctx: &Ctx, buffers: &Buffers, timestep: f32) {
        buffers.time.embed(ctx, &buffers.rows, &self.time, timestep);
        let time_silu = (&buffers.time.activated, 1);
        buffers
            .rows
            .project_synced(ctx, &self.modulation, time_silu, |_, col, value| {
                buffers.projected.set(col, value);
            });
        self.modulate(ctx, buffers);
        ctx.barrier();
    }

    /// Each block's own modulation parameters plus the projected time, and the head's plus the
    /// time embedding; `1 +` the scales.
    fn modulate(&self, ctx: &Ctx, buffers: &Buffers) {
        let hidden = self.shape.hidden;
        let per_block = BLOCK_CHUNKS * hidden;
        let blocks = self.blocks.len() * per_block;
        let (projected, time) = (buffers.projected.read(), buffers.time.embedding.read());
        for at in ctx.split(blocks + HEAD_CHUNKS * hidden, 1) {
            if at < blocks {
                let within = at % per_block;
                let parameter = self.blocks[at / per_block].modulation.values()[within];
                let value = parameter + projected[within];
                let chunk = within / hidden;
                buffers
                    .modulations
                    .set(at, scaled(chunk, &BLOCK_SCALES, value));
            } else {
                let within = at - blocks;
                let value = self.head_modulation.values()[within] + time[within % hidden];
                let chunk = within / hidden;
                buffers
                    .head_modulation
                    .set(within, scaled(chunk, &HEAD_SCALES, value));
            }
        }
    }

    /// `buffers.velocity` = the velocity predicted for `buffers.latents` (`video`) under `context`
    /// at the timestep last given to `time`.
    ///
    /// # Panics
    /// When the buffers hold fewer tokens than `video` has.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, context: &Context, video: Video) {
        let pass = Pass {
            ctx,
            buffers,
            shape: &self.shape,
            patching: Patching {
                video,
                patch: self.shape.patch,
            },
            rotary: &self.rotary,
        };
        assert!(
            pass.tokens() <= buffers.tokens,
            "the buffers hold the video's tokens"
        );
        let hidden = self.shape.hidden;
        self.patchify(&pass);
        let patched = (&buffers.patched, pass.tokens());
        buffers
            .rows
            .project_synced(ctx, &self.patch, patched, |token, col, value| {
                buffers.hidden.set(token * hidden + col, value);
            });
        for (index, (block, keys)) in self.blocks.iter().zip(&context.layers).enumerate() {
            block.forward(&pass, index, keys);
        }
        self.finish(&pass);
    }
}

impl Load for Dit {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            heads,
            text,
            frequencies,
            ..
        } = shape;
        check_axes(prefix, (hidden, heads), shape.rope_sections)?;
        let blocks = load_each(
            store,
            &format!("{prefix}.blocks"),
            shape.depth,
            |store, name| Block::load(store, name, shape),
        )?;
        let text = load_pair(
            store,
            &format!("{prefix}.text"),
            [[hidden, text], [hidden; 2]],
        )?;
        let time = load_pair(
            store,
            &format!("{prefix}.time"),
            [[hidden, frequencies], [hidden; 2]],
        )?;
        let head_prefix = format!("{prefix}.head");
        let mut head = Scope::new(store, &head_prefix);
        let head_modulation = head.load("modulation", HEAD_CHUNKS * hidden)?;
        let head_linear = head.load("linear", [shape.patch_values(), hidden])?;
        let mut scope = Scope::new(store, prefix);
        let rotation = Rotation::Axes(shape.rope_sections);
        Ok(Self {
            shape,
            patch: scope.load("patch", [hidden, shape.patch_values()])?,
            text,
            time,
            modulation: scope.load("modulation", [BLOCK_CHUNKS * hidden, hidden])?,
            head_modulation,
            head: head_linear,
            blocks,
            rotary: Rotary::new(rotation, shape.rope_base, shape.head_dim()),
        })
    }
}
