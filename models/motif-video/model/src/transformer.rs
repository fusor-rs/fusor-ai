//! Motif-Video's transformer around its `mmdit` blocks: the latents (with the first frame's
//! condition and a mask) in patches of 2 × 2 latent pixels, one token each; a context of the
//! first image's SigLIP states (projected) and the prompt's T5Gemma2 states (through an MLP);
//! the timestep's sinusoids through an MLP; 12 dual-stream blocks and the first 16 single-stream
//! blocks over the video's tokens and the context's rows; then the 8 single-stream blocks of the
//! DDT decoder over the patch embeddings and the encoder's video tokens; a modulated head and
//! the patches back into velocities.
//!
//! Tensors under `DIT`: the linears `patch`, `context.{0,1}`, `image.{0,1}`, `time.{0,1}`,
//! `head.modulation`, `head.linear`, the LayerNorms `image.norm_in` and `image.norm_out`, and the
//! blocks `dual.{i}` and `single.{j}` (`mmdit`'s).
//!
//! Every method runs inside a team run. A request's text and image condition a `Contexts` once
//! (`embed_text`, `embed_image`); each step computes the timestep's modulations once (`time`),
//! then `forward` for each branch of guidance.

mod buffers;
mod embedding;

use blocks::mmdit::{Cross, Dual, Rows, Single};
use blocks::norm::LayerNorm;
use blocks::rotary::Position;
use blocks::timestep_sinusoid;
use blocks::video::Patching;
use bundle::{Load, Scope, Store, load_each, load_pair};
use kernels::Linear;
use kernels::ops::{layer_norm, silu};
use team::Ctx;

use crate::Error;
use crate::model::{
    DIT, DIT_SHAPE, DUAL_BLOCKS, ENCODER_SINGLE_BLOCKS, IMAGE_HIDDEN, INPUT_CHANNELS,
    LATENT_CHANNELS, NORM_EPS, SINGLE_BLOCKS, TEXT_HIDDEN, TIME_FREQUENCIES,
};
use crate::session::Branch;
pub(crate) use buffers::{BRANCHES, Buffers, Contexts};
use buffers::{PATCHED_INPUT, PATCHED_OUTPUT};

const HIDDEN: usize = DIT_SHAPE.hidden;

pub(crate) struct Transformer {
    patch: Linear,
    context: [Linear; 2],
    image: ImageProjection,
    time: [Linear; 2],
    dual: Vec<Dual>,
    single: Vec<Single>,
    head_modulation: Linear,
    head: Linear,
}

/// SigLIP's states into the context: LayerNorm, MLP, LayerNorm.
struct ImageProjection {
    norm_in: LayerNorm,
    linear: [Linear; 2],
    norm_out: LayerNorm,
}

/// Single-stream block `index`'s cross-attention: the encoder's carry one to the text.
fn cross(index: usize) -> Cross {
    if index < ENCODER_SINGLE_BLOCKS {
        Cross::Text
    } else {
        Cross::None
    }
}

/// One forward's inputs beside the buffers.
#[derive(Clone, Copy)]
pub(crate) struct Call<'call> {
    pub(crate) contexts: &'call Contexts,
    pub(crate) branch: Branch,
    pub(crate) patching: Patching,
    /// The video's tokens' positions, one each.
    pub(crate) positions: &'call [Position],
}

impl Transformer {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let dual = load_each(store, &format!("{DIT}.dual"), DUAL_BLOCKS, |store, name| {
            Dual::load(store, name, DIT_SHAPE)
        })?;
        let single = (0..SINGLE_BLOCKS)
            .map(|index| {
                let name = format!("{DIT}.single.{index}");
                Single::load(store, &name, (DIT_SHAPE, cross(index)))
            })
            .collect::<Result<_, _>>()?;
        let context = load_pair(
            store,
            &format!("{DIT}.context"),
            [[HIDDEN, TEXT_HIDDEN], [HIDDEN; 2]],
        )?;
        let time = load_pair(
            store,
            &format!("{DIT}.time"),
            [[HIDDEN, TIME_FREQUENCIES], [HIDDEN; 2]],
        )?;
        let image = ImageProjection::load(store, &format!("{DIT}.image"))?;
        let mut scope = Scope::new(store, DIT);
        Ok(Self {
            patch: scope.load("patch", [HIDDEN, PATCHED_INPUT])?,
            context,
            image,
            time,
            dual,
            single,
            head_modulation: scope.load("head.modulation", [2 * HIDDEN, HIDDEN])?,
            head: scope.load("head.linear", [PATCHED_OUTPUT, HIDDEN])?,
        })
    }

    /// Every block's and the head's modulations at `timestep`.
    pub(crate) fn time(&self, ctx: &Ctx, buffers: &Buffers, timestep: f32) {
        if ctx.is_first() {
            let frequencies = buffers.sinusoid.part(0..TIME_FREQUENCIES);
            timestep_sinusoid(timestep, blocks::SINUSOID_BASE, frequencies);
        }
        ctx.barrier();
        let sinusoid = buffers.sinusoid.read();
        buffers.project(ctx, &self.time[0], (sinusoid, 1), |_, col, value| {
            buffers.time_hidden.set(col, silu(value));
        });
        let time_hidden = buffers.time_hidden.read();
        buffers.project(ctx, &self.time[1], (time_hidden, 1), |_, col, value| {
            buffers.time.set(col, silu(value));
        });
        let time = buffers.time.read();
        let (dual, single) = buffers.modulations.split_at(DUAL_BLOCKS);
        for (block, out) in self.dual.iter().zip(dual) {
            block.modulate(ctx, &buffers.blocks, (time, out));
        }
        for (block, out) in self.single.iter().zip(single) {
            block.modulate(ctx, &buffers.blocks, (time, out));
        }
        buffers.project(ctx, &self.head_modulation, (time, 1), |_, col, value| {
            let modulation = if col < HIDDEN { 1.0 + value } else { value };
            buffers.head_modulation.set(col, modulation);
        });
    }

    /// `buffers.velocity` for `buffers.input` (`call.patching`'s video) under `call.branch`'s
    /// context at the timestep last given to `time`.
    ///
    /// # Panics
    /// When the buffers hold fewer tokens than the video has.
    pub(crate) fn forward(&self, ctx: &Ctx, buffers: &Buffers, call: Call) {
        let tokens = call.patching.tokens();
        assert!(
            tokens <= buffers.tokens,
            "the buffers hold the video's tokens"
        );
        self.embed_patches(ctx, buffers, &call.patching);
        embedding::copy_context(ctx, buffers, &call);
        let text = call.contexts.text_rows(call.branch);
        let encoder_rows = Rows {
            positions: call.positions,
            context: call.contexts.image_rows() + text,
            text,
        };
        let (dual_modulations, single_modulations) = buffers.modulations.split_at(DUAL_BLOCKS);
        for (block, modulation) in self.dual.iter().zip(dual_modulations) {
            block.forward(ctx, &buffers.blocks, modulation.read(), encoder_rows);
        }
        let (encoder, decoder) = self.single.split_at(ENCODER_SINGLE_BLOCKS);
        let (encoder_modulations, decoder_modulations) =
            single_modulations.split_at(ENCODER_SINGLE_BLOCKS);
        for (block, modulation) in encoder.iter().zip(encoder_modulations) {
            block.forward(ctx, &buffers.blocks, modulation.read(), encoder_rows);
        }
        hand_over(ctx, buffers, tokens);
        let decoder_rows = Rows {
            positions: call.positions,
            context: tokens,
            text: 0,
        };
        for (block, modulation) in decoder.iter().zip(decoder_modulations) {
            block.forward(ctx, &buffers.blocks, modulation.read(), decoder_rows);
        }
        self.finish(ctx, buffers, &call.patching);
    }

    /// `hidden` and `stream_input` = every token's patch embedding.
    fn embed_patches(&self, ctx: &Ctx, buffers: &Buffers, patching: &Patching) {
        let input = (buffers.input.read(), INPUT_CHANNELS);
        for token in ctx.split(patching.tokens(), 1) {
            let patch = buffers
                .patched
                .part(token * PATCHED_INPUT..(token + 1) * PATCHED_INPUT);
            patching.gather(input, token, patch);
        }
        ctx.barrier();
        let patched = (buffers.patched.read(), patching.tokens());
        buffers.project(ctx, &self.patch, patched, |token, col, value| {
            buffers.blocks.hidden.set(token * HIDDEN + col, value);
            buffers.stream_input.set(token * HIDDEN + col, value);
        });
    }

    /// The head over the decoder's video tokens, and the velocity patches put back where they
    /// came from.
    fn finish(&self, ctx: &Ctx, buffers: &Buffers, patching: &Patching) {
        let tokens = patching.tokens();
        let modulation = buffers.head_modulation.read();
        let (scale, shift) = modulation.split_at(HIDDEN);
        let hidden = buffers.blocks.hidden.read();
        for token in ctx.split(tokens, 1) {
            let row = token * HIDDEN..(token + 1) * HIDDEN;
            let normed = buffers.stream_input.part(row.clone());
            layer_norm(&hidden[row], scale, shift, NORM_EPS, normed);
        }
        ctx.barrier();
        let normed = (buffers.stream_input.read(), tokens);
        buffers.project(ctx, &self.head, normed, |token, col, value| {
            buffers.head_output.set(token * PATCHED_OUTPUT + col, value);
        });
        let head_output = buffers.head_output.read();
        for token in ctx.split(tokens, 1) {
            let patch = &head_output[token * PATCHED_OUTPUT..(token + 1) * PATCHED_OUTPUT];
            patching.scatter((patch, LATENT_CHANNELS), token, &buffers.velocity);
        }
        ctx.barrier();
    }
}

/// The decoder's rows: the patch embeddings, then the encoder's video tokens.
fn hand_over(ctx: &Ctx, buffers: &Buffers, tokens: usize) {
    let hidden = &buffers.blocks.hidden;
    for row in ctx.split(tokens, 1) {
        let encoded = &hidden.read()[row * HIDDEN..(row + 1) * HIDDEN];
        hidden
            .part((tokens + row) * HIDDEN..(tokens + row + 1) * HIDDEN)
            .copy_from_slice(encoded);
    }
    ctx.barrier();
    for row in ctx.split(tokens, 1) {
        let embedded = &buffers.stream_input.read()[row * HIDDEN..(row + 1) * HIDDEN];
        hidden
            .part(row * HIDDEN..(row + 1) * HIDDEN)
            .copy_from_slice(embedded);
    }
    ctx.barrier();
}

impl ImageProjection {
    fn load(store: &mut Store, prefix: &str) -> Result<Self, Error> {
        let linear = load_pair(store, prefix, [[IMAGE_HIDDEN; 2], [HIDDEN, IMAGE_HIDDEN]])?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            norm_in: scope.load("norm_in", IMAGE_HIDDEN)?,
            linear,
            norm_out: scope.load("norm_out", HIDDEN)?,
        })
    }
}
