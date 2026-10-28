//! An audio encoder into a language model's tokens (LFM2-Audio's, LFM3-Omni's): a clip's
//! log-mel features through `conformer`'s subsampler, an `interleaved` stack and its final RMS
//! norm (`Encoder::encode`), then a projector (`Encoder::project`): each frame LayerNorm-ed,
//! through `down(gelu(up(·)))` (exact GELU) into one token of the language model's width.
//!
//! Tensors under a prefix `p`: the subsampler's under `p.subsampling`, the stack's
//! (`p.layers.{i}…`), `p.norm`, and the projector's under `p.projector`: `norm` (`weight` and
//! `bias`, a frame wide), `up` and `down` (linears with biases).
//!
//! `Encoder::forward`, `encode` and `project` run inside a team run; a clip is read from its
//! first frame, after `Buffers::clear`.

use bundle::{Load, Scope, Store};
use kernels::ops::{gelu_erf, rms_norm};
use kernels::{Linear, SharedRows, Vector, project};
use team::{Ctx, SharedBuf};

use crate::conformer::subsampling::{self, Frames, Subsampling};
use crate::hybrid::Rows;
use crate::interleaved::{self, Interleaved, State};
use crate::layer;
use crate::norm::LayerNorm;
use crate::rotary::Position;
use crate::transformer::Mask;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub subsampling: subsampling::Shape,
    pub stack: interleaved::Shape,
    /// Width of a token: the language model's.
    pub hidden: usize,
    pub projector_eps: f32,
}

impl Shape {
    /// The tokens of a clip of `mel_frames` mel frames: its subsampled frames.
    pub fn tokens(&self, mel_frames: usize) -> usize {
        Frames::of(&self.subsampling, mel_frames).valid
    }

    /// Activations of clips of up to `mel_frames` mel frames, each thread splitting the
    /// subsampler's frames in blocks of `block_frames`.
    pub fn buffers(&self, mel_frames: usize, block_frames: usize, threads: usize) -> Buffers {
        let frames = Frames::of(&self.subsampling, mel_frames).all;
        Buffers {
            subsampling: subsampling::Buffers::new(
                self.subsampling,
                mel_frames,
                block_frames,
                threads,
            ),
            stack: self.stack.buffers(frames, threads),
            state: State::new(&self.stack, frames, frames),
            positions: (0..frames).map(Position::uniform).collect(),
            encoded: SharedBuf::new(frames * self.stack.hidden),
            normed: SharedBuf::new(frames * self.stack.hidden),
            inner: SharedBuf::new(frames * self.hidden),
            tokens: SharedBuf::new(frames * self.hidden),
        }
    }
}

/// Activations of one clip.
pub struct Buffers {
    subsampling: subsampling::Buffers,
    stack: layer::Buffers,
    state: State,
    positions: Vec<Position>,
    /// The stack's output frames, final-normed, `frames × stack hidden`: `encode`'s output and
    /// `project`'s input.
    pub encoded: SharedBuf,
    /// `encoded` LayerNorm-ed.
    normed: SharedBuf,
    /// The projector's inner activations, `frames × hidden`.
    inner: SharedBuf,
    /// The clip's tokens, `frames × hidden`.
    pub tokens: SharedBuf,
}

impl Buffers {
    /// Forgets the clip read before, to read the next from its first frame.
    pub fn clear(&mut self) {
        self.state.clear();
    }
}

pub struct Encoder {
    shape: Shape,
    subsampling: Subsampling,
    stack: Interleaved,
    norm: Vector,
    projector_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

impl Encoder {
    /// The tokens of the clip whose features are `features` (`mel_frames × mels` row-major)
    /// into `buffers.tokens`, `Shape::tokens` of them: `encode`, then `project`.
    ///
    /// # Panics
    /// When there are no features or more than the buffers hold.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, features: &[f32]) {
        let frames = self.encode(ctx, buffers, features);
        self.project(ctx, buffers, frames);
    }

    /// The clip's frames through the subsampler, the stack and its final norm into
    /// `buffers.encoded`: returns how many.
    ///
    /// # Panics
    /// As `forward`.
    pub fn encode(&self, ctx: &Ctx, buffers: &Buffers, features: &[f32]) -> usize {
        let frames = self
            .subsampling
            .forward(ctx, &buffers.subsampling, features)
            .valid;
        let (hidden, eps) = (self.shape.stack.hidden, self.shape.stack.norm_eps);
        let subsampled = buffers.subsampling.hidden.read();
        for frame in ctx.split(frames, 1) {
            let at = frame * hidden..(frame + 1) * hidden;
            buffers
                .stack
                .hidden
                .part(at.clone())
                .copy_from_slice(&subsampled[at]);
        }
        ctx.barrier();
        let rows = Rows {
            first: 0,
            positions: &buffers.positions[..frames],
            mask: Mask::Causal,
        };
        self.stack
            .forward(ctx, &buffers.state, &buffers.stack, &rows);
        let output = buffers.stack.hidden.read();
        for frame in ctx.split(frames, 1) {
            let at = frame * hidden..(frame + 1) * hidden;
            let out = buffers.encoded.part(at.clone());
            rms_norm(&output[at], self.norm.values(), eps, out);
        }
        ctx.barrier();
        frames
    }

    /// `buffers.encoded`'s first `frames` frames through the projector into `buffers.tokens`.
    pub fn project(&self, ctx: &Ctx, buffers: &Buffers, frames: usize) {
        let hidden = self.shape.stack.hidden;
        let encoded = buffers.encoded.read();
        for frame in ctx.split(frames, 1) {
            let at = frame * hidden..(frame + 1) * hidden;
            let out = buffers.normed.part(at.clone());
            self.projector_norm
                .apply(&encoded[at], self.shape.projector_eps, out);
        }
        ctx.barrier();
        let quant = &mut buffers.stack.scratch.get(ctx).quant;
        let rows = |values| SharedRows {
            values,
            rows: frames,
        };
        project(ctx, &self.up, rows(&buffers.normed), quant, |at, value| {
            buffers.inner.set(at, gelu_erf(value));
        });
        ctx.barrier();
        project(ctx, &self.down, rows(&buffers.inner), quant, |at, value| {
            buffers.tokens.set(at, value);
        });
        ctx.barrier();
    }
}

impl Load for Encoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let width = shape.stack.hidden;
        let subsampling =
            Subsampling::load(store, &format!("{prefix}.subsampling"), shape.subsampling)?;
        let stack = Interleaved::load(store, prefix, shape.stack)?;
        let norm = Scope::new(store, prefix).load("norm", width)?;
        let projector_prefix = format!("{prefix}.projector");
        let mut projector = Scope::new(store, &projector_prefix);
        Ok(Self {
            shape,
            subsampling,
            stack,
            norm,
            projector_norm: projector.load("norm", width)?,
            up: projector.load("up", [shape.hidden, width])?,
            down: projector.load("down", [shape.hidden, shape.hidden])?,
        })
    }
}
