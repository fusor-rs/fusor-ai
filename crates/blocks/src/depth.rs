//! A depth decoder: the codes of one audio frame, one per codebook, scored from a language
//! model's hidden state by unmasking (LFM3-Audio's). The hidden state is projected to one slot
//! per codebook; each slot adds its codebook's embedding of the code where the code is known,
//! or a learned mask embedding where it is not; an `interleaved` stack whose slots all see each
//! other and its final RMSNorm follow; a codebook's logits are its slot RMS-normed by the
//! codebook's own norm, times the codebook's embeddings (tied).
//!
//! Tensors under a prefix `p`: `p.slots` (a linear with a bias, `codebooks · hidden` outputs,
//! slot-major), `p.mask` (`hidden`), the stack's (`p.layers.{i}…`), `p.norm`, and
//! `p.codebooks.{k}.embedding` (`[vocabulary, hidden]`) and `p.codebooks.{k}.norm`.
//!
//! `Depth::slots` once per frame, then `Depth::score` once per pass, run inside team runs.

use bundle::{Load, Scope, Store, load_each};
use kernels::ops::rms_norm;
use kernels::{Linear, QuantizedRows, SharedRows, Vector, project};
use team::{Ctx, PerThread, SharedBuf};

use crate::hybrid::Rows;
use crate::interleaved::{self, Interleaved};
use crate::layer;
use crate::lm::Prediction;
use crate::rotary::Position;
use crate::transformer::Mask;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub codebooks: usize,
    /// Codes of a codebook.
    pub vocabulary: usize,
    /// Width of the language model's hidden state.
    pub input: usize,
    pub stack: interleaved::Shape,
}

struct Codebook {
    embedding: Linear,
    norm: Vector,
}

pub struct Depth {
    shape: Shape,
    slots: Linear,
    mask: Vector,
    stack: Interleaved,
    norm: Vector,
    codebooks: Vec<Codebook>,
}

/// The activations of one frame.
pub struct Buffers {
    /// The projected hidden state, `codebooks × hidden`.
    slots: SharedBuf,
    stack: layer::Buffers,
    state: interleaved::State,
    positions: Vec<Position>,
    normed: PerThread<Vec<f32>>,
    quant: PerThread<QuantizedRows>,
    /// One per codebook: the logits of the codebooks the last pass scored.
    pub predictions: Vec<Prediction>,
}

impl Buffers {
    pub fn new(shape: &Shape, threads: usize) -> Self {
        let Shape {
            codebooks,
            vocabulary,
            ..
        } = *shape;
        let hidden = shape.stack.hidden;
        Self {
            slots: SharedBuf::new(codebooks * hidden),
            stack: shape.stack.buffers(codebooks, threads),
            state: interleaved::State::new(&shape.stack, codebooks, codebooks),
            positions: (0..codebooks).map(Position::uniform).collect(),
            normed: PerThread::new(threads, || vec![0.0; hidden]),
            quant: PerThread::new(threads, QuantizedRows::default),
            predictions: (0..codebooks)
                .map(|_| Prediction::new(hidden, vocabulary, threads))
                .collect(),
        }
    }
}

impl Depth {
    /// The slots of a frame from the language model's final-normed `hidden` state (one row).
    pub fn slots(&self, ctx: &Ctx, buffers: &Buffers, hidden: &SharedBuf) {
        let rows = SharedRows {
            values: hidden,
            rows: 1,
        };
        let quant = buffers.quant.get(ctx);
        project(ctx, &self.slots, rows, quant, |at, value| {
            buffers.slots.set(at, value);
        });
        ctx.barrier();
    }

    /// One pass over the frame's slots: `codes[k]` is codebook `k`'s code, `None` where it is
    /// masked; then the logits of each codebook in `scored` into its `buffers.predictions`,
    /// read after the run.
    pub fn score(&self, ctx: &Ctx, buffers: &Buffers, codes: &[Option<u32>], scored: &[usize]) {
        let hidden = self.shape.stack.hidden;
        let slots = buffers.slots.read();
        for codebook in ctx.split(self.shape.codebooks, 1) {
            let at = codebook * hidden;
            let row = buffers.stack.hidden.part(at..at + hidden);
            match codes[codebook] {
                Some(code) => self.codebooks[codebook]
                    .embedding
                    .matrix()
                    .dequantize_row(code as usize, row),
                None => row.copy_from_slice(self.mask.values()),
            }
            for (value, slot) in row.iter_mut().zip(&slots[at..at + hidden]) {
                *value += slot;
            }
        }
        ctx.barrier();
        let rows = Rows {
            first: 0,
            positions: &buffers.positions,
            mask: Mask::Full,
        };
        self.stack
            .forward(ctx, &buffers.state, &buffers.stack, &rows);
        let eps = self.shape.stack.norm_eps;
        let out = buffers.stack.hidden.read();
        let normed = buffers.normed.get(ctx);
        for &index in scored {
            let codebook = &self.codebooks[index];
            let row = &out[index * hidden..(index + 1) * hidden];
            rms_norm(row, self.norm.values(), eps, normed);
            let norm = (&codebook.norm, eps);
            buffers.predictions[index].predict(ctx, normed, norm, &codebook.embedding);
        }
    }
}

impl Load for Depth {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let hidden = shape.stack.hidden;
        let stack = Interleaved::load(store, prefix, shape.stack)?;
        let codebooks = load_each(
            store,
            &format!("{prefix}.codebooks"),
            shape.codebooks,
            |store, name| {
                let mut scope = Scope::new(store, name);
                Ok(Codebook {
                    embedding: scope.load("embedding", [shape.vocabulary, hidden])?,
                    norm: scope.load("norm", hidden)?,
                })
            },
        )?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            slots: scope.load("slots", [shape.codebooks * hidden, shape.input])?,
            mask: scope.load("mask", hidden)?,
            stack,
            norm: scope.load("norm", hidden)?,
            codebooks,
        })
    }
}
