//! The token-and-duration transducer, decoded greedily as the reference does: from the first
//! frame, the joint of the frame and the prediction network's output picks a token (or the
//! blank) and a duration. A token runs through the prediction network; the duration, at least
//! one frame after a blank, moves to the next frame. A recording gets at most
//! `MAX_SYMBOLS_PER_FRAME` decisions per frame.
//!
//! `decode` is one team run. Per decision: the joint head by output columns into one of two
//! logit buffers (alternating, so a thread still reading the last decision's never sees them
//! overwritten), a barrier, and every thread takes the same argmaxes. Per token: each LSTM layer
//! by units (every thread first copies the layer's input and previous state, then a barrier) and
//! the projection by output columns.

use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, matmul};
use serde::Serialize;
use team::{Ctx, PerThread, SharedBuf};

use crate::model::{BLANK, DECODER_HIDDEN, DECODER_LAYERS, DURATIONS, Model, TOKENS};

/// The most decisions per encoder frame.
pub const MAX_SYMBOLS_PER_FRAME: usize = 10;
/// LSTM gates per unit: input, forget, cell, output.
const GATES: usize = 4;

/// One decision of the transducer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Step {
    /// A token of the vocabulary, or `BLANK`.
    pub token: u32,
    /// Encoder frames it advances by.
    pub frames: u32,
}

/// The prediction network's state and the decoder's activations.
pub(crate) struct State {
    /// The encoder's frames through the joint's encoder projection, `frames × DECODER_HIDDEN`.
    pub(crate) encoded: SharedBuf,
    /// Per layer, the LSTM's hidden and cell state.
    hidden: SharedBuf,
    cell: SharedBuf,
    /// The prediction network's output.
    output: SharedBuf,
    logits: [SharedBuf; 2],
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    input: Vec<f32>,
    gates: Vec<f32>,
    quant: QuantizedRows,
    steps: Vec<Step>,
}

impl State {
    /// Room for `frames` encoder frames.
    pub(crate) fn new(frames: usize, threads: usize) -> Self {
        let logits = || SharedBuf::new(TOKENS + DURATIONS.len());
        Self {
            encoded: SharedBuf::new(frames * DECODER_HIDDEN),
            hidden: SharedBuf::new(DECODER_LAYERS * DECODER_HIDDEN),
            cell: SharedBuf::new(DECODER_LAYERS * DECODER_HIDDEN),
            output: SharedBuf::new(DECODER_HIDDEN),
            logits: [logits(), logits()],
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    /// Clears the state for a new recording.
    pub(crate) fn reset(&mut self) {
        self.hidden.get_mut().fill(0.0);
        self.cell.get_mut().fill(0.0);
    }

    /// The decisions of the last `decode`.
    pub(crate) fn take_steps(&mut self) -> Vec<Step> {
        std::mem::take(&mut self.scratch.get_mut(0).steps)
    }
}

impl Model {
    /// `state.encoded` = the joint's encoder projection of the first `frames` rows of `hidden`,
    /// this thread's columns, then a barrier.
    pub(crate) fn project_encoded(
        &self,
        ctx: &Ctx,
        hidden: &SharedBuf,
        frames: usize,
        state: &State,
    ) {
        let linear = &self.joint.encoder;
        let matrix = linear.matrix();
        let scratch = state.scratch.get(ctx);
        let input = scratch
            .quant
            .input(&matrix, &hidden.read()[..frames * matrix.cols], frames);
        let bias = linear.bias();
        let cols = ctx.split(matrix.rows, COLUMN_ALIGN);
        matmul(&matrix, &input, frames, cols, |row, col, value| {
            let biased = value + bias.map_or(0.0, |bias| bias[col]);
            state.encoded.set(row * DECODER_HIDDEN + col, biased);
        });
        ctx.barrier();
    }

    /// Decodes the first `frames` rows of `state.encoded` into each thread's `steps`, from a
    /// reset state.
    pub(crate) fn decode(&self, ctx: &Ctx, state: &State, frames: usize) {
        let encoded = state.encoded.read();
        state.scratch.get(ctx).steps.clear();
        self.predict(ctx, state, BLANK);
        let mut frame = 0;
        let mut budget = MAX_SYMBOLS_PER_FRAME * frames;
        let mut decision = 0;
        while frame < frames && budget > 0 {
            let logits = &state.logits[decision % 2];
            let row = &encoded[frame * DECODER_HIDDEN..(frame + 1) * DECODER_HIDDEN];
            self.joint(ctx, state, row, logits);
            ctx.barrier();
            let step = choose(logits.read());
            state.scratch.get(ctx).steps.push(step);
            frame += step.frames as usize;
            if step.token as usize != BLANK {
                self.predict(ctx, state, step.token as usize);
            }
            budget -= 1;
            decision += 1;
        }
    }

    /// The joint's logits for one encoder row and the prediction network's output, this
    /// thread's columns.
    fn joint(&self, ctx: &Ctx, state: &State, encoded: &[f32], logits: &SharedBuf) {
        let scratch = state.scratch.get(ctx);
        let output = state.output.read();
        scratch.input.clear();
        let sums = encoded.iter().zip(output);
        scratch
            .input
            .extend(sums.map(|(frame, token)| (frame + token).max(0.0)));
        let head = &self.joint.head;
        project(
            ctx,
            head,
            &scratch.input,
            &mut scratch.quant,
            |col, value| logits.set(col, value),
        );
    }

    /// Runs `token` through the prediction network: its state and `output`.
    fn predict(&self, ctx: &Ctx, state: &State, token: usize) {
        let decoder = &self.decoder;
        for (layer, weights) in decoder.layers.iter().enumerate() {
            let scratch = state.scratch.get(ctx);
            scratch.input.resize(2 * DECODER_HIDDEN, 0.0);
            let (input, previous) = scratch.input.split_at_mut(DECODER_HIDDEN);
            let hidden = state.hidden.read();
            match layer.checked_sub(1) {
                None => decoder
                    .embedding
                    .matrix()
                    .expect("checked at load")
                    .dequantize_row(token, input),
                Some(below) => {
                    input.copy_from_slice(&hidden[below * DECODER_HIDDEN..][..DECODER_HIDDEN])
                }
            }
            previous.copy_from_slice(&hidden[layer * DECODER_HIDDEN..][..DECODER_HIDDEN]);
            ctx.barrier();
            self.cell(ctx, state, (layer, weights));
            ctx.barrier();
        }
        let scratch = state.scratch.get(ctx);
        let top = (DECODER_LAYERS - 1) * DECODER_HIDDEN;
        scratch.input.clear();
        scratch
            .input
            .extend_from_slice(&state.hidden.read()[top..top + DECODER_HIDDEN]);
        let projector = &decoder.projector;
        project(
            ctx,
            projector,
            &scratch.input,
            &mut scratch.quant,
            |col, value| {
                state.output.set(col, value);
            },
        );
        ctx.barrier();
    }

    /// One LSTM layer for this thread's units, from `[input; previous hidden]` in its scratch.
    fn cell(&self, ctx: &Ctx, state: &State, (layer, weights): (usize, &Linear)) {
        let scratch = state.scratch.get(ctx);
        let units = ctx.split(DECODER_HIDDEN, COLUMN_ALIGN / GATES);
        let matrix = weights.matrix();
        let bias = weights
            .bias()
            .expect("the compiler sums the LSTM biases into one");
        let input = scratch.quant.input(&matrix, &scratch.input, 1);
        let first = GATES * units.start;
        scratch.gates.resize(GATES * units.len(), 0.0);
        let gates = &mut scratch.gates;
        matmul(
            &matrix,
            &input,
            1,
            first..GATES * units.end,
            |_, col, value| {
                gates[col - first] = value + bias[col];
            },
        );
        for (unit, gate) in units.zip(gates.as_chunks::<GATES>().0) {
            let at = layer * DECODER_HIDDEN + unit;
            let [input, forget, cell, output] = *gate;
            let cell = sigmoid(forget) * state.cell.get(at) + sigmoid(input) * cell.tanh();
            state.cell.set(at, cell);
            state.hidden.set(at, sigmoid(output) * cell.tanh());
        }
    }
}

/// `linear(x)` for one row, this thread's output columns (in whole units of `COLUMN_ALIGN`, the
/// last one short: the joint's outputs are not a multiple of it).
fn project(
    ctx: &Ctx,
    linear: &Linear,
    x: &[f32],
    quant: &mut QuantizedRows,
    mut store: impl FnMut(usize, f32),
) {
    let matrix = linear.matrix();
    let bias = linear.bias();
    let input = quant.input(&matrix, x, 1);
    let outputs = matrix.rows;
    let cols = ctx.split(outputs.next_multiple_of(COLUMN_ALIGN), COLUMN_ALIGN);
    let cols = cols.start.min(outputs)..cols.end.min(outputs);
    matmul(&matrix, &input, 1, cols, |_, col, value| {
        store(col, value + bias.map_or(0.0, |bias| bias[col]));
    });
}

/// The decision of a row of joint logits: the most likely token and the most likely duration
/// (the first on ties), a blank advancing at least one frame.
fn choose(logits: &[f32]) -> Step {
    let (tokens, durations) = logits.split_at(TOKENS);
    let token = argmax(tokens);
    let frames = match (token, DURATIONS[argmax(durations)]) {
        (BLANK, 0) => 1,
        (_, frames) => frames,
    };
    Step {
        token: token as u32,
        frames: frames as u32,
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
