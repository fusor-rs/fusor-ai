//! The gated delta rule (Gated DeltaNet): queries, keys and values through a causal depthwise
//! convolution and SiLU, the queries and keys L2-normalized; per value head a memory
//! (`value_dim × key_dim`) decayed by `exp(−exp(a_log) · softplus(a + dt_bias))`, corrected
//! toward each value by `sigmoid(b)` of the error of its key's recall, and read by the query; the
//! reading RMS-normalized and gated by `silu(z)`.
//!
//! Tensors under a layer's prefix `p`: `p.input` (rows: the query heads, the key heads, the value
//! heads, `z` (one value head's width per value head), `b`, `a` (one per value head)), `p.conv`
//! (`channels × kernel`: each query, key and value channel's taps, its last on the current row),
//! `p.a_log` and `p.dt_bias` (one per value head), `p.norm` (`value_dim`), `p.out`.
//!
//! `mix` is three phases: the projection by columns; the convolution by channels; the rule by
//! value heads, each running through the rows in order.

use bundle::{Scope, Store};
use kernels::ops::{rms_norm, sigmoid, silu};
use kernels::{
    COLUMN_ALIGN, DOTS, Input, Linear, QUAD, Table, Vector, decay_rows, dot_f32, dot_f32_many,
};
use team::{PerThread, SharedBuf};

use crate::layer::Pass;

/// The L2 normalization's epsilon, under the root of the sum of squares (FLA's `l2norm`).
const L2_EPS: f32 = 1e-6;
/// Above this, `softplus(x)` is `x` (PyTorch's threshold).
const SOFTPLUS_LINEAR: f32 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// Taps of the convolution.
    pub kernel: usize,
}

impl Shape {
    fn keys(&self) -> usize {
        self.key_heads * self.key_dim
    }

    pub(super) fn values(&self) -> usize {
        self.value_heads * self.value_dim
    }

    /// The convolution's channels: the queries, the keys and the values.
    pub(super) fn channels(&self) -> usize {
        2 * self.keys() + self.values()
    }

    /// Columns of the input projection.
    pub(super) fn input(&self) -> usize {
        self.channels() + self.values() + 2 * self.value_heads
    }

    /// Rows of the convolution's input it keeps for the next call.
    fn history(&self) -> usize {
        self.kernel - 1
    }

    fn memory(&self) -> usize {
        self.value_dim * self.key_dim
    }
}

pub(super) struct Weights {
    pub(super) input: Linear,
    /// The convolution's taps, `channels × kernel`.
    taps: Vec<f32>,
    /// `−exp(a_log)`, one per value head.
    decay_rate: Vec<f32>,
    dt_bias: Vector,
    norm: Vector,
    eps: f32,
    pub(super) out: Linear,
    shape: Shape,
}

/// The rule's state in every delta layer: the convolution's last inputs (per layer, channel-major,
/// `history` per channel, oldest first) and each value head's memory.
pub(super) struct State {
    conv: SharedBuf,
    memory: SharedBuf,
}

impl State {
    pub(super) fn new(shape: &Shape, layers: usize) -> Self {
        Self {
            conv: SharedBuf::new(layers * shape.channels() * shape.history()),
            memory: SharedBuf::new(layers * shape.value_heads * shape.memory()),
        }
    }

    pub(super) fn clear(&mut self) {
        self.conv.get_mut().fill(0.0);
        self.memory.get_mut().fill(0.0);
    }

    pub(super) fn copy_from(&mut self, other: &State) {
        self.conv.get_mut().copy_from_slice(other.conv.read());
        self.memory.get_mut().copy_from_slice(other.memory.read());
    }
}

/// One thread's query, key and reading of one head.
#[derive(Default)]
pub(super) struct Scratch {
    query: Vec<f32>,
    key: Vec<f32>,
    reading: Vec<f32>,
}

/// One call's rule: layer `slot` of the state, and where the convolution's output goes.
pub(super) struct Call<'call> {
    pub(super) state: &'call State,
    pub(super) slot: usize,
    pub(super) convolved: &'call SharedBuf,
    pub(super) scratch: &'call PerThread<Scratch>,
}

impl Weights {
    pub(super) fn load(
        store: &mut Store,
        prefix: &str,
        shape: &Shape,
        hidden: usize,
        eps: f32,
    ) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        let taps: Table = scope.load("conv", [shape.channels(), shape.kernel])?;
        let a_log: Vector = scope.load("a_log", shape.value_heads)?;
        Ok(Self {
            input: scope.load("input", [shape.input(), hidden])?,
            taps: taps.dequantize(),
            decay_rate: a_log.values().iter().map(|a_log| -a_log.exp()).collect(),
            dt_bias: scope.load("dt_bias", shape.value_heads)?,
            norm: scope.load("norm", shape.value_dim)?,
            eps,
            out: scope.load("out", [hidden, shape.values()])?,
            shape: *shape,
        })
    }

    /// `buffers.attention`: the rule over the normed rows `input` (prepared for `input`).
    pub(super) fn mix(&self, pass: &Pass, input: &Input, call: &Call) {
        let width = self.shape.input();
        let qkv = &pass.buffers.qkv;
        let cols = pass.ctx.split(width, COLUMN_ALIGN);
        self.input.project_quads(input, cols, |row, first, values| {
            let at = row * width + first;
            qkv.part(at..at + QUAD).copy_from_slice(&values);
        });
        pass.ctx.barrier();
        self.convolve(pass, call);
        pass.ctx.barrier();
        self.recur(pass, call);
        pass.ctx.barrier();
    }

    /// This thread's channels of `convolved`: the convolution over the kept inputs and the
    /// call's rows, through SiLU; then the last inputs kept.
    fn convolve(&self, pass: &Pass, call: &Call) {
        let (channels, width) = (self.shape.channels(), self.shape.input());
        let (kernel, history) = (self.shape.kernel, self.shape.history());
        let mine = pass.ctx.split(channels, 1);
        let raw = pass.buffers.qkv.read();
        let first = call.slot * channels * history;
        let kept = call
            .state
            .conv
            .part(first + mine.start * history..first + mine.end * history);
        let input = |kept: &[f32], channel: usize, at: usize| match at.checked_sub(history) {
            Some(row) => raw[row * width + channel],
            None => kept[at],
        };
        for row in 0..pass.rows {
            let at = row * channels;
            let out = call.convolved.part(at + mine.start..at + mine.end);
            for (offset, channel) in mine.clone().enumerate() {
                let taps = &self.taps[channel * kernel..(channel + 1) * kernel];
                let kept = &kept[offset * history..(offset + 1) * history];
                let sum = (0..kernel).map(|tap| taps[tap] * input(kept, channel, row + tap));
                out[offset] = silu(sum.sum());
            }
        }
        for (offset, channel) in mine.enumerate() {
            let kept = &mut kept[offset * history..(offset + 1) * history];
            for slot in 0..history {
                kept[slot] = input(kept, channel, pass.rows + slot);
            }
        }
    }

    /// This thread's value heads of `buffers.attention`: the rule through the call's rows.
    fn recur(&self, pass: &Pass, call: &Call) {
        let Shape {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        } = self.shape;
        let (keys, values) = (self.shape.keys(), self.shape.values());
        let (channels, width) = (self.shape.channels(), self.shape.input());
        let (convolved, raw) = (call.convolved.read(), pass.buffers.qkv.read());
        let scratch = call.scratch.get(pass.ctx);
        scratch.query.resize(key_dim, 0.0);
        scratch.key.resize(key_dim, 0.0);
        scratch.reading.resize(value_dim, 0.0);
        let scale = 1.0 / (key_dim as f32).sqrt();
        for head in pass.ctx.split(value_heads, 1) {
            let key_head = head / (value_heads / key_heads);
            let at = (call.slot * value_heads + head) * self.shape.memory();
            let memory = call.state.memory.part(at..at + self.shape.memory());
            for row in 0..pass.rows {
                let qkv = &convolved[row * channels..(row + 1) * channels];
                let query = &qkv[key_head * key_dim..][..key_dim];
                normalize(query, scale, &mut scratch.query);
                normalize(
                    &qkv[keys..][key_head * key_dim..][..key_dim],
                    1.0,
                    &mut scratch.key,
                );
                let gates = &raw[row * width + channels..(row + 1) * width];
                let rate =
                    softplus(gates[values + value_heads + head] + self.dt_bias.values()[head]);
                let rule = Rule {
                    decay: (self.decay_rate[head] * rate).exp(),
                    correction: sigmoid(gates[values + head]),
                };
                let value = &qkv[2 * keys + head * value_dim..][..value_dim];
                rule.step(memory, value, scratch);
                let at = row * values + head * value_dim;
                let out = pass.buffers.attention.part(at..at + value_dim);
                rms_norm(&scratch.reading, self.norm.values(), self.eps, out);
                let z = &gates[head * value_dim..(head + 1) * value_dim];
                for (value, z) in out.iter_mut().zip(z) {
                    *value *= silu(*z);
                }
            }
        }
    }
}

/// One row's step of a head's memory.
struct Rule {
    decay: f32,
    correction: f32,
}

impl Rule {
    /// `memory`, decayed, corrected toward `value` by its recall of the key, then read by the
    /// query into `scratch.reading`, `DOTS` rows at a time (`value_dim` is a multiple of `DOTS`,
    /// checked on load). Its own function: V8 optimizes a function on a later call, never a loop
    /// already running (knowledge.md).
    #[inline(never)]
    fn step(&self, memory: &mut [f32], value: &[f32], scratch: &mut Scratch) {
        let Scratch {
            query,
            key,
            reading,
        } = scratch;
        let width = key.len();
        let groups = memory
            .chunks_exact_mut(DOTS * width)
            .zip(value.as_chunks::<DOTS>().0);
        for ((group, values), readings) in groups.zip(reading.as_chunks_mut::<DOTS>().0) {
            let (r0, rest) = group.split_at_mut(width);
            let (r1, rest) = rest.split_at_mut(width);
            let (r2, r3) = rest.split_at_mut(width);
            let dots = dot_f32_many(key, [r0, r1, r2, r3]);
            let deltas = std::array::from_fn(|row| {
                let recalled = self.decay * dots[row];
                (values[row] - recalled) * self.correction
            });
            decay_rows([r0, r1, r2, r3], self.decay, deltas, key);
            *readings = dot_f32_many(query, [r0, r1, r2, r3]);
        }
    }
}

/// `out = x / √(Σx² + ε) · scale`.
fn normalize(x: &[f32], scale: f32, out: &mut [f32]) {
    let inverse = 1.0 / (dot_f32(x, x) + L2_EPS).sqrt();
    for (out, value) in out.iter_mut().zip(x) {
        *out = value * inverse * scale;
    }
}

fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_LINEAR {
        x
    } else {
        x.exp().ln_1p()
    }
}
