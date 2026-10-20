//! The flow-matching head (DiT + Euler/CFG sampler) and the aggregator (plan §1.3, §1.4).
//!
//! Both are stacks of 8 `DiTBlock`s (hidden 1024, 16 heads of 64, pre-RMSNorm, non-causal
//! attention, GELU-tanh MLP of 4096) over short sequences, with x-transformers' interleaved
//! RoPE (base 1e4) at positions `0..len`. The flow head's sequence is
//! `[t_emb + c_emb(z), x_emb(history), x_emb(x)]`; under CFG a second sequence with `c = 0`
//! (so `t_emb + c_embedder.bias`) is batched with it. The aggregator's is `[cls, x_emb(latent)]`.

use kernels::ops::{
    attend, gelu_tanh, rms_norm, rope_angles, rope_interleaved, rope_inv_freq, silu,
};
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, SharedRows, Tensor, matmul, project_residual};
use team::{Ctx, PerThread, SharedBuf};

pub const HIDDEN: usize = 1024;
pub const LATENT: usize = 64;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const QKV: usize = 3 * HIDDEN;
const FF: usize = 4 * HIDDEN;
const ROPE_BASE: f32 = 1e4;
/// The longest sequence: the flow head's three tokens.
const MAX_SEQ: usize = 3;
/// Rows of one flow-head pass: two sequences (conditional, unconditional) of three tokens.
const MAX_ROWS: usize = 2 * MAX_SEQ;
const FLOW_SEQ: usize = 3;
const AGG_SEQ: usize = 2;

pub struct Block {
    pub norm1: Tensor,
    pub qkv: Linear,
    pub out: Linear,
    pub norm2: Tensor,
    pub ff1: Linear,
    pub ff2: Linear,
}

pub struct Dit {
    blocks: Vec<Block>,
    /// cos/sin of position `p`, pair `i`: `[p · HEAD_DIM / 2 + i]`.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

/// Activations of one DiT pass (shared by the flow head and the aggregator).
pub struct Buffers {
    h: SharedBuf,
    qkv: SharedBuf,
    attn: SharedBuf,
    ff: SharedBuf,
    /// Flow head: the conditioning row `c_emb(z)` and the history row, per frame.
    cond: SharedBuf,
    history: SharedBuf,
    /// The final layer's outputs (flow head: one velocity per sequence; aggregator: the embed).
    out: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    normed: Vec<f32>,
    quant: QuantizedRows,
    scores: Vec<f32>,
    /// The flow head's current sample, kept identical on every thread.
    y: Vec<f32>,
}

impl Buffers {
    pub fn new(threads: usize) -> Self {
        Self {
            h: SharedBuf::new(MAX_ROWS * HIDDEN),
            qkv: SharedBuf::new(MAX_ROWS * QKV),
            attn: SharedBuf::new(MAX_ROWS * HIDDEN),
            ff: SharedBuf::new(MAX_ROWS * FF),
            cond: SharedBuf::new(HIDDEN),
            history: SharedBuf::new(HIDDEN),
            out: SharedBuf::new(2 * HIDDEN),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    pub fn output(&self, len: usize) -> &[f32] {
        &self.out.read()[..len]
    }
}

impl Dit {
    pub fn new(blocks: Vec<Block>) -> Self {
        let inv_freq = rope_inv_freq(ROPE_BASE, HEAD_DIM);
        let half = HEAD_DIM / 2;
        let mut rope_cos = vec![0.0; MAX_SEQ * half];
        let mut rope_sin = vec![0.0; MAX_SEQ * half];
        for p in 0..MAX_SEQ {
            rope_angles(
                p,
                &inv_freq,
                &mut rope_cos[p * half..(p + 1) * half],
                &mut rope_sin[p * half..(p + 1) * half],
            );
        }
        Self {
            blocks,
            rope_cos,
            rope_sin,
        }
    }

    /// All blocks over `rows` rows of `buffers.h`: consecutive sequences of `seq` tokens.
    fn forward(&self, ctx: &Ctx, buffers: &Buffers, rows: usize, seq: usize) {
        for block in &self.blocks {
            self.attention_input(ctx, block, buffers, rows, seq);
            ctx.barrier();
            attention(ctx, buffers, rows, seq);
            ctx.barrier();
            let attention = SharedRows {
                values: &buffers.attn,
                rows,
            };
            let quant = &mut buffers.scratch.get(ctx).quant;
            project_residual(ctx, &block.out, attention, &buffers.h, quant);
            ctx.barrier();
            feed_forward(ctx, block, buffers, rows);
            ctx.barrier();
            let hidden = SharedRows {
                values: &buffers.ff,
                rows,
            };
            let quant = &mut buffers.scratch.get(ctx).quant;
            project_residual(ctx, &block.ff2, hidden, &buffers.h, quant);
            ctx.barrier();
        }
    }

    /// QKV for this thread's heads, then RoPE on its q and k heads.
    fn attention_input(
        &self,
        ctx: &Ctx,
        block: &Block,
        buffers: &Buffers,
        rows: usize,
        seq: usize,
    ) {
        let scratch = buffers.scratch.get(ctx);
        normalize(&buffers.h, rows, &block.norm1, &mut scratch.normed);
        let w = block.qkv.matrix();
        let input = scratch.quant.input(&w, &scratch.normed, rows);
        let heads = ctx.split(QKV, HEAD_DIM);
        let bias = block.qkv.bias().expect("qkv has a bias");
        matmul(&w, &input, rows, heads.clone(), |m, n, v| {
            buffers.qkv.set(m * QKV + n, v + bias[n])
        });
        let half = HEAD_DIM / 2;
        for m in 0..rows {
            let p = m % seq;
            let (cos, sin) = (
                &self.rope_cos[p * half..(p + 1) * half],
                &self.rope_sin[p * half..(p + 1) * half],
            );
            for start in heads.clone().step_by(HEAD_DIM).filter(|&s| s < 2 * HIDDEN) {
                rope_interleaved(
                    buffers
                        .qkv
                        .part(m * QKV + start..m * QKV + start + HEAD_DIM),
                    cos,
                    sin,
                );
            }
        }
    }
}

fn normalize(x: &SharedBuf, rows: usize, weight: &Tensor, out: &mut Vec<f32>) {
    out.resize(rows * HIDDEN, 0.0);
    rms_norm(
        &x.read()[..rows * HIDDEN],
        weight.values().expect("norm weights are f32"),
        out,
    );
}

/// Non-causal attention inside each sequence, split over (sequence, head) pairs.
fn attention(ctx: &Ctx, buffers: &Buffers, rows: usize, seq: usize) {
    let scratch = buffers.scratch.get(ctx);
    let qkv = buffers.qkv.read();
    for unit in ctx.split(rows / seq * HEADS, 1) {
        let (sequence, head) = (unit / HEADS, unit % HEADS);
        let first = sequence * seq;
        let at = |row: usize, part: usize| (first + row) * QKV + part * HIDDEN + head * HEAD_DIM;
        for r in 0..seq {
            let out = buffers.attn.part(
                (first + r) * HIDDEN + head * HEAD_DIM
                    ..(first + r) * HIDDEN + (head + 1) * HEAD_DIM,
            );
            attend(
                &qkv[at(r, 0)..at(r, 0) + HEAD_DIM],
                seq,
                |j| {
                    let (key, value) = (at(j, 1), at(j, 2));
                    (&qkv[key..key + HEAD_DIM], &qkv[value..value + HEAD_DIM])
                },
                &mut scratch.scores,
                out,
            );
        }
    }
}

/// `ff[:, cols] = gelu(norm2(h) · W1ᵀ + b1)` for this thread's columns.
fn feed_forward(ctx: &Ctx, block: &Block, buffers: &Buffers, rows: usize) {
    let scratch = buffers.scratch.get(ctx);
    normalize(&buffers.h, rows, &block.norm2, &mut scratch.normed);
    let w = block.ff1.matrix();
    let input = scratch.quant.input(&w, &scratch.normed, rows);
    let bias = block.ff1.bias().expect("ff1 has a bias");
    let cols = ctx.split(FF, COLUMN_ALIGN);
    matmul(&w, &input, rows, cols.clone(), |m, c, v| {
        buffers.ff.set(m * FF + c, v + bias[c])
    });
    for m in 0..rows {
        buffers
            .ff
            .part(m * FF + cols.start..m * FF + cols.end)
            .iter_mut()
            .for_each(|v| *v = gelu_tanh(*v));
    }
}

/// `out[g · n + cols] = final(norm(h[row_g]))` for the given rows, this thread's columns.
fn final_layer(ctx: &Ctx, norm: &Tensor, linear: &Linear, buffers: &Buffers, rows: &[usize]) {
    let scratch = buffers.scratch.get(ctx);
    let h = buffers.h.read();
    scratch.normed.resize(rows.len() * HIDDEN, 0.0);
    for (g, &row) in rows.iter().enumerate() {
        rms_norm(
            &h[row * HIDDEN..(row + 1) * HIDDEN],
            norm.values().expect("norm weights are f32"),
            &mut scratch.normed[g * HIDDEN..(g + 1) * HIDDEN],
        );
    }
    let w = linear.matrix();
    let n = w.rows;
    let input = scratch.quant.input(&w, &scratch.normed, rows.len());
    let bias = linear.bias().expect("final layer has a bias");
    matmul(&w, &input, rows.len(), ctx.split(n, 8), |g, c, v| {
        buffers.out.set(g * n + c, v + bias[c])
    });
}

/// `x_emb(x)[cols]` written into row `row` of `h`, for this thread's columns.
fn embed_latent(ctx: &Ctx, x_embed: &Linear, latent: &[f32], target: &SharedBuf, row: usize) {
    let w = x_embed.matrix();
    let bias = x_embed.bias().expect("x_embedder has a bias");
    let mut quant = QuantizedRows::default();
    let input = quant.input(&w, latent, 1);
    matmul(&w, &input, 1, ctx.split(HIDDEN, COLUMN_ALIGN), |_, c, v| {
        target.set(row * HIDDEN + c, v + bias[c])
    });
}

/// The sampler's timesteps and per-step constants for one preset.
pub struct Schedule {
    /// Knots `t_0 = 0 < … < t_steps = 1` after sway warping.
    pub knots: Vec<f32>,
    /// Whether step `s` evaluates the unconditional branch (CFG); otherwise `v = pred`.
    pub guided: Vec<bool>,
    /// `t_emb[s]` and `t_emb[s] + c_embedder.bias` (the unconditional token), `HIDDEN` each.
    t_emb: Vec<f32>,
    uncond: Vec<f32>,
}

pub struct FlowHead {
    pub x_embed: Linear,
    pub c_embed: Linear,
    pub t_mlp0: Linear,
    pub t_mlp2: Linear,
    pub dit: Dit,
    pub final_norm: Tensor,
    pub final_linear: Linear,
}

/// What one frame's sampling starts from.
pub struct FrameInput<'a> {
    pub z: &'a [f32],
    pub history: &'a [f32],
    pub noise: &'a [f32],
}

impl FlowHead {
    /// The schedule for EPSS `knots` (in units of 1/32) and the guided step set.
    pub fn schedule(&self, epss: &[u32], guided: impl Fn(usize, f32) -> bool) -> Schedule {
        let knots: Vec<f32> = epss.iter().map(|&k| sway(k as f32 / 32.0)).collect();
        let steps = knots.len() - 1;
        let mut t_emb = vec![0.0; steps * HIDDEN];
        let mut hidden = vec![0.0; HIDDEN];
        let mut quant = QuantizedRows::default();
        for (s, emb) in t_emb.as_chunks_mut::<HIDDEN>().0.iter_mut().enumerate() {
            self.t_mlp0
                .apply(&sinusoid(knots[s]), 1, &mut quant, &mut hidden);
            hidden.iter_mut().for_each(|v| *v = silu(*v));
            self.t_mlp2.apply(&hidden, 1, &mut quant, emb);
        }
        let c_bias = self.c_embed.bias().expect("c_embedder has a bias");
        let uncond = t_emb
            .as_chunks::<HIDDEN>()
            .0
            .iter()
            .flat_map(|t| t.iter().zip(c_bias).map(|(a, b)| a + b))
            .collect();
        let guided = (0..steps).map(|s| guided(s, knots[s])).collect();
        Schedule {
            knots,
            guided,
            t_emb,
            uncond,
        }
    }

    /// One frame: integrates the flow from `input.noise` to the frame latent, written to
    /// `buffers.output(LATENT)` by the first thread. `cfg` is the guidance scale.
    pub fn sample(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        input: &FrameInput,
        schedule: &Schedule,
        cfg: f32,
    ) {
        let w = self.c_embed.matrix();
        let c_bias = self.c_embed.bias().expect("c_embedder has a bias");
        let scratch = buffers.scratch.get(ctx);
        let z = scratch.quant.input(&w, input.z, 1);
        matmul(&w, &z, 1, ctx.split(HIDDEN, COLUMN_ALIGN), |_, c, v| {
            buffers.cond.set(c, v + c_bias[c])
        });
        embed_latent(ctx, &self.x_embed, input.history, &buffers.history, 0);
        let y = &mut buffers.scratch.get(ctx).y;
        y.clear();
        y.extend_from_slice(input.noise);
        ctx.barrier();
        for (s, &guided) in schedule.guided.iter().enumerate() {
            let sequences = if guided { 2 } else { 1 };
            self.fill_rows(ctx, buffers, schedule, s, sequences);
            ctx.barrier();
            self.dit
                .forward(ctx, buffers, sequences * FLOW_SEQ, FLOW_SEQ);
            let x_rows = [FLOW_SEQ - 1, 2 * FLOW_SEQ - 1];
            final_layer(
                ctx,
                &self.final_norm,
                &self.final_linear,
                buffers,
                &x_rows[..sequences],
            );
            ctx.barrier();
            let dt = schedule.knots[s + 1] - schedule.knots[s];
            let out = buffers.out.read();
            let y = &mut buffers.scratch.get(ctx).y;
            for (i, y) in y.iter_mut().enumerate() {
                let pred = out[i];
                let v = if guided {
                    pred + (pred - out[LATENT + i]) * cfg
                } else {
                    pred
                };
                *y += v * dt;
            }
            // The next step's final layer overwrites `out`: every thread must have read it.
            ctx.barrier();
        }
        if ctx.is_first() {
            buffers
                .out
                .part(0..LATENT)
                .copy_from_slice(&buffers.scratch.get(ctx).y);
        }
    }

    /// Rows `[t_emb + c, history, x_emb(y)]` (and the unconditional sequence), own columns.
    fn fill_rows(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        schedule: &Schedule,
        step: usize,
        sequences: usize,
    ) {
        let cols = ctx.split(HIDDEN, COLUMN_ALIGN);
        let t = &schedule.t_emb[step * HIDDEN..(step + 1) * HIDDEN];
        let uncond = &schedule.uncond[step * HIDDEN..(step + 1) * HIDDEN];
        let (cond, history) = (buffers.cond.read(), buffers.history.read());
        for c in cols {
            buffers.h.set(c, t[c] + cond[c]);
            buffers.h.set(HIDDEN + c, history[c]);
            if sequences == 2 {
                buffers.h.set(3 * HIDDEN + c, uncond[c]);
                buffers.h.set(4 * HIDDEN + c, history[c]);
            }
        }
        let y = &buffers.scratch.get(ctx).y;
        for sequence in 0..sequences {
            embed_latent(ctx, &self.x_embed, y, &buffers.h, sequence * FLOW_SEQ + 2);
        }
    }
}

/// `t + coef · (cos(π t / 2) − 1 + t)` with the reference's sway coefficient −1.
fn sway(t: f32) -> f32 {
    const SWAY: f32 = -1.0;
    t + SWAY * ((std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
}

/// `SinusPositionEmbedding(256)` at `scale = 1000`: `[sin(1000·t·f_i), cos(1000·t·f_i)]`,
/// `f_i = exp(−i · ln(10000) / 127)`.
fn sinusoid(t: f32) -> Vec<f32> {
    const HALF: usize = 128;
    let step = 10000f32.ln() / (HALF - 1) as f32;
    let angles: Vec<f32> = (0..HALF)
        .map(|i| 1000.0 * t * (i as f32 * -step).exp())
        .collect();
    angles
        .iter()
        .map(|a| a.sin())
        .chain(angles.iter().map(|a| a.cos()))
        .collect()
}

pub struct Aggregator {
    pub cls: Tensor,
    pub x_embed: Linear,
    pub dit: Dit,
    pub final_norm: Tensor,
    pub final_linear: Linear,
}

impl Aggregator {
    /// The next LLM input embedding for `latent`, written to `buffers.output(896)`.
    pub fn embed(&self, ctx: &Ctx, buffers: &Buffers, latent: &[f32]) {
        let cls = self.cls.values().expect("cls is f32");
        for c in ctx.split(HIDDEN, COLUMN_ALIGN) {
            buffers.h.set(c, cls[c]);
        }
        embed_latent(ctx, &self.x_embed, latent, &buffers.h, 1);
        ctx.barrier();
        self.dit.forward(ctx, buffers, AGG_SEQ, AGG_SEQ);
        final_layer(ctx, &self.final_norm, &self.final_linear, buffers, &[0]);
    }
}
