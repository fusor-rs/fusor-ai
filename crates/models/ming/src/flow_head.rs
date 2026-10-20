//! Ming's flow-matching head and aggregator: two DiT stacks (`blocks::dit`, 8 blocks of width
//! 1024) wired the way the checkpoint was trained.
//!
//! The flow head samples one 64-d latent per frame from the LLM's conditioning vector `z` and the
//! previous latent: its sequence is `[t_emb + c_emb(z), x_emb(history), x_emb(x)]`, and a guided
//! step batches the unconditional sequence (`c = 0`, so `t_emb + c_embedder.bias`) with it. The
//! timesteps are EPSS knots warped by sway sampling, `t_emb` a sinusoid through a two-layer MLP.
//! The aggregator turns a latent into the LLM's next input embedding: `[cls, x_emb(latent)]`,
//! projected from the `cls` row.

use blocks::dit::{self, Dit, FinalLayer, Sequences};
use blocks::flow::{self, Schedule};
use bundle::{Load, Store};
use kernels::ops::silu;
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, Tensor, matmul};
use team::{Ctx, PerThread, SharedBuf};

use crate::model::HIDDEN;

pub const LATENT: usize = 64;
pub const DIT_SHAPE: dit::Shape = dit::Shape {
    hidden: 1024,
    heads: 16,
    head_dim: 64,
    feed_forward: 4096,
    depth: 8,
    rope_base: 1e4,
    max_sequence: FLOW_SEQUENCE,
};
const FLOW_SEQUENCE: usize = 3;
const AGGREGATOR_SEQUENCE: usize = 2;
/// Rows of one guided flow-head pass: the conditional and the unconditional sequence.
const MAX_ROWS: usize = 2 * FLOW_SEQUENCE;
/// The x rows of the conditional and unconditional sequences, whose outputs are the velocities.
const X_ROWS: [usize; 2] = [FLOW_SEQUENCE - 1, 2 * FLOW_SEQUENCE - 1];
/// Sway coefficient of the reference sampler.
const SWAY: f32 = -1.0;
/// EPSS knots are in units of 1/32.
const EPSS_UNIT: f32 = 32.0;
/// `SinusPositionEmbedding(256)` at scale 1000.
const TIMESTEP_HALF: usize = 128;
const TIMESTEP_SCALE: f32 = 1000.0;

pub(crate) struct FlowHead {
    x_embed: Linear,
    c_embed: Linear,
    t_mlp0: Linear,
    t_mlp2: Linear,
    dit: Dit,
    final_layer: FinalLayer,
}

pub(crate) struct Aggregator {
    cls: Tensor,
    x_embed: Linear,
    dit: Dit,
    final_layer: FinalLayer,
}

/// Activations shared by the flow head and the aggregator.
pub(crate) struct Buffers {
    dit: dit::Buffers,
    /// The flow head's conditioning row `c_emb(z)` and history row, per frame.
    cond: SharedBuf,
    history: SharedBuf,
    /// The flow head's current sample, identical on every thread, and its quantization scratch.
    samples: PerThread<(Vec<f32>, QuantizedRows)>,
}

impl Buffers {
    pub(crate) fn new(threads: usize) -> Self {
        let width = DIT_SHAPE.hidden;
        Self {
            dit: dit::Buffers::new(DIT_SHAPE, MAX_ROWS, HIDDEN, threads),
            cond: SharedBuf::new(width),
            history: SharedBuf::new(width),
            samples: PerThread::new(threads, Default::default),
        }
    }

    pub(crate) fn output(&self, len: usize) -> &[f32] {
        self.dit.output(len)
    }
}

/// A preset's timesteps and guidance scale, with the per-step constants of the conditioning
/// tokens.
pub(crate) struct Steps {
    schedule: Schedule,
    scale: f32,
    /// `t_emb[s]` and `t_emb[s] + c_embedder.bias` (the unconditional token), per step.
    t_emb: Vec<f32>,
    uncond: Vec<f32>,
}

/// What one frame's sampling starts from.
pub(crate) struct FrameInput<'frame> {
    pub(crate) z: &'frame [f32],
    pub(crate) history: &'frame [f32],
    pub(crate) noise: &'frame [f32],
}

impl FlowHead {
    /// The steps for EPSS `knots`, the guided step set (`guided(step, t)`) and the guidance
    /// scale.
    pub(crate) fn steps(
        &self,
        epss: &[u32],
        guided: impl Fn(usize, f32) -> bool,
        scale: f32,
    ) -> Steps {
        let width = DIT_SHAPE.hidden;
        let knots = epss
            .iter()
            .map(|&knot| flow::sway(knot as f32 / EPSS_UNIT, SWAY))
            .collect();
        let schedule = Schedule::new(knots, guided);
        let steps = schedule.guided.len();
        let mut t_emb = vec![0.0; steps * width];
        let mut hidden = vec![0.0; width];
        let mut quant = QuantizedRows::default();
        for (time, emb) in schedule.knots.iter().zip(t_emb.chunks_exact_mut(width)) {
            let sinusoid = flow::timestep_embedding(*time, TIMESTEP_HALF, TIMESTEP_SCALE);
            self.t_mlp0.apply(&sinusoid, 1, &mut quant, &mut hidden);
            hidden.iter_mut().for_each(|value| *value = silu(*value));
            self.t_mlp2.apply(&hidden, 1, &mut quant, emb);
        }
        let c_bias = self.c_embed.bias().expect("c_embedder has a bias");
        let per_step = t_emb.chunks_exact(width);
        let uncond = per_step
            .flat_map(|step| step.iter().zip(c_bias).map(|(time, bias)| time + bias))
            .collect();
        Steps {
            schedule,
            scale,
            t_emb,
            uncond,
        }
    }

    /// One frame: integrates the flow from `input.noise` to the frame latent, written to
    /// `buffers.output(LATENT)` by the first thread.
    pub(crate) fn sample(&self, ctx: &Ctx, buffers: &Buffers, input: &FrameInput, steps: &Steps) {
        self.condition(ctx, buffers, input);
        ctx.barrier();
        let schedule = &steps.schedule;
        for (step, &guided) in schedule.guided.iter().enumerate() {
            let sequences = if guided { 2 } else { 1 };
            self.fill_rows(ctx, buffers, steps, step);
            ctx.barrier();
            let rows = Sequences {
                rows: sequences * FLOW_SEQUENCE,
                length: FLOW_SEQUENCE,
            };
            self.dit.forward(ctx, &buffers.dit, rows);
            self.dit
                .project(ctx, &buffers.dit, &self.final_layer, &X_ROWS[..sequences]);
            ctx.barrier();
            let out = buffers.dit.out.read();
            let unconditional = guided.then(|| &out[LATENT..2 * LATENT]);
            let (sample, _) = buffers.samples.get(ctx);
            flow::euler_step(
                sample,
                &out[..LATENT],
                unconditional,
                steps.scale,
                schedule.dt(step),
            );
            // The next step's projection overwrites `out`: every thread must have read it.
            ctx.barrier();
        }
        if ctx.is_first() {
            let (sample, _) = buffers.samples.get(ctx);
            buffers.dit.out.part(0..LATENT).copy_from_slice(sample);
        }
    }

    /// `c_emb(z)` and `x_emb(history)` for this thread's columns; the sample starts as the noise.
    fn condition(&self, ctx: &Ctx, buffers: &Buffers, input: &FrameInput) {
        let matrix = self.c_embed.matrix();
        let c_bias = self.c_embed.bias().expect("c_embedder has a bias");
        let (sample, quant) = buffers.samples.get(ctx);
        let z = quant.input(&matrix, input.z, 1);
        let cols = ctx.split(DIT_SHAPE.hidden, COLUMN_ALIGN);
        matmul(&matrix, &z, 1, cols.clone(), |_, col, value| {
            buffers.cond.set(col, value + c_bias[col])
        });
        let matrix = self.x_embed.matrix();
        let x_bias = self.x_embed.bias().expect("x_embedder has a bias");
        let history = quant.input(&matrix, input.history, 1);
        matmul(&matrix, &history, 1, cols, |_, col, value| {
            buffers.history.set(col, value + x_bias[col])
        });
        sample.clear();
        sample.extend_from_slice(input.noise);
    }

    /// Rows `[t_emb + c, history, x_emb(y)]` (and the unconditional sequence), own columns.
    fn fill_rows(&self, ctx: &Ctx, buffers: &Buffers, steps: &Steps, step: usize) {
        let guided = steps.schedule.guided[step];
        let width = DIT_SHAPE.hidden;
        let t_emb = &steps.t_emb[step * width..(step + 1) * width];
        let uncond = &steps.uncond[step * width..(step + 1) * width];
        let (cond, history) = (buffers.cond.read(), buffers.history.read());
        let hidden = &buffers.dit.hidden;
        for col in ctx.split(width, COLUMN_ALIGN) {
            hidden.set(col, t_emb[col] + cond[col]);
            hidden.set(width + col, history[col]);
            if guided {
                hidden.set(3 * width + col, uncond[col]);
                hidden.set(4 * width + col, history[col]);
            }
        }
        let (sample, _) = buffers.samples.get(ctx);
        for sequence in 0..if guided { 2 } else { 1 } {
            buffers
                .dit
                .embed_row(ctx, &self.x_embed, sample, sequence * FLOW_SEQUENCE + 2);
        }
    }
}

impl Aggregator {
    /// The next LLM input embedding for `latent`, written to `buffers.output(HIDDEN)`.
    pub(crate) fn embed(&self, ctx: &Ctx, buffers: &Buffers, latent: &[f32]) {
        let cls = self.cls.values().expect("cls is f32");
        for col in ctx.split(DIT_SHAPE.hidden, COLUMN_ALIGN) {
            buffers.dit.hidden.set(col, cls[col]);
        }
        buffers.dit.embed_row(ctx, &self.x_embed, latent, 1);
        ctx.barrier();
        let rows = Sequences {
            rows: AGGREGATOR_SEQUENCE,
            length: AGGREGATOR_SEQUENCE,
        };
        self.dit.forward(ctx, &buffers.dit, rows);
        self.dit.project(ctx, &buffers.dit, &self.final_layer, &[0]);
    }
}

/// `p.x_embed`, `p.c_embed`, `p.t_mlp0`, `p.t_mlp2`, the DiT stack under `p` and its final layer.
impl Load for FlowHead {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        Ok(Self {
            x_embed: Linear::load(store, &name("x_embed"), ())?,
            c_embed: Linear::load(store, &name("c_embed"), ())?,
            t_mlp0: Linear::load(store, &name("t_mlp0"), ())?,
            t_mlp2: Linear::load(store, &name("t_mlp2"), ())?,
            dit: Dit::load(store, prefix, DIT_SHAPE)?,
            final_layer: FinalLayer::load(store, prefix, ())?,
        })
    }
}

/// `p.cls`, `p.x_embed`, the DiT stack under `p` and its final layer.
impl Load for Aggregator {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        Ok(Self {
            cls: Tensor::load(store, &format!("{prefix}.cls"), ())?,
            x_embed: Linear::load(store, &format!("{prefix}.x_embed"), ())?,
            dit: Dit::load(store, prefix, DIT_SHAPE)?,
            final_layer: FinalLayer::load(store, prefix, ())?,
        })
    }
}
