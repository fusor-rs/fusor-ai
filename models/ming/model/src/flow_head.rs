//! Ming's flow-matching head and aggregator: two DiT stacks (`blocks::dit`, 8 blocks of width
//! 1024) wired the way the checkpoint was trained.
//!
//! The flow head samples one 64-d latent per frame from the LLM's conditioning vector `z` and the
//! previous latent: its sequence is `[t_emb + c_emb(z), x_emb(history), x_emb(x)]`, and a guided
//! step batches the unconditional sequence (`c = 0`, so `t_emb + c_embedder.bias`) with it. The
//! timesteps are EPSS knots warped by sway sampling, `t_emb` a sinusoid through a two-layer MLP.
//! The aggregator turns a latent into the LLM's next input embedding: `[cls, x_emb(latent)]`,
//! projected from the `cls` row.

use blocks::dit::{self, Buffers as DitBuffers, Dit, FinalLayer, Projection, Sequences};
use blocks::flow::{self, Guidance, Schedule};
use bundle::{Load, Scope, Store};
use kernels::ops::silu;
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, Vector};
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
    norm_eps: 1e-6,
};
const FLOW_SEQUENCE: usize = 3;
/// The tokens of a flow sequence, in order.
const CONDITION_ROW: usize = 0;
const HISTORY_ROW: usize = 1;
const SAMPLE_ROW: usize = 2;
const AGGREGATOR_SEQUENCE: usize = 2;
/// The sequences of a guided step: the conditional, then the unconditional.
const GUIDED_SEQUENCES: usize = 2;
/// Rows of one guided flow-head pass.
const MAX_ROWS: usize = GUIDED_SEQUENCES * FLOW_SEQUENCE;
/// The sample rows of the conditional and unconditional sequences, whose outputs are the
/// velocities.
const X_ROWS: [usize; GUIDED_SEQUENCES] = [SAMPLE_ROW, FLOW_SEQUENCE + SAMPLE_ROW];
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
    cls: Vector,
    x_embed: Linear,
    dit: Dit,
    final_layer: FinalLayer,
}

/// Activations shared by the flow head and the aggregator.
pub(crate) struct Buffers {
    dit: DitBuffers,
    /// The final layers' outputs: the velocities of a step, or the aggregator's embedding.
    out: SharedBuf,
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
            dit: DIT_SHAPE.buffers(MAX_ROWS, threads),
            out: SharedBuf::new(HIDDEN.max(X_ROWS.len() * LATENT)),
            cond: SharedBuf::new(width),
            history: SharedBuf::new(width),
            samples: PerThread::new(threads, Default::default),
        }
    }

    pub(crate) fn output(&self, len: usize) -> &[f32] {
        &self.out.read()[..len]
    }
}

/// A preset's timesteps and guidance scale, with the per-step constants of the conditioning
/// tokens.
pub(crate) struct Steps {
    schedule: Schedule,
    scale: f32,
    /// `t_emb[s]` and `t_emb[s] + c_emb(0)` (the unconditional token), per step.
    t_emb: Vec<f32>,
    uncond: Vec<f32>,
}

impl Steps {
    /// The sequences step `step` runs: the unconditional one too when it is guided.
    fn sequences(&self, step: usize) -> usize {
        if self.schedule.guided(step) {
            GUIDED_SEQUENCES
        } else {
            1
        }
    }
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
        let mut t_emb = vec![0.0; schedule.steps() * width];
        let mut hidden = vec![0.0; width];
        let mut quant = QuantizedRows::default();
        for (time, emb) in schedule.times().iter().zip(t_emb.chunks_exact_mut(width)) {
            let sinusoid = flow::timestep_embedding(*time, TIMESTEP_HALF, TIMESTEP_SCALE);
            self.t_mlp0.apply(&sinusoid, &mut quant, &mut hidden);
            hidden.iter_mut().for_each(|value| *value = silu(*value));
            self.t_mlp2.apply(&hidden, &mut quant, emb);
        }
        let mut unconditioned = vec![0.0; width];
        self.c_embed
            .apply(&[0.0; HIDDEN], &mut quant, &mut unconditioned);
        let per_step = t_emb.chunks_exact(width);
        let uncond = per_step
            .flat_map(|step| {
                step.iter()
                    .zip(&unconditioned)
                    .map(|(time, zero)| time + zero)
            })
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
        for step in 0..schedule.steps() {
            let sequences = steps.sequences(step);
            self.fill_rows(ctx, buffers, steps, step);
            ctx.barrier();
            let rows = Sequences {
                rows: sequences * FLOW_SEQUENCE,
                length: FLOW_SEQUENCE,
            };
            self.dit.forward(ctx, &buffers.dit, rows);
            let projected = &X_ROWS[..sequences];
            self.final_layer
                .project(ctx, &buffers.dit, projected, &buffers.out);
            ctx.barrier();
            let out = buffers.out.read();
            let guidance = schedule.guided(step).then(|| Guidance {
                unconditional: &out[LATENT..2 * LATENT],
                scale: steps.scale,
            });
            let (sample, _) = buffers.samples.get(ctx);
            flow::euler_step(sample, &out[..LATENT], guidance, schedule.dt(step));
            // The next step's projection overwrites `out`: every thread must have read it.
            ctx.barrier();
        }
        if ctx.is_first() {
            let (sample, _) = buffers.samples.get(ctx);
            buffers.out.part(0..LATENT).copy_from_slice(sample);
        }
    }

    /// `c_emb(z)` and `x_emb(history)` for this thread's columns; the sample starts as the noise.
    fn condition(&self, ctx: &Ctx, buffers: &Buffers, input: &FrameInput) {
        let (sample, quant) = buffers.samples.get(ctx);
        let cols = ctx.split(DIT_SHAPE.hidden, COLUMN_ALIGN);
        let embeddings = [
            (&self.c_embed, input.z, &buffers.cond),
            (&self.x_embed, input.history, &buffers.history),
        ];
        for (linear, x, out) in embeddings {
            let x = quant.input(&linear.matrix(), x);
            linear.project(&x, cols.clone(), |_, col, value| out.set(col, value));
        }
        sample.clear();
        sample.extend_from_slice(input.noise);
    }

    /// The rows of each of the step's sequences, `[t_emb + c, history, x_emb(y)]` (`c` zero in
    /// the unconditional one), this thread's columns.
    fn fill_rows(&self, ctx: &Ctx, buffers: &Buffers, steps: &Steps, step: usize) {
        let width = DIT_SHAPE.hidden;
        let t_emb = &steps.t_emb[step * width..(step + 1) * width];
        let uncond = &steps.uncond[step * width..(step + 1) * width];
        let (cond, history) = (buffers.cond.read(), buffers.history.read());
        let hidden = &buffers.dit.hidden;
        let sequences = steps.sequences(step);
        for col in ctx.split(width, COLUMN_ALIGN) {
            for sequence in 0..sequences {
                let token = |row: usize| (sequence * FLOW_SEQUENCE + row) * width + col;
                let condition = match sequence {
                    0 => t_emb[col] + cond[col],
                    _ => uncond[col],
                };
                hidden.set(token(CONDITION_ROW), condition);
                hidden.set(token(HISTORY_ROW), history[col]);
            }
        }
        let (sample, _) = buffers.samples.get(ctx);
        for sequence in 0..sequences {
            let row = sequence * FLOW_SEQUENCE + SAMPLE_ROW;
            buffers.dit.embed_row(ctx, &self.x_embed, sample, row);
        }
    }
}

impl Aggregator {
    /// The next LLM input embedding for `latent`, written to `buffers.output(HIDDEN)`.
    pub(crate) fn embed(&self, ctx: &Ctx, buffers: &Buffers, latent: &[f32]) {
        let cls = self.cls.values();
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
        self.final_layer
            .project(ctx, &buffers.dit, &[0], &buffers.out);
    }
}

/// `p.x_embed`, `p.c_embed`, `p.t_mlp0`, `p.t_mlp2`, the DiT stack under `p` and its final layer.
impl Load for FlowHead {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let width = DIT_SHAPE.hidden;
        let projection = Projection {
            shape: DIT_SHAPE,
            outputs: LATENT,
        };
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            x_embed: scope.load("x_embed", [width, LATENT])?,
            c_embed: scope.load("c_embed", [width, HIDDEN])?,
            t_mlp0: scope.load("t_mlp0", [width, 2 * TIMESTEP_HALF])?,
            t_mlp2: scope.load("t_mlp2", [width, width])?,
            dit: Dit::load(store, prefix, DIT_SHAPE)?,
            final_layer: FinalLayer::load(store, prefix, projection)?,
        })
    }
}

/// `p.cls`, `p.x_embed`, the DiT stack under `p` and its final layer.
impl Load for Aggregator {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let width = DIT_SHAPE.hidden;
        let projection = Projection {
            shape: DIT_SHAPE,
            outputs: HIDDEN,
        };
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            cls: scope.load("cls", width)?,
            x_embed: scope.load("x_embed", [width, LATENT])?,
            dit: Dit::load(store, prefix, DIT_SHAPE)?,
            final_layer: FinalLayer::load(store, prefix, projection)?,
        })
    }
}
