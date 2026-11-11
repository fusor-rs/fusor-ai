//! The dual-stream block: the stream and the context each their own weights, one attention over
//! all rows.

use bundle::{Load, Scope, Store};
use kernels::ops::gelu_tanh;
use kernels::{Linear, Vector};
use team::{Ctx, SharedBuf};

use super::pass::{HeadNorms, Modulation, Pass, modulate};
use super::{Buffers, DualNorm, Rows, Shape};
use crate::rotary::Rotary;

/// One stream's weights.
struct Weights {
    qkv: Linear,
    out: Linear,
    ff1: Linear,
    ff2: Linear,
    q_norm: Vector,
    k_norm: Vector,
}

impl Weights {
    fn norms(&self) -> HeadNorms<'_> {
        HeadNorms {
            query: &self.q_norm,
            key: &self.k_norm,
        }
    }

    /// The weights named `{tag}qkv` and so on.
    fn load(scope: &mut Scope, tag: &str, shape: &Shape) -> Result<Self, bundle::Error> {
        let Shape {
            hidden,
            feed_forward,
            ..
        } = *shape;
        let head_dim = shape.head_dim();
        Ok(Self {
            qkv: scope.load(&format!("{tag}qkv"), [3 * hidden, hidden])?,
            out: scope.load(&format!("{tag}out"), [hidden, hidden])?,
            ff1: scope.load(&format!("{tag}ff1"), [feed_forward, hidden])?,
            ff2: scope.load(&format!("{tag}ff2"), [hidden, feed_forward])?,
            q_norm: scope.load(&format!("{tag}q_norm"), head_dim)?,
            k_norm: scope.load(&format!("{tag}k_norm"), head_dim)?,
        })
    }
}

pub struct Dual {
    shape: Shape,
    rotary: Rotary,
    modulation: Linear,
    stream: Weights,
    context: Weights,
}

impl Dual {
    /// This block's modulations of `time` (the SiLU of the time embedding) into `out`, every scale
    /// as `1 +` it; this thread's columns, no barrier.
    pub fn modulate(&self, ctx: &Ctx, buffers: &Buffers, (time, out): (&[f32], &SharedBuf)) {
        modulate(ctx, buffers, &self.modulation, (time, out));
    }

    /// The block over `rows` of `buffers.hidden` (in and out) under `modulation` (its
    /// `modulate`'s).
    ///
    /// # Panics
    /// When the buffers hold fewer rows than `rows` has.
    pub fn forward(&self, ctx: &Ctx, buffers: &Buffers, modulation: &[f32], rows: Rows) {
        let pass = Pass::new(ctx, buffers, (&self.shape, &self.rotary), rows);
        let norm = |norm: DualNorm| Modulation::of(modulation, self.shape.hidden, norm as usize);
        let attention = (
            &norm(DualNorm::StreamAttention),
            &norm(DualNorm::ContextAttention),
        );
        self.attention(&pass, attention);
        let feed_forward = (
            &norm(DualNorm::StreamFeedForward),
            &norm(DualNorm::ContextFeedForward),
        );
        self.feed_forward(&pass, feed_forward);
    }

    fn attention(&self, pass: &Pass, (stream, context): (&Modulation, &Modulation)) {
        let buffers = pass.buffers;
        let width = 3 * self.shape.hidden;
        pass.modulated_norm(stream, context);
        let weights = (&self.stream.qkv, &self.context.qkv);
        pass.product(weights, &buffers.normed, |row, col, value| {
            buffers.qkv.set(row * width + col, value);
        });
        pass.ctx.barrier();
        pass.head_norms(self.stream.norms(), self.context.norms());
        pass.joint_attention();
        let weights = (&self.stream.out, &self.context.out);
        pass.gated_streams(weights, &buffers.attention, (stream.gate, context.gate));
    }

    fn feed_forward(&self, pass: &Pass, (stream, context): (&Modulation, &Modulation)) {
        let buffers = pass.buffers;
        let width = self.shape.feed_forward;
        pass.modulated_norm(stream, context);
        let weights = (&self.stream.ff1, &self.context.ff1);
        pass.product(weights, &buffers.normed, |row, col, value| {
            buffers.inner.set(row * width + col, gelu_tanh(value));
        });
        pass.ctx.barrier();
        let weights = (&self.stream.ff2, &self.context.ff2);
        pass.gated_streams(weights, &buffers.inner, (stream.gate, context.gate));
    }
}

impl Load for Dual {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let rotary = shape.rotary(prefix)?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            rotary,
            modulation: scope.load("modulation", [shape.dual_modulations(), shape.hidden])?,
            stream: Weights::load(&mut scope, "", &shape)?,
            context: Weights::load(&mut scope, "context_", &shape)?,
        })
    }
}
