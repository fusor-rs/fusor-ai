//! A layer's weights, loaded as its spec names and builds them.

use bundle::{Scope, Store};
use kernels::ops::{gelu_erf, gelu_tanh_sweep, quick_gelu, rms_norm, silu_sweep};
use kernels::{Linear, Vector};

use super::{FeedForwardSpec, HeadNorm, LayerSpec, Widths};
use crate::norm::LayerNorm;

pub(crate) enum Norm {
    Rms { weight: Vector, eps: f32 },
    Layer { norm: LayerNorm, eps: f32 },
}

impl Norm {
    pub(crate) fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self {
            Norm::Rms { weight, eps } => rms_norm(x, weight.values(), *eps, out),
            Norm::Layer { norm, eps } => norm.apply(x, *eps, out),
        }
    }
}

pub(crate) enum FeedForward {
    /// `down(activation(gate(x)) · up(x))`.
    Gated {
        gate: Linear,
        up: Linear,
        down: Linear,
        activation: Activation,
    },
    /// `down(activation(up(x)))`.
    Plain {
        up: Linear,
        down: Linear,
        activation: Activation,
    },
}

/// A feed-forward's pointwise activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
    GeluErf,
    QuickGelu,
}

impl Activation {
    /// Of every value of `values`, in place.
    ///
    /// Never inlined: V8 optimizes a wasm function on a later call, never a loop already
    /// running, and the feed-forward sweeps a row at a time from a function a layer enters
    /// once.
    #[inline(never)]
    pub(super) fn sweep(self, values: &mut [f32]) {
        match self {
            Activation::Silu => silu_sweep(values),
            Activation::GeluTanh => gelu_tanh_sweep(values),
            Activation::GeluErf => values.iter_mut().for_each(|x| *x = gelu_erf(*x)),
            Activation::QuickGelu => values.iter_mut().for_each(|x| *x = quick_gelu(*x)),
        }
    }
}

impl FeedForward {
    /// The gated feed-forward under `scope`: `gate`, `up` and `down`, `width` wide inside.
    pub(crate) fn gated(
        scope: &mut Scope,
        hidden: usize,
        width: usize,
        activation: Activation,
    ) -> Result<Self, bundle::Error> {
        Ok(FeedForward::Gated {
            gate: scope.load("gate", [width, hidden])?,
            up: scope.load("up", [width, hidden])?,
            down: scope.load("down", [hidden, width])?,
            activation,
        })
    }

    /// The projection that reads `norm2(hidden)` (the gate's input is the up's too).
    pub(super) fn first(&self) -> &Linear {
        match self {
            FeedForward::Gated { gate, .. } => gate,
            FeedForward::Plain { up, .. } => up,
        }
    }

    pub(super) fn down(&self) -> &Linear {
        match self {
            FeedForward::Gated { down, .. } | FeedForward::Plain { down, .. } => down,
        }
    }
}

pub(crate) struct Layer {
    pub(crate) norm1: Norm,
    pub(crate) qkv: Linear,
    pub(crate) head_norms: Option<HeadNorms>,
    pub(crate) out: Linear,
    pub(crate) norm2: Norm,
    pub(crate) feed_forward: FeedForward,
}

impl Layer {
    /// The layer under `prefix`, as `spec` names and builds it, `widths` wide.
    pub(crate) fn load(
        store: &mut Store,
        prefix: &str,
        spec: &LayerSpec,
        widths: &Widths,
        eps: f32,
    ) -> Result<Self, bundle::Error> {
        let Widths {
            hidden,
            heads,
            feed_forward,
            ..
        } = *widths;
        let mut scope = Scope::new(store, prefix);
        let [norm1, norm2] = spec.norms;
        let norm1 = spec.norm.load(&mut scope, norm1, hidden, eps)?;
        let qkv = scope.load("qkv", [heads.qkv(), hidden])?;
        let out = scope.load(spec.out, [hidden, heads.width()])?;
        let norm2 = spec.norm.load(&mut scope, norm2, hidden, eps)?;
        let feed_forward = match spec.feed_forward {
            FeedForwardSpec::Gated(activation) => {
                FeedForward::gated(&mut scope, hidden, feed_forward, activation)?
            }
            FeedForwardSpec::Plain(activation) => FeedForward::Plain {
                up: scope.load("ff1", [feed_forward, hidden])?,
                down: scope.load("ff2", [hidden, feed_forward])?,
                activation,
            },
        };
        let head_norms = match spec.head_norm {
            HeadNorm::None => None,
            HeadNorm::Rms => Some(HeadNorms {
                query: scope.load("q_norm", heads.head_dim)?,
                key: scope.load("k_norm", heads.head_dim)?,
                eps,
            }),
        };
        Ok(Self {
            norm1,
            qkv,
            head_norms,
            out,
            norm2,
            feed_forward,
        })
    }
}

/// RMSNorms over each head of the queries and of the keys, `head_dim` wide.
pub(crate) struct HeadNorms {
    pub(crate) query: Vector,
    pub(crate) key: Vector,
    pub(crate) eps: f32,
}
