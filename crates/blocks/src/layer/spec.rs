//! What sets one architecture's layer apart from another's, and every architecture's spec.

use bundle::Scope;

use super::{Activation, Norm};

/// What sets one architecture's layer apart from another's: its norms and their tensors, the
/// name of its output projection, its feed-forward, whether it normalizes each query and key
/// head, and how its rotary embeddings pair a head's elements (`None` without them). The other
/// tensors are named alike in every layer: `qkv`; a gated feed-forward's `gate`, `up` and
/// `down`, a plain one's `ff1` and `ff2`; the head norms' `q_norm` and `k_norm`.
#[derive(Clone, Copy)]
pub(crate) struct LayerSpec {
    pub(crate) norm: NormSpec,
    pub(crate) norms: [&'static str; 2],
    pub(crate) out: &'static str,
    pub(crate) feed_forward: FeedForwardSpec,
    pub(crate) head_norm: HeadNorm,
    pub(crate) rope: Option<Rope>,
}

impl LayerSpec {
    pub(crate) const QWEN2: Self = Self {
        norm: NormSpec::Rms,
        norms: ["input_norm", "post_norm"],
        out: "o",
        feed_forward: FeedForwardSpec::Gated(Activation::Silu),
        head_norm: HeadNorm::None,
        rope: Some(Rope::Half),
    };
    pub(crate) const QWEN3: Self = Self {
        head_norm: HeadNorm::Rms,
        ..Self::QWEN2
    };
    /// umT5's encoder layer: RMSNorms, a bias-free fused QKV and output `o`, a GELU-tanh-gated
    /// feed-forward, no rotary embeddings (its positions are a relative bias added to the scores).
    pub(crate) const UMT5: Self = Self {
        norm: NormSpec::Rms,
        norms: ["norm1", "norm2"],
        out: "o",
        feed_forward: FeedForwardSpec::Gated(Activation::GeluTanh),
        head_norm: HeadNorm::None,
        rope: None,
    };
    pub(crate) const KYUTAI: Self = Self::plain(
        NormSpec::Layer,
        Activation::GeluErf,
        Some(Rope::Interleaved),
    );
    pub(crate) const CLIP: Self = Self::plain(NormSpec::Layer, Activation::QuickGelu, None);
    pub(crate) const ENCODER: Self = Self::plain(NormSpec::Layer, Activation::GeluErf, None);
    pub(crate) const VIT: Self =
        Self::plain(NormSpec::Layer, Activation::GeluTanh, Some(Rope::Half));
    pub(crate) const DIT: Self =
        Self::plain(NormSpec::Rms, Activation::GeluTanh, Some(Rope::Interleaved));
    /// Iris's layer-wise text blocks: the DiT layer with a SiLU MLP and no rotary embeddings.
    pub(crate) const LAYERWISE: Self = Self::plain(NormSpec::Rms, Activation::Silu, None);

    /// The layer of Kyutai's transformer, CLIP's text encoder, the encoder and the ViT, DiT and
    /// layer-wise blocks: `norm1`, `out`, `norm2` and a plain feed-forward with `activation`.
    const fn plain(norm: NormSpec, activation: Activation, rope: Option<Rope>) -> Self {
        Self {
            norm,
            norms: ["norm1", "norm2"],
            out: "out",
            feed_forward: FeedForwardSpec::Plain(activation),
            head_norm: HeadNorm::None,
            rope,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum NormSpec {
    Rms,
    Layer,
}

impl NormSpec {
    pub(super) fn load(
        self,
        scope: &mut Scope,
        name: &str,
        hidden: usize,
        eps: f32,
    ) -> Result<Norm, bundle::Error> {
        Ok(match self {
            NormSpec::Rms => Norm::Rms {
                weight: scope.load(name, hidden)?,
                eps,
            },
            NormSpec::Layer => Norm::Layer {
                norm: scope.load(name, hidden)?,
                eps,
            },
        })
    }
}

/// A layer's `FeedForward` and its activation.
#[derive(Clone, Copy)]
pub(crate) enum FeedForwardSpec {
    Gated(Activation),
    Plain(Activation),
}

/// Whether a layer RMS-normalizes each query and key head before the rotary embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadNorm {
    None,
    Rms,
}

/// How rotary embeddings pair the elements of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rope {
    /// Element `i` with `i + d/2` (`rotate_half`, Llama/Qwen).
    Half,
    /// Element `2i` with `2i + 1` (x-transformers).
    Interleaved,
}
