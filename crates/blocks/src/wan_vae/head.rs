//! The head after the last level of a decoder, and after an encoder's middle: a per-pixel norm,
//! SiLU and a causal convolution.
//!
//! Tensors under a prefix `p`: the vector `p.head.norm` and the convolution `p.head.conv`.

use bundle::{Load, Scope, Store};
use kernels::Vector;
use kernels::conv::{CausalConv3d, CausalShape};
use kernels::ops::silu;

use super::walk::Walk;

pub(super) struct Head {
    norm: Vector,
    conv: CausalConv3d,
}

impl Head {
    /// `store` of the head over the current plane, which stays the current one.
    pub(super) fn forward(&self, walk: &mut Walk, store: impl FnMut(usize, usize, f32)) {
        let [normed, _] = walk.free();
        walk.norm((walk.current, normed), &self.norm, silu);
        walk.causal(&self.conv, normed, store);
    }
}

impl Load for Head {
    /// The channels the norm reads and the convolution's dimensions.
    type Shape = (usize, CausalShape);

    fn load(
        store: &mut Store,
        prefix: &str,
        (channels, conv): (usize, CausalShape),
    ) -> Result<Self, bundle::Error> {
        let head_prefix = format!("{prefix}.head");
        let mut scope = Scope::new(store, &head_prefix);
        Ok(Self {
            norm: scope.load("norm", channels)?,
            conv: scope.load("conv", conv)?,
        })
    }
}
