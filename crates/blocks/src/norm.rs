//! LayerNorm with its affine parameters.

use bundle::{Load, Scope, Store};
use kernels::Vector;
use kernels::ops::layer_norm;

pub struct LayerNorm {
    weight: Vector,
    bias: Vector,
}

impl LayerNorm {
    /// The LayerNorm of every row of `x` (rows of the norm's width).
    pub fn apply(&self, x: &[f32], eps: f32, out: &mut [f32]) {
        layer_norm(x, self.weight.values(), self.bias.values(), eps, out);
    }
}

/// `p.weight` and `p.bias`, f32 vectors of the norm's width.
impl Load for LayerNorm {
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, width: usize) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            weight: scope.load("weight", width)?,
            bias: scope.load("bias", width)?,
        })
    }
}
