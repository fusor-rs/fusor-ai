use kernels::{Linear, Tensor};

use crate::{Error, Store};

/// A part of a model that builds itself from a bundle's tensors, under names relative to
/// `prefix`. `Shape` is what the tensors alone don't say (dimensions, depth); `()` when nothing.
pub trait Load: Sized {
    type Shape;

    fn load(store: &mut Store, prefix: &str, shape: Self::Shape) -> Result<Self, Error>;
}

/// The tensor named `prefix` itself.
impl Load for Tensor {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        store.take(prefix)
    }
}

/// `prefix.weight` and, when the bundle has one, `prefix.bias`.
impl Load for Linear {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let weight = store.take(&format!("{prefix}.weight"))?;
        let bias = store.take_optional(&format!("{prefix}.bias"));
        Ok(Linear::new(weight, bias)?)
    }
}
