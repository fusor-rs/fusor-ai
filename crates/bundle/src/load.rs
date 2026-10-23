use kernels::conv::{self, Conv2d, Depthwise2d, DepthwiseShape};
use kernels::{Linear, Table, Vector};

use crate::{Error, Store};

/// A part of a model that builds itself from a bundle's tensors, under names relative to
/// `prefix`. `Shape` is what the tensors must be (dimensions, depth), checked as they load.
pub trait Load: Sized {
    type Shape;

    fn load(store: &mut Store, prefix: &str, shape: Self::Shape) -> Result<Self, Error>;
}

/// The tensors of a store under one prefix: `scope.load("norm", width)` loads `prefix.norm`.
pub struct Scope<'store, 'prefix> {
    store: &'store mut Store,
    prefix: &'prefix str,
}

impl<'store, 'prefix> Scope<'store, 'prefix> {
    pub fn new(store: &'store mut Store, prefix: &'prefix str) -> Self {
        Self { store, prefix }
    }

    pub fn load<T: Load>(&mut self, part: &str, shape: T::Shape) -> Result<T, Error> {
        T::load(self.store, &format!("{}.{part}", self.prefix), shape)
    }
}

/// `count` parts, each loaded by `load` under `prefix.0`, `prefix.1` … (a stack's layers).
pub fn load_each<T>(
    store: &mut Store,
    prefix: &str,
    count: usize,
    mut load: impl FnMut(&mut Store, &str) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    (0..count)
        .map(|index| load(store, &format!("{prefix}.{index}")))
        .collect()
}

/// The f32 vector named `prefix`, of this length.
impl Load for Vector {
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, len: usize) -> Result<Self, Error> {
        let vector = Vector::new(store.take(prefix)?)?;
        let found = vector.values().len();
        if found != len {
            return Err(Error::Format(format!(
                "{prefix}: {found} values, not {len}"
            )));
        }
        Ok(vector)
    }
}

/// The `[rows, cols]` matrix named `prefix`.
impl Load for Table {
    type Shape = [usize; 2];

    fn load(store: &mut Store, prefix: &str, shape: [usize; 2]) -> Result<Self, Error> {
        let table = Table::new(store.take(prefix)?)?;
        let matrix = table.matrix();
        let found = [matrix.rows, matrix.cols];
        if found != shape {
            return Err(Error::Format(format!("{prefix}: {found:?}, not {shape:?}")));
        }
        Ok(table)
    }
}

/// `prefix.weight`, `[outputs, inputs]`, and, when the bundle has one, `prefix.bias`.
impl Load for Linear {
    type Shape = [usize; 2];

    fn load(store: &mut Store, prefix: &str, shape: [usize; 2]) -> Result<Self, Error> {
        let weight = Table::load(store, &format!("{prefix}.weight"), shape)?;
        let bias = store.take_optional(&format!("{prefix}.bias"));
        Ok(Linear::new(weight, bias.map(Vector::new).transpose()?)?)
    }
}

/// The convolution `prefix`: `prefix.weight`, `[outputs, kernel² · inputs]`, and its bias when
/// the bundle has one.
impl Load for Conv2d {
    type Shape = conv::Shape;

    fn load(store: &mut Store, prefix: &str, shape: conv::Shape) -> Result<Self, Error> {
        let linear = Linear::load(store, prefix, shape.matrix())?;
        Ok(Conv2d::new(linear, shape.kernel, shape.stride)?)
    }
}

/// The depthwise convolution `prefix`: `prefix.weight`, `[channels, height · width]`, each
/// channel's taps row-major, and its bias when the bundle has one.
impl Load for Depthwise2d {
    type Shape = DepthwiseShape;

    fn load(store: &mut Store, prefix: &str, shape: DepthwiseShape) -> Result<Self, Error> {
        let [height, width] = shape.window;
        let weight = Table::load(
            store,
            &format!("{prefix}.weight"),
            [shape.channels, height * width],
        )?;
        let bias = store
            .take_optional(&format!("{prefix}.bias"))
            .map(Vector::new)
            .transpose()?;
        Ok(Depthwise2d::new(
            &weight,
            bias.as_ref(),
            shape.window,
            shape.stride,
        )?)
    }
}
