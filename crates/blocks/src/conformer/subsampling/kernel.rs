//! A stage's convolution over its input plane: a `kernel`-wide square window at `stride`, with
//! zero padding of half the window.

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::Vector;

use super::Shape;

/// A stage's convolution: taps and bias.
pub(super) struct Kernel {
    taps: Vector,
    bias: Vector,
}

/// Rows of a stage's input: `values` holds rows `first..` of `width × depth` values each; rows
/// outside it are zeros. Depth 1 is the features (every channel reads the one value).
pub(super) struct Plane<'values> {
    pub(super) values: &'values [f32],
    pub(super) first: usize,
    pub(super) width: usize,
    pub(super) depth: usize,
}

impl Plane<'_> {
    /// The values at `(row, column)`, or `None` for the zeros around the plane (`None`
    /// coordinates are the padding before it).
    fn at(&self, row: Option<usize>, column: Option<usize>) -> Option<&[f32]> {
        let (row, column) = (row?, column?);
        let rows = self.values.len() / (self.width * self.depth);
        if !(self.first..self.first + rows).contains(&row) || column >= self.width {
            return None;
        }
        let start = ((row - self.first) * self.width + column) * self.depth;
        Some(&self.values[start..start + self.depth])
    }
}

impl Kernel {
    /// Output rows `rows` of this convolution of `input` into `out`, `rows × outputs(width) ×
    /// channels`; rows from `valid` on are zeros.
    pub(super) fn apply(
        &self,
        shape: &Shape,
        input: &Plane,
        (rows, valid): (Range<usize>, usize),
        out: &mut Vec<f32>,
    ) {
        let channels = self.bias.values().len();
        let width = shape.stage_outputs(input.width);
        out.clear();
        out.resize(rows.len() * width * channels, 0.0);
        let outputs = out.chunks_exact_mut(width * channels);
        for (row, values) in rows.take_while(|&row| row < valid).zip(outputs) {
            for (column, sums) in values.chunks_exact_mut(channels).enumerate() {
                self.sum_at(shape, input, (row, column), sums);
            }
        }
    }

    /// The bias plus every tap's product at output `(row, column)`.
    fn sum_at(
        &self,
        shape: &Shape,
        input: &Plane,
        (row, column): (usize, usize),
        sums: &mut [f32],
    ) {
        let taps = self.taps.values();
        let (side, stride, padding) = (shape.kernel, shape.stride, shape.kernel / 2);
        let channels = sums.len();
        sums.copy_from_slice(self.bias.values());
        for tap in 0..side * side {
            let source = (stride * row + tap / side).checked_sub(padding);
            let at = (stride * column + tap % side).checked_sub(padding);
            if let Some(x) = input.at(source, at) {
                accumulate(sums, &taps[tap * channels..(tap + 1) * channels], x);
            }
        }
    }
}

/// `sums += weights · x`, `x` one value for every channel or one per channel.
fn accumulate(sums: &mut [f32], weights: &[f32], x: &[f32]) {
    if let [value] = x {
        for (sum, weight) in sums.iter_mut().zip(weights) {
            *sum += weight * value;
        }
        return;
    }
    for ((sum, weight), value) in sums.iter_mut().zip(weights).zip(x) {
        *sum += weight * value;
    }
}

impl Load for Kernel {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let taps = shape.kernel * shape.kernel;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            taps: scope.load("weight", taps * shape.channels)?,
            bias: scope.load("bias", shape.channels)?,
        })
    }
}
