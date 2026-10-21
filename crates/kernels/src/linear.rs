//! A linear layer (`y = x·Wᵀ + b`) and the team-split projections built on it.

use std::ops::Range;

use team::{Ctx, SharedBuf};

use crate::Error;
use crate::matmul::{Input, QuantizedRows, matmul};
use crate::tensor::{Matrix, Table, Vector};

/// Column granularity of the output splits (a multiple of the SIMD width).
pub const COLUMN_ALIGN: usize = 16;
/// The granularity for projections with few outputs, so every thread gets some.
pub const NARROW_COLUMN_ALIGN: usize = 8;

pub struct Linear {
    weight: Table,
    bias: Option<Vector>,
}

/// The first `rows` rows of a shared activation buffer.
#[derive(Clone, Copy)]
pub struct SharedRows<'buf> {
    pub values: &'buf SharedBuf,
    pub rows: usize,
}

impl Linear {
    /// # Errors
    /// When the bias does not have one value per output.
    pub fn new(weight: Table, bias: Option<Vector>) -> Result<Self, Error> {
        let outputs = weight.matrix().rows;
        if let Some(bias) = &bias
            && bias.values().len() != outputs
        {
            return Err(Error::Format(format!(
                "bias of {} for {outputs} outputs",
                bias.values().len()
            )));
        }
        Ok(Self { weight, bias })
    }

    pub fn matrix(&self) -> Matrix<'_> {
        self.weight.matrix()
    }

    /// `store(row, col, input[row] · W[col] + b[col])` for every row of `input` and every output
    /// column in `cols`.
    pub fn project(
        &self,
        input: &Input,
        cols: Range<usize>,
        mut store: impl FnMut(usize, usize, f32),
    ) {
        let bias = self.bias.as_ref().map(Vector::values);
        matmul(&self.matrix(), input, cols, |row, col, value| {
            store(row, col, value + bias.map_or(0.0, |bias| bias[col]));
        });
    }

    /// `out[row] = x[row]·Wᵀ + b` for every row of `x` and every output, on the calling thread
    /// (small layers, setup).
    pub fn apply(&self, x: &[f32], quant: &mut QuantizedRows, out: &mut [f32]) {
        let outputs = self.matrix().rows;
        let input = quant.input(&self.matrix(), x);
        self.project(&input, 0..outputs, |row, col, value| {
            out[row * outputs + col] = value;
        });
    }
}

/// `store(row · outputs + col, input · Wᵀ + b)` for this thread's share of the output columns.
pub fn project(
    ctx: &Ctx,
    linear: &Linear,
    input: SharedRows,
    quant: &mut QuantizedRows,
    mut store: impl FnMut(usize, f32),
) {
    let matrix = linear.matrix();
    let outputs = matrix.rows;
    let x = &input.values.read()[..input.rows * matrix.cols];
    let x = quant.input(&matrix, x);
    let cols = ctx.split(outputs, COLUMN_ALIGN);
    linear.project(&x, cols, |row, col, value| {
        store(row * outputs + col, value)
    });
}

/// `residual[:, cols] += input · Wᵀ + b` for this thread's share of the output columns.
pub fn project_residual(
    ctx: &Ctx,
    linear: &Linear,
    input: SharedRows,
    residual: &SharedBuf,
    quant: &mut QuantizedRows,
) {
    project(ctx, linear, input, quant, |at, value| {
        residual.set(at, residual.get(at) + value);
    });
}
