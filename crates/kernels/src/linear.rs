//! A linear layer (`y = x·Wᵀ + b`) and the team-split projections built on it.

use std::ops::Range;

use team::{Ctx, SharedBuf};

use crate::Error;
use crate::matmul::{Input, QuantizedRows, matmul};
use crate::quant::QUAD;
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

    /// `store(row, first, input[row] · W[first + lane] + b[first + lane])` for every row of
    /// `input` and every quad of output columns from `first` in `cols`, a range of whole quads
    /// within the outputs.
    pub fn project_quads(
        &self,
        input: &Input,
        cols: Range<usize>,
        mut store: impl FnMut(usize, usize, [f32; QUAD]),
    ) {
        assert!(
            cols.start.is_multiple_of(QUAD) && cols.end.is_multiple_of(QUAD),
            "whole quads of output columns"
        );
        let bias = self.bias.as_ref().map(Vector::values);
        matmul(&self.matrix(), input, cols, |row, first, mut values| {
            if let Some(bias) = bias {
                for (value, bias) in values.iter_mut().zip(&bias[first..]) {
                    *value += bias;
                }
            }
            store(row, first, values);
        });
    }

    /// `store(row, col, input[row] · W[col] + b[col])` for every row of `input` and every output
    /// column in `cols` (any range within the outputs; the matmul runs over whole quads).
    pub fn project(
        &self,
        input: &Input,
        cols: Range<usize>,
        mut store: impl FnMut(usize, usize, f32),
    ) {
        let quads = cols.start / QUAD * QUAD..cols.end.next_multiple_of(QUAD);
        self.project_quads(input, quads, |row, first, values| {
            for (col, value) in (first..).zip(values).filter(|(col, _)| cols.contains(col)) {
                store(row, col, value);
            }
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

/// `store(row · outputs + col, input · Wᵀ + b)` for this thread's share of the output columns,
/// the rows quantized by this thread when the weights are.
pub fn project(
    ctx: &Ctx,
    linear: &Linear,
    input: SharedRows,
    quant: &mut QuantizedRows,
    store: impl FnMut(usize, f32),
) {
    project_input(ctx, linear, &prepare(linear, input, quant), store);
}

/// `project` of rows already prepared as `linear`'s input.
pub fn project_input(ctx: &Ctx, linear: &Linear, input: &Input, mut store: impl FnMut(usize, f32)) {
    let outputs = linear.matrix().rows;
    let cols = ctx.split(outputs, COLUMN_ALIGN);
    linear.project(input, cols, |row, col, value| {
        store(row * outputs + col, value)
    });
}

/// `residual[:, cols] += input · Wᵀ + b` for this thread's share of the output columns, the
/// rows quantized by this thread when the weights are.
pub fn project_residual(
    ctx: &Ctx,
    linear: &Linear,
    input: SharedRows,
    residual: &SharedBuf,
    quant: &mut QuantizedRows,
) {
    project_residual_input(ctx, linear, &prepare(linear, input, quant), residual);
}

/// `project_residual` of rows already prepared as `linear`'s input.
pub fn project_residual_input(ctx: &Ctx, linear: &Linear, input: &Input, residual: &SharedBuf) {
    let outputs = linear.matrix().rows;
    let cols = ctx.split(outputs, COLUMN_ALIGN);
    linear.project_quads(input, cols, |row, first, values| {
        let at = row * outputs + first;
        for (slot, value) in residual.part(at..at + QUAD).iter_mut().zip(values) {
            *slot += value;
        }
    });
}

fn prepare<'rows>(
    linear: &Linear,
    input: SharedRows<'rows>,
    quant: &'rows mut QuantizedRows,
) -> Input<'rows> {
    let matrix = linear.matrix();
    quant.input(&matrix, &input.values.read()[..input.rows * matrix.cols])
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::matmul::QuantizedRows;
    use crate::tensor::{DType, Table, Tensor, Vector};

    #[test]
    fn project_quads_adds_the_bias_to_each_lane() {
        let (outputs, inputs) = (8, 32);
        let weights: Vec<f32> = (1..=outputs).flat_map(|row| [row as f32; 32]).collect();
        let weight = Tensor::encode(DType::F32, vec![outputs, inputs], &weights).unwrap();
        let bias = Tensor::encode(DType::F32, vec![outputs], &[0.5; 8]).unwrap();
        let bias = Vector::new(bias).unwrap();
        let linear = Linear::new(Table::new(weight).unwrap(), Some(bias)).unwrap();
        let mut quant = QuantizedRows::default();
        let input = quant.input(&linear.matrix(), &[1.0; 32]);
        let mut got = Vec::new();
        linear.project_quads(&input, 0..outputs, |row, first, values| {
            assert_eq!((row, first), (0, got.len()));
            got.extend(values);
        });
        assert_eq!(got, [32.5, 64.5, 96.5, 128.5, 160.5, 192.5, 224.5, 256.5]);
    }
}
