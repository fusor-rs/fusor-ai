//! A linear layer (`y = x·Wᵀ + b`) and the team-split projection built on it.

use team::{Ctx, SharedBuf};

use crate::Error;
use crate::matmul::{QuantizedRows, matmul};
use crate::tensor::{Matrix, Tensor};

/// Column granularity of the output splits (a multiple of the SIMD width).
pub const COLUMN_ALIGN: usize = 16;
/// The granularity for projections with few outputs (a 64-d latent), so every thread gets some.
pub const NARROW_COLUMN_ALIGN: usize = 8;

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

/// The first `rows` rows of a shared activation buffer.
#[derive(Clone, Copy)]
pub struct SharedRows<'buf> {
    pub values: &'buf SharedBuf,
    pub rows: usize,
}

impl Linear {
    /// Checks the shapes once, so `matrix` and `bias` cannot fail in the forward passes.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self, Error> {
        let outputs = weight.matrix()?.rows;
        if let Some(bias) = &bias {
            let len = bias.values()?.len();
            if len != outputs {
                return Err(Error::Format(format!(
                    "bias of {len} for {outputs} outputs"
                )));
            }
        }
        Ok(Self { weight, bias })
    }

    pub fn matrix(&self) -> Matrix<'_> {
        self.weight.matrix().expect("checked in Linear::new")
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias
            .as_ref()
            .map(|bias| bias.values().expect("checked in Linear::new"))
    }

    /// `out[row] = x[row]·Wᵀ + b` for all outputs on the calling thread (small layers, setup).
    pub fn apply(&self, x: &[f32], rows: usize, quant: &mut QuantizedRows, out: &mut [f32]) {
        let matrix = self.matrix();
        let outputs = matrix.rows;
        let input = quant.input(&matrix, x, rows);
        let bias = self.bias();
        matmul(&matrix, &input, rows, 0..outputs, |row, col, value| {
            out[row * outputs + col] = value + bias.map_or(0.0, |bias| bias[col]);
        });
    }
}

/// `residual[:, cols] += input · Wᵀ + b` for this thread's share of the output columns.
pub fn project_residual(
    ctx: &Ctx,
    linear: &Linear,
    input: SharedRows,
    residual: &SharedBuf,
    quant: &mut QuantizedRows,
) {
    let matrix = linear.matrix();
    let outputs = matrix.rows;
    let rows = input.rows;
    let x = quant.input(&matrix, &input.values.read()[..rows * matrix.cols], rows);
    let bias = linear.bias();
    let cols = ctx.split(outputs, COLUMN_ALIGN);
    matmul(&matrix, &x, rows, cols, |row, col, value| {
        let at = row * outputs + col;
        let biased = value + bias.map_or(0.0, |bias| bias[col]);
        residual.set(at, residual.get(at) + biased);
    });
}
