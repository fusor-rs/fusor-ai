//! Block-quantized weights and the kernels of transformer inference on the CPU, for native
//! hosts and WebAssembly.
//!
//! Weights are f32, quantized per block of 32 (`Q7b32`, `Q4b32`) or ternary with a scale per 128
//! (`T2b128`, see `quant`); activations are quantized to int8 per scale group on the fly, and
//! `matmul` multiplies them with the target's dot-product instructions (`arch`). `Linear` adds
//! the bias; `project` and `project_residual` split a linear layer's output columns over the
//! threads of a `team` run, and `conv` splits products over many rows (images, long sequences)
//! by rows, 2-D convolutions among them.

mod arch;
mod blob;
pub mod conv;
mod linear;
mod matmul;
pub mod numpy;
pub mod ops;
mod quant;
pub mod rng;
mod tensor;

pub use arch::{DOTS, decay_rows, dot_f32, dot_f32_many};
pub use linear::{
    COLUMN_ALIGN, Linear, NARROW_COLUMN_ALIGN, SharedRows, project, project_input,
    project_residual, project_residual_input,
};
pub use matmul::{Input, Quantized, QuantizedMatrix, QuantizedRows, matmul, quantize_rows};
pub use quant::{BLOCK, QUAD, QuantFormat, TERNARY_GROUP};
pub use tensor::{DType, Matrix, MatrixStorage, Table, Tensor, Vector};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor whose type or shape does not fit its use.
    #[error("{0}")]
    Format(String),
}
