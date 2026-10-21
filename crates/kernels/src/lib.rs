//! Block-quantized weights and the kernels of transformer inference on the CPU, for native
//! hosts and WebAssembly.
//!
//! Weights are f32, quantized per block of 32 (`Q7b32`, `Q4b32`) or ternary with a scale per 128
//! (`T2b128`, see `quant`); activations are quantized to int8 per scale group on the fly, and
//! `matmul` multiplies them with the target's dot-product instructions (`arch`). `Linear` adds
//! the bias; `project` and `project_residual` split a linear layer's output columns over the
//! threads of a `team` run.

mod arch;
mod blob;
mod linear;
mod matmul;
pub mod ops;
mod quant;
pub mod rng;
mod tensor;
pub mod wav;

pub use arch::{DOTS, dot_f32, dot_f32_many};
pub use linear::{
    COLUMN_ALIGN, Linear, NARROW_COLUMN_ALIGN, SharedRows, project, project_residual,
};
pub use matmul::{Input, QuantizedRows, matmul};
pub use quant::{BLOCK, QuantFormat, TERNARY_GROUP};
pub use tensor::{DType, Matrix, MatrixStorage, Table, Tensor, Vector};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor whose type or shape does not fit its use.
    #[error("{0}")]
    Format(String),
}
