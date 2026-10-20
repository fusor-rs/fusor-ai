//! Block-quantized weights and the kernels of transformer inference on the CPU, for native
//! hosts and WebAssembly.
//!
//! Weights are f32 or quantized per block of 32 (`Q7b32`, `Q4b32`, see `tensor`); activations
//! are quantized to int8 per block on the fly, and `matmul` multiplies them with the target's
//! dot-product instructions (`arch`). `Linear` and `project_residual` split the output columns
//! over a `team` run.

mod arch;
mod blob;
mod linear;
mod matmul;
pub mod ops;
mod tensor;

pub use arch::dot_f32;
pub use linear::{COLUMN_ALIGN, Linear, SharedRows, project_residual};
pub use matmul::{Input, QuantizedRows, matmul};
pub use tensor::{BLOCK, DType, Matrix, MatrixStorage, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor whose type or shape does not fit its use.
    #[error("{0}")]
    Format(String),
}
