//! Block-quantized weights and the kernels of transformer inference on the CPU, for native
//! hosts and WebAssembly.
//!
//! Weights are f32, quantized per block of 32 (`Q7b32`, `Q4b32`) or ternary with a scale per 128
//! (`T2b128`, see `tensor`); activations are quantized to int8 per scale group on the fly, and
//! `matmul` multiplies them with the target's dot-product instructions (`arch`). `Linear` and
//! `project_residual` split the output columns over a `team` run.

mod arch;
mod blob;
mod linear;
mod matmul;
pub mod ops;
pub mod rng;
mod tensor;
pub mod wav;

pub use arch::{DOTS, dot_f32, dot_f32_many};
pub use linear::{COLUMN_ALIGN, Linear, NARROW_COLUMN_ALIGN, SharedRows, project_residual};
pub use matmul::{Input, QuantizedRows, matmul};
pub use tensor::{BLOCK, DType, Matrix, MatrixStorage, TERNARY_GROUP, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor whose type or shape does not fit its use.
    #[error("{0}")]
    Format(String),
}
