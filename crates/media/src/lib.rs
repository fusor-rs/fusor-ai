//! The inputs and outputs of models around the kernels: WAV files, and images resampled the way
//! vision models' reference preprocessing resamples them (Pillow, PyTorch, OpenCV).

pub mod image;
pub mod opencv;
pub mod wav;
