//! The inputs and outputs of models around the kernels: WAV files, mel spectrograms as audio
//! models' reference preprocessing computes them, and images resampled the way vision models'
//! reference preprocessing resamples them (Pillow, PyTorch, OpenCV).

pub mod image;
pub mod mel;
pub mod opencv;
pub mod wav;
