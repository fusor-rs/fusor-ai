//! P-JiT 21K (Linum), a pixel-space text-to-image diffusion transformer conditioned on
//! Qwen3.5-4B, on the CPU, for native hosts and WebAssembly. Two engines: the text engine turns a
//! caption into the hidden states of three of Qwen3.5-4B's layers; the image engine takes the
//! caption's and the negative prompt's states, denoises from seeded noise with Euler steps and
//! adaptive projected guidance, and previews every step's clean image. `pipeline` lists the
//! stages in order; `session` runs them.

pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod session;

pub use bundle::Error;
pub use media::image::Size;
