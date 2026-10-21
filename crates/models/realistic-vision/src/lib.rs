//! Realistic Vision V6.0 B1 (SG161222), a Stable Diffusion 1.5 fine-tune for photorealism, on the
//! CPU, for native hosts and WebAssembly, decoding with Stability AI's SD VAE ft-MSE; built on
//! fusor-ai's kernels, thread team, blocks and bundle format.
//!
//! A prompt and a negative prompt become 77 CLIP tokens each (`prompt`), which CLIP's text
//! encoder turns into the UNet's context. From seeded Gaussian latents, DPM-Solver++(2M) at
//! Karras noise levels asks the UNet for the noise of both prompts at every step and guides
//! between them; the autoencoder's decoder turns the last latents into the image. `pipeline`
//! lists the stages in order; `session` runs them a step at a time.

pub mod model;
mod picture;
pub mod pipeline;
pub mod prompt;
pub mod session;
mod text;

pub use bundle::Error;
pub use kernels::conv::Grid;
