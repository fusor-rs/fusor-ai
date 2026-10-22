//! The browser runtime: the `parakeet` crate behind a wasm-bindgen surface.
//!
//! Values cross the boundary as strings (JSON), numbers, byte arrays and f32 arrays. The module
//! performs no I/O: the loader pushes the bundle's parts in, the page pushes 16 kHz PCM in and
//! pulls text out.

#[cfg(target_arch = "wasm32")]
mod wasm;
