//! The browser runtime: the `lfm3` crate behind a wasm-bindgen surface.
//!
//! Values cross the boundary as strings (JSON), numbers and byte arrays. The module performs no
//! I/O: the loader pushes the bundle's parts in.

#[cfg(target_arch = "wasm32")]
mod wasm;
