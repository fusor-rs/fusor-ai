//! The browser runtime: the `realistic_vision` crate behind a wasm-bindgen surface.
//!
//! Values cross the boundary as strings (JSON), numbers and byte arrays. The module performs no
//! I/O: the loader pushes the bundle's parts in, the page pushes a request in and pulls pixels
//! out.

#[cfg(target_arch = "wasm32")]
mod wasm;
