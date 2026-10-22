//! The browser runtime: the `oxr` crate behind a wasm-bindgen surface.
//!
//! Values cross the boundary as strings (JSON), numbers and byte arrays. The module performs no
//! I/O: the loader pushes the bundle's parts in, the page pushes an image's RGBA pixels in and
//! pulls text out.

#[cfg(target_arch = "wasm32")]
mod wasm;
