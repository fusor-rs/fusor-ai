//! The browser runtimes: the `motif_video` crate behind a wasm-bindgen surface.
//!
//! One module exports two classes, `TextRuntime` and `VideoRuntime`: each page worker
//! instantiates the module on its own memory and constructs the class of its bundle. Values
//! cross the boundary as strings (JSON), numbers and byte arrays. The module performs no I/O:
//! the loader pushes the bundle's parts in, the page pushes a request in and pulls frames out,
//! and the text engine's states (of the prompts and of a first image) cross from one worker to
//! the other as bytes through the page. A free function prepares a first image.

#[cfg(target_arch = "wasm32")]
mod wasm;
