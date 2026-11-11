//! The browser runtimes: the `wan2_1` crate behind a wasm-bindgen surface.
//!
//! One module exports two classes, `TextRuntime` and `VideoRuntime`: each page worker
//! instantiates the module on its own memory and constructs the class of its bundle. Values
//! cross the boundary as strings (JSON), numbers and byte arrays. The module performs no I/O:
//! the loader pushes the bundle's parts in, the page pushes a request in and pulls frames out,
//! and the text encoder's states cross from one worker to the other as bytes through the page.

#[cfg(target_arch = "wasm32")]
mod wasm;
