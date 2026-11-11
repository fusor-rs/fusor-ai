//! The browser runtimes: the `iris` crate behind a wasm-bindgen surface.
//!
//! One module exports two classes, `TextRuntime` and `ImageRuntime`: each page worker
//! instantiates the module on its own memory and constructs the class of its bundle. Values
//! cross the boundary as strings (JSON), numbers and byte arrays. The module performs no I/O:
//! the loader pushes the bundle's parts in, the page pushes a request (and a photo's RGBA bytes)
//! in and pulls the image out, and the text encoder's states cross from one worker to the other
//! as bytes through the page.

#[cfg(target_arch = "wasm32")]
mod wasm;
