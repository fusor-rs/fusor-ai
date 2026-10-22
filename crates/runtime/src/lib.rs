//! The browser runtime shell every model shares: a model's [`bundle::Engine`] behind a
//! wasm-bindgen surface.
//!
//! Values cross the boundary as strings (JSON), numbers and byte arrays. The module performs no
//! I/O: the loader pushes the bundle's parts in, and once the last one is checked the engine is
//! built. wasm-bindgen exports no generic type, so a model's runtime crate names its engine once,
//! `runtime::runtime!(Engine);`, which defines the exported `Runtime` with the shell's methods,
//! and adds its own `#[wasm_bindgen] impl Runtime` (`start`, `step`, …) that reaches the engine
//! through `self.shell.engine()?`.

#![cfg(target_arch = "wasm32")]

mod entropy;

use std::sync::{Mutex, Once, PoisonError};

use bundle::{Engine, Manifest, Store};
use wasm_bindgen::prelude::*;

/// The path the [`runtime!`] expansion names wasm-bindgen by.
pub use wasm_bindgen;
/// Threaded builds: `initThreadPool(n)` (wasm-bindgen-rayon) starts `n` compute workers sharing
/// this module's memory; the loader calls it before creating the `Runtime`.
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

static LAST_PANIC: Mutex<String> = Mutex::new(String::new());
static PANIC_HOOK: Once = Once::new();

/// The message of the last panic, for the host to read after a `RuntimeError: unreachable`.
#[wasm_bindgen]
pub fn last_panic() -> String {
    LAST_PANIC
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// A failure as the host reads it: `{"kind", "message"}` (`bundle::Error::kind`).
pub fn error(error: impl Into<bundle::Error>) -> JsError {
    let error = error.into();
    JsError::new(
        &serde_json::json!({ "kind": error.kind(), "message": error.to_string() }).to_string(),
    )
}

pub fn json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("plain data serializes")
}

/// The store while the bundle's parts arrive, then the engine built from it.
pub struct Shell<E> {
    store: Option<Store>,
    engine: Option<E>,
}

impl<E: Engine> Shell<E> {
    pub fn new(manifest_json: &str) -> Result<Self, JsError> {
        PANIC_HOOK.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                *LAST_PANIC.lock().unwrap_or_else(PoisonError::into_inner) = info.to_string();
            }));
        });
        let manifest = Manifest::parse(manifest_json).map_err(error)?;
        Ok(Self {
            store: Some(Store::new(manifest).map_err(error)?),
            engine: None,
        })
    }

    fn store(&mut self) -> Result<&mut Store, JsError> {
        self.store
            .as_mut()
            .ok_or_else(|| JsError::new("the bundle was already loaded (or failed to build)"))
    }

    pub fn write_part(&mut self, part: usize, bytes: &[u8]) -> Result<(), JsError> {
        self.store()?.write(part, bytes).map_err(error)
    }

    pub fn restart_part(&mut self, part: usize) -> Result<(), JsError> {
        self.store()?.restart_part(part).map_err(error)
    }

    /// Checks the part's size and BLAKE3; after the last part, builds the engine. Returns
    /// whether the engine is ready.
    pub fn end_part(&mut self, part: usize) -> Result<bool, JsError> {
        let store = self.store()?;
        store.end_part(part).map_err(error)?;
        if store.complete() {
            let mut store = self.store.take().expect("checked above");
            self.engine = Some(E::new(&mut store).map_err(error)?);
        }
        Ok(self.engine.is_some())
    }

    pub fn engine(&mut self) -> Result<&mut E, JsError> {
        self.engine
            .as_mut()
            .ok_or_else(|| JsError::new("the model is not loaded yet"))
    }

    pub fn threads(&mut self) -> Result<usize, JsError> {
        Ok(self.engine()?.threads())
    }

    pub fn cancel(&mut self) -> Result<(), JsError> {
        self.engine()?.cancel();
        Ok(())
    }
}

/// Defines the exported `Runtime` over a [`Shell`] of `$engine`, with the methods every model's
/// runtime has: the constructor (the manifest's JSON), the part protocol, `threads` and `cancel`.
#[macro_export]
macro_rules! runtime {
    ($engine:ty) => {
        #[$crate::wasm_bindgen::prelude::wasm_bindgen(wasm_bindgen = $crate::wasm_bindgen)]
        pub struct Runtime {
            shell: $crate::Shell<$engine>,
        }

        #[$crate::wasm_bindgen::prelude::wasm_bindgen(wasm_bindgen = $crate::wasm_bindgen)]
        impl Runtime {
            #[wasm_bindgen(constructor)]
            pub fn new(manifest_json: &str) -> Result<Runtime, $crate::wasm_bindgen::JsError> {
                Ok(Self {
                    shell: $crate::Shell::new(manifest_json)?,
                })
            }

            /// The next bytes of part `part` (index into the manifest's `parts`), in order.
            pub fn write_part(
                &mut self,
                part: usize,
                bytes: &[u8],
            ) -> Result<(), $crate::wasm_bindgen::JsError> {
                self.shell.write_part(part, bytes)
            }

            /// Discards what was written to part `part` (after a failed `end_part`), to write it
            /// again.
            pub fn restart_part(
                &mut self,
                part: usize,
            ) -> Result<(), $crate::wasm_bindgen::JsError> {
                self.shell.restart_part(part)
            }

            /// Checks the part's size and BLAKE3; after the last part, builds the engine. Returns
            /// whether the engine is ready.
            pub fn end_part(&mut self, part: usize) -> Result<bool, $crate::wasm_bindgen::JsError> {
                self.shell.end_part(part)
            }

            pub fn threads(&mut self) -> Result<usize, $crate::wasm_bindgen::JsError> {
                self.shell.threads()
            }

            pub fn cancel(&mut self) -> Result<(), $crate::wasm_bindgen::JsError> {
                self.shell.cancel()
            }
        }
    };
}
