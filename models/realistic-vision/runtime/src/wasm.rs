use realistic_vision::session::{Engine, Request};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// Image pixels per latent pixel along each side: `preview` is this much smaller.
#[wasm_bindgen]
pub fn latent_factor() -> usize {
    realistic_vision::model::LATENT_FACTOR
}

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts a generation (`session::Request` as JSON); returns the plan
    /// (`{"steps","width","height","dropped"}`).
    pub fn start(&mut self, request_json: &str) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(realistic_vision::Error::Request(invalid.to_string())))?;
        Ok(json(&self.shell.engine()?.start(&request).map_err(error)?))
    }

    /// The next denoising step, or the decoding after the last; returns
    /// `{"step","steps","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// A preview of the clean image the last step predicts: RGBA bytes, rows top to bottom,
    /// `latent_factor()` times smaller along each side than the image.
    pub fn preview(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.preview().map_err(error)
    }

    /// The decoded image's RGBA bytes, rows top to bottom; ends the generation.
    pub fn take_image(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.take_image().map_err(error)
    }
}
