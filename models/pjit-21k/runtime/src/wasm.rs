use pjit_21k::session::{ImageEngine, Request, TextEngine};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

runtime::runtime!(TextRuntime, TextEngine);
runtime::runtime!(ImageRuntime, ImageEngine);

#[wasm_bindgen]
impl TextRuntime {
    /// The hidden states of `caption` after Qwen3.5-4B's layers 7, 15 and 27: rows of 7680
    /// values, one per token of the templated caption.
    pub fn encode(&mut self, caption: &str) -> Result<Vec<f32>, JsError> {
        self.shell.engine()?.encode(caption).map_err(error)
    }
}

#[wasm_bindgen]
impl ImageRuntime {
    /// Starts a generation (`session::Request` as JSON) from the text encoder's states of the
    /// caption and of the negative prompt; returns the plan (`{"steps","width","height"}`).
    pub fn start(
        &mut self,
        request_json: &str,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(pjit_21k::Error::Request(invalid.to_string())))?;
        let plan = self.shell.engine()?.start(&request, prompt, negative);
        Ok(json(&plan.map_err(error)?))
    }

    /// The next denoising step; returns `{"step","steps","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The clean image the last step predicts: RGBA bytes, rows top to bottom.
    pub fn preview(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.preview().map_err(error)
    }

    /// The image's RGBA bytes, rows top to bottom; ends the generation.
    pub fn take_image(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.take_image().map_err(error)
    }
}
