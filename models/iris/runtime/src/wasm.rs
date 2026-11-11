use iris::session::{DepthRequest, Generation, ImageEngine, TextEngine, Upscale, photo};
use runtime::{error, json};
use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;

runtime::runtime!(TextRuntime, TextEngine);
runtime::runtime!(ImageRuntime, ImageEngine);

/// A request from its JSON.
fn parse<T: DeserializeOwned>(request_json: &str) -> Result<T, JsError> {
    serde_json::from_str(request_json)
        .map_err(|invalid| error(iris::Error::Request(invalid.to_string())))
}

#[wasm_bindgen]
impl TextRuntime {
    /// Qwen3-VL's states of `text` in Iris's template: for each row from the caption on, the
    /// residual stream after each of 12 layers, 12 × 2560 values.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        self.shell.engine()?.encode(text).map_err(error)
    }
}

#[wasm_bindgen]
impl ImageRuntime {
    /// The transformer the bundle holds: "base", "turbo", "depth" or "upscaler".
    pub fn variant(&mut self) -> Result<String, JsError> {
        Ok(self.shell.engine()?.variant().name().into())
    }

    /// Starts an image (`session::Generation` as JSON) from the text encoder's states of the
    /// prompt and of the negative prompt (read only by a guided base run); returns the plan
    /// (`{"steps","width","height"}`).
    pub fn start_generation(
        &mut self,
        request_json: &str,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<String, JsError> {
        let request: Generation = parse(request_json)?;
        let engine = self.shell.engine()?;
        let plan = engine.start_generation(&request, prompt, negative);
        Ok(json(&plan.map_err(error)?))
    }

    /// Starts the depth map (`session::DepthRequest` as JSON) of a photo of `width` pixels wide
    /// (`rgba`); returns the plan.
    pub fn start_depth(
        &mut self,
        request_json: &str,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> Result<String, JsError> {
        let request: DepthRequest = parse(request_json)?;
        let photo = photo(rgba, width, height).map_err(error)?;
        let plan = self.shell.engine()?.start_depth(&request, &photo);
        Ok(json(&plan.map_err(error)?))
    }

    /// Starts the four times enlarged restoration (`session::Upscale` as JSON) of a photo; returns
    /// the plan, whose steps are tiles.
    pub fn start_upscale(
        &mut self,
        request_json: &str,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> Result<String, JsError> {
        let request: Upscale = parse(request_json)?;
        let photo = photo(rgba, width, height).map_err(error)?;
        let plan = self.shell.engine()?.start_upscale(&request, &photo);
        Ok(json(&plan.map_err(error)?))
    }

    /// The next step (a denoising step, the depth pass or a tile); returns
    /// `{"step","steps","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The clean image the last denoising step predicts: RGBA bytes, rows top to bottom.
    pub fn preview(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.preview().map_err(error)
    }

    /// The finished job's RGBA bytes, rows top to bottom (the plan's size).
    pub fn take_image(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.take_image().map_err(error)
    }
}
