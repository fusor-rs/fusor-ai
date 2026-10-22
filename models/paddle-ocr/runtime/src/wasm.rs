use paddle_ocr::session::{Engine, Request};
use paddle_ocr::{Pixels, Size};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts reading an image of `width × height` RGBA pixels under `request_json`
    /// (`{"task": "ocr" | "table" | "formula" | "chart" | "seal", "max_tokens"?}`); returns its
    /// plan (`{"layout", "image_tokens", "prompt_tokens"}`).
    pub fn start(
        &mut self,
        rgba: Vec<u8>,
        width: usize,
        height: usize,
        request_json: &str,
    ) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(paddle_ocr::Error::Request(invalid.to_string())))?;
        if rgba.len() != width * height * RGBA {
            return Err(error(paddle_ocr::Error::Request(format!(
                "{} bytes for a {width}×{height} RGBA image",
                rgba.len()
            ))));
        }
        let image = Pixels {
            size: Size { width, height },
            channels: RGBA,
            values: rgba,
        };
        Ok(json(
            &self
                .shell
                .engine()?
                .start(&image, &request)
                .map_err(error)?,
        ))
    }

    /// Runs the next stage; returns `{"stage", "text", "tokens", "done"}`, the text being the
    /// whole answer so far.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }
}
