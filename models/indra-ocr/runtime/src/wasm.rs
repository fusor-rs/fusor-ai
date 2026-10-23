use indra_ocr::session::{Engine, Options};
use indra_ocr::{Pixels, Size};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts reading the page whose RGBA pixels `rgba` holds, `width × height`, with
    /// `options_json` (`{"math": bool}`: read lines with math as LaTeX).
    pub fn start(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        options_json: &str,
    ) -> Result<(), JsError> {
        let options: Options = serde_json::from_str(options_json)
            .map_err(|invalid| error(indra_ocr::Error::Request(invalid.to_string())))?;
        let size = Size {
            width: width as usize,
            height: height as usize,
        };
        if rgba.len() != size.width * size.height * RGBA {
            return Err(error(indra_ocr::Error::Request(format!(
                "{} bytes for a {width}×{height} RGBA image",
                rgba.len()
            ))));
        }
        let page = Pixels {
            size,
            channels: RGBA,
            values: rgba.to_vec(),
        };
        self.shell.engine()?.start(&page, options);
        Ok(())
    }

    /// Runs the next stage, or the next batch of lines; returns `{"stage", "batches",
    /// "planned", "page"}`, the page (as the reference's JSON writes it) once read, else
    /// `null`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }
}
