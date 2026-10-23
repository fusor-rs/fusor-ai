use kronos::session::{Engine, Request};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts forecasting `request_json` (`{"csv", "lookback", "horizon", "end"?, "sampling"?,
    /// "samples"?, "seed"?}`); returns its plan (`{"context", "actual", "times", "samples",
    /// "slides"}`).
    pub fn start(&mut self, request_json: &str) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(kronos::Error::Request(invalid.to_string())))?;
        Ok(json(&self.shell.engine()?.start(&request).map_err(error)?))
    }

    /// Draws the next row of every sample; returns `{"stage", "step", "steps", "forecast"}`, the
    /// forecast (`{"times", "values"}`) after the last row, else `null`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }
}
