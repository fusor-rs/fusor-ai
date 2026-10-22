use phonon::prompt::VOICES;
use phonon::session::{Engine, Request};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// `[{"id","label"}]` of the built-in voices.
#[wasm_bindgen]
pub fn voices() -> String {
    let list: Vec<_> = VOICES
        .iter()
        .map(|(id, label)| serde_json::json!({ "id": id, "label": label }))
        .collect();
    serde_json::Value::from(list).to_string()
}

/// The rate of the PCM `take_audio` returns, in Hz.
#[wasm_bindgen]
pub fn sample_rate() -> u32 {
    phonon::SAMPLE_RATE
}

/// `samples` as a 16-bit PCM WAV file at `sample_rate()`.
#[wasm_bindgen]
pub fn wav(samples: &[f32]) -> Vec<u8> {
    phonon::wav::encode(samples, phonon::SAMPLE_RATE)
}

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts a synthesis (`session::Request` as JSON); returns the plan (`{"fragments"}`).
    pub fn start(&mut self, request_json: &str) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(phonon::Error::Request(invalid.to_string())))?;
        Ok(json(&self.shell.engine()?.start(request).map_err(error)?))
    }

    /// One frame; returns `{"fragment","fragments","frame","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The samples (mono, `sample_rate()` Hz) produced since the last call.
    pub fn take_audio(&mut self) -> Result<Vec<f32>, JsError> {
        Ok(self.shell.engine()?.take_audio())
    }
}
