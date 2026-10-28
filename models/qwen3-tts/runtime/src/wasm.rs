use qwen3_tts::model::{Language, MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS, SAMPLE_RATE};
use qwen3_tts::session::{Engine, Request};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// The rate of the voice clip `start` takes and the speech `speech` returns, in Hz.
#[wasm_bindgen]
pub fn sample_rate() -> u32 {
    SAMPLE_RATE
}

/// `samples` as a 16-bit PCM WAV file at `sample_rate()`.
#[wasm_bindgen]
pub fn wav(samples: &[f32]) -> Vec<u8> {
    qwen3_tts::wav::encode(samples, SAMPLE_RATE)
}

/// What a request may ask for, as JSON: `{"sample_rate", "min_reference_seconds",
/// "max_reference_seconds", "languages"}`.
#[wasm_bindgen]
pub fn limits() -> String {
    let languages = Language::ALL.map(Language::name);
    serde_json::json!({
        "sample_rate": SAMPLE_RATE,
        "min_reference_seconds": MIN_REFERENCE_SECONDS,
        "max_reference_seconds": MAX_REFERENCE_SECONDS,
        "languages": languages,
    })
    .to_string()
}

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts speaking in the voice of `reference`, 24 kHz mono samples: `request_json` is
    /// `{"text", "language", "reference_text"?, "seed", "max_frames"?}`, with a transcript the
    /// voice is cloned from the clip's frames and the transcript, without from its speaker
    /// embedding alone. Returns the plan (`{"mode", "reference_frames", "prompt_rows",
    /// "max_frames"}`).
    pub fn start(&mut self, request_json: &str, reference: &[f32]) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(qwen3_tts::Error::Request(invalid.to_string())))?;
        let engine = self.shell.engine()?;
        Ok(json(&engine.start(&request, reference).map_err(error)?))
    }

    /// Runs the next stage; returns `{"stage", "frames", "done", "similarity"}`, the voice match
    /// (cosine of speaker embeddings) with the last step.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The 24 kHz mono samples spoken since the last call.
    pub fn speech(&mut self) -> Result<Vec<f32>, JsError> {
        Ok(self.shell.engine()?.speech())
    }
}
