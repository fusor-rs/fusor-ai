use parakeet::session::Engine;
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// The rate of the PCM the runtime reads, in Hz.
#[wasm_bindgen]
pub fn sample_rate() -> u32 {
    parakeet::SAMPLE_RATE
}

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts transcribing a recording (16 kHz mono); returns its plan
    /// (`{"segments": [{"start","end"}], "seconds"}`).
    pub fn start(&mut self, pcm: Vec<f32>) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.start(pcm).map_err(error)?))
    }

    /// Transcribes the next segment; returns `{"segment","segments","sentences","done"}`, each
    /// sentence `{"text","start","end"}` in seconds.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// Starts listening to live audio afresh.
    pub fn listen(&mut self) -> Result<(), JsError> {
        self.shell.engine()?.listen();
        Ok(())
    }

    /// The next live samples (16 kHz mono).
    pub fn hear(&mut self, samples: &[f32]) -> Result<(), JsError> {
        self.shell.engine()?.hear(samples);
        Ok(())
    }

    /// Transcribes what was heard; returns `{"committed","provisional","heard"}`.
    pub fn update(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.update().map_err(error)?))
    }

    /// Makes everything heard final; returns `{"committed","provisional","heard"}`.
    pub fn finish(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.finish().map_err(error)?))
    }
}
