use runtime::{error, json};
use wan2_1::session::{Request, TextEngine, VideoEngine};
use wasm_bindgen::prelude::*;

/// Frames per second of the clip.
#[wasm_bindgen]
pub fn fps() -> usize {
    wan2_1::model::FPS
}

runtime::runtime!(TextRuntime, TextEngine);
runtime::runtime!(VideoRuntime, VideoEngine);

#[wasm_bindgen]
impl TextRuntime {
    /// umT5-XXL's last hidden states of `text`: rows of 4096 values, one per token.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        self.shell.engine()?.encode(text).map_err(error)
    }
}

#[wasm_bindgen]
impl VideoRuntime {
    /// Starts a generation (`session::Request` as JSON) from the text encoder's states of the
    /// prompt and of the negative prompt; returns the plan (`{"steps","width","height","frames"}`).
    pub fn start(
        &mut self,
        request_json: &str,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(wan2_1::Error::Request(invalid.to_string())))?;
        let plan = self.shell.engine()?.start(&request, prompt, negative);
        Ok(json(&plan.map_err(error)?))
    }

    /// The next denoising step, or the next latent frame decoded; returns
    /// `{"step","steps","decoded","frames","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The first frame of the clean video the last step predicts: RGBA bytes, rows top to bottom.
    pub fn preview(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.preview().map_err(error)
    }

    /// The RGBA bytes of the frames decoded since the last call, one after another.
    pub fn take_frames(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.take_frames().map_err(error)
    }
}
