use motif_video::image::first_frame as resize_first_frame;
use motif_video::session::{FirstImage, Request, TextEngine, VideoEngine};
use motif_video::{Pixels, Size};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// Bytes of a pixel the page decodes a first image to.
const RGBA_CHANNELS: usize = 4;

/// Frames per second of the clip.
#[wasm_bindgen]
pub fn fps() -> usize {
    motif_video::model::FPS
}

/// The first image (`rgba`, `image_width` pixels wide) resized to the video's size as the
/// pipeline resizes it: pixel-major RGB in [−1, 1].
#[wasm_bindgen]
pub fn first_frame(
    rgba: &[u8],
    image_width: usize,
    video_width: usize,
    video_height: usize,
) -> Result<Vec<f32>, JsError> {
    let row_bytes = match image_width.checked_mul(RGBA_CHANNELS) {
        Some(row_bytes) if row_bytes > 0 && rgba.len().is_multiple_of(row_bytes) => row_bytes,
        _ => {
            let invalid = format!(
                "{} bytes are not rows of {image_width} RGBA pixels",
                rgba.len()
            );
            return Err(error(motif_video::Error::Request(invalid)));
        }
    };
    let image = Pixels {
        size: Size {
            width: image_width,
            height: rgba.len() / row_bytes,
        },
        channels: RGBA_CHANNELS,
        values: rgba.to_vec(),
    };
    let video = Size {
        width: video_width,
        height: video_height,
    };
    resize_first_frame(&image, video).map_err(error)
}

runtime::runtime!(TextRuntime, TextEngine);
runtime::runtime!(VideoRuntime, VideoEngine);

#[wasm_bindgen]
impl TextRuntime {
    /// T5Gemma2's last hidden states of `text`: rows of 2560 values, `<bos>`'s first.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        self.shell.engine()?.encode(text).map_err(error)
    }

    /// SigLIP's states of a first frame (`first_frame`'s): 4096 rows of 1152 values.
    pub fn embed_frame(
        &mut self,
        frame: &[f32],
        video_width: usize,
        video_height: usize,
    ) -> Result<Vec<f32>, JsError> {
        let size = Size {
            width: video_width,
            height: video_height,
        };
        self.shell.engine()?.embed_frame(frame, size).map_err(error)
    }
}

#[wasm_bindgen]
impl VideoRuntime {
    /// The first image of the next `start` only: SigLIP's states and the frame they came from.
    pub fn first_image(&mut self, states: Vec<f32>, frame: Vec<f32>) -> Result<(), JsError> {
        self.shell
            .engine()?
            .first_image(FirstImage { states, frame });
        Ok(())
    }

    /// Starts a generation (`session::Request` as JSON) from the text encoder's states of the
    /// prompt and of the negative prompt; returns the plan
    /// (`{"steps","width","height","frames","image"}`).
    pub fn start(
        &mut self,
        request_json: &str,
        prompt: &[f32],
        negative: &[f32],
    ) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(motif_video::Error::Request(invalid.to_string())))?;
        let plan = self.shell.engine()?.start(&request, prompt, negative);
        Ok(json(&plan.map_err(error)?))
    }

    /// The next denoising step, or the next strip of rows decoded; returns
    /// `{"step","steps","decoded","strips","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The first frame of the clean video the last step predicts: RGBA bytes, rows top to bottom.
    pub fn preview(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.preview().map_err(error)
    }

    /// The RGBA bytes of every frame, one after another, once the last strip is decoded; empty
    /// before.
    pub fn take_frames(&mut self) -> Result<Vec<u8>, JsError> {
        self.shell.engine()?.take_frames().map_err(error)
    }
}
