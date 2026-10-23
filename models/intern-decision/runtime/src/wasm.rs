use intern_decision::request::Request;
use intern_decision::session::Engine;
use intern_decision::{Pixels, Size};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts deciding `request_json` (`{"state", "questions"}`, as the reference's
    /// `DecisionEngine.predict` takes it) about the images whose RGBA pixels `rgba` holds one
    /// after another, image `i` `sizes[2i]` wide and `sizes[2i + 1]` high; returns its plan
    /// (`{"images", "image_tokens", "prompt_tokens", "questions"}`).
    pub fn start(
        &mut self,
        request_json: &str,
        rgba: &[u8],
        sizes: &[u32],
    ) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|invalid| error(intern_decision::Error::Request(invalid.to_string())))?;
        let images = images(rgba, sizes).map_err(error)?;
        Ok(json(
            &self
                .shell
                .engine()?
                .start(&request, &images)
                .map_err(error)?,
        ))
    }

    /// Runs the next stage; returns `{"stage", "read", "rows", "response"}`, the response
    /// (the reference's) once the whole prompt is read, else `null`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }
}

/// The images `rgba` holds, of `sizes` (width, height) each.
fn images(rgba: &[u8], sizes: &[u32]) -> Result<Vec<Pixels>, intern_decision::Error> {
    let mut rest = rgba;
    let images = sizes.as_chunks::<2>().0.iter().map(|&[width, height]| {
        let size = Size {
            width: width as usize,
            height: height as usize,
        };
        let len = size.width * size.height * RGBA;
        let (values, after) = rest.split_at_checked(len).ok_or_else(|| {
            intern_decision::Error::Request(format!("too few bytes for a {width}×{height} image"))
        })?;
        rest = after;
        Ok(Pixels {
            size,
            channels: RGBA,
            values: values.to_vec(),
        })
    });
    let images = images.collect::<Result<Vec<_>, intern_decision::Error>>()?;
    if !rest.is_empty() {
        return Err(intern_decision::Error::Request(format!(
            "{} bytes past the images' sizes",
            rest.len()
        )));
    }
    Ok(images)
}
