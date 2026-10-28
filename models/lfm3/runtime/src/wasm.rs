use lfm3::session::{Engine, Message};
use lfm3::{Pixels, Size};
use runtime::{error, json};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

/// Where the page's pixels and samples lie: each image's size and each clip's samples, every
/// image, and every clip, one after another.
#[derive(Deserialize)]
struct Layout {
    images: Vec<[usize; 2]>,
    #[serde(default)]
    clips: Vec<usize>,
}

#[wasm_bindgen]
impl Runtime {
    /// Starts replying to `message_json` (`{"text", "system"?, "thinking"?, "mode"?,
    /// "max_tokens"?, "sampling"?, "seed"}`, `mode` one of `text`, `transcribe`, `speech` and
    /// `interleaved`), the conversation's next turn, with the images whose RGBA pixels `rgba`
    /// holds and the 16 kHz mono clips whose samples `audio` holds, as `layout_json` lays them out
    /// (`{"images": [[width, height]…], "clips": [samples…]}`); returns its plan (`{"images",
    /// "clips", "prompt_tokens", "max_tokens", "sampling"}`).
    pub fn start(
        &mut self,
        message_json: &str,
        rgba: &[u8],
        audio: &[f32],
        layout_json: &str,
    ) -> Result<String, JsError> {
        let invalid = |problem: serde_json::Error| error(lfm3::Error::Request(problem.to_string()));
        let message: Message = serde_json::from_str(message_json).map_err(invalid)?;
        let layout: Layout = serde_json::from_str(layout_json).map_err(invalid)?;
        let images = images(rgba, &layout).map_err(error)?;
        let clips = clips(audio, &layout).map_err(error)?;
        let engine = self.shell.engine()?;
        Ok(json(
            &engine.start(&message, &images, &clips).map_err(error)?,
        ))
    }

    /// Runs the next stage; returns `{"stage", "reasoning", "answer", "tokens", "frames",
    /// "done"}`, the reply so far.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The 24 kHz mono samples the reply spoke since the last call.
    pub fn speech(&mut self) -> Result<Vec<f32>, JsError> {
        Ok(self.shell.engine()?.speech())
    }

    /// Forgets the conversation, to start a new one.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.shell.engine()?.reset();
        Ok(())
    }
}

/// The images `rgba` holds as `layout` lays them out.
fn images(rgba: &[u8], layout: &Layout) -> Result<Vec<Pixels>, lfm3::Error> {
    let mut rest = rgba;
    let mut images = Vec::with_capacity(layout.images.len());
    for &[width, height] in &layout.images {
        let (values, after) = rest
            .split_at_checked(width * height * RGBA)
            .ok_or_else(|| {
                lfm3::Error::Request(format!("too few bytes for a {width}×{height} image"))
            })?;
        rest = after;
        images.push(Pixels {
            size: Size { width, height },
            channels: RGBA,
            values: values.to_vec(),
        });
    }
    if !rest.is_empty() {
        return Err(lfm3::Error::Request(format!(
            "{} bytes past the images' sizes",
            rest.len()
        )));
    }
    Ok(images)
}

/// The clips `audio` holds as `layout` lays them out.
fn clips(audio: &[f32], layout: &Layout) -> Result<Vec<Vec<f32>>, lfm3::Error> {
    let total: usize = layout.clips.iter().sum();
    if total != audio.len() {
        return Err(lfm3::Error::Request(format!(
            "{} samples for clips of {total}",
            audio.len()
        )));
    }
    let mut rest = audio;
    let clips = layout.clips.iter().map(|&samples| {
        let (clip, after) = rest.split_at(samples);
        rest = after;
        clip.to_vec()
    });
    Ok(clips.collect())
}
