use d1::request::Request;
use d1::session::Engine;
use d1::{Pixels, Size};
use runtime::{error, json};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

/// Where the page's media lie: each image's size, every image one after another in the RGBA
/// bytes, and each clip's samples, every clip one after another in the samples.
#[derive(Deserialize)]
struct Layout {
    images: Vec<[usize; 2]>,
    #[serde(default)]
    clips: Vec<usize>,
}

#[wasm_bindgen]
impl Runtime {
    /// Starts answering `request_json` (`{"state", "questions"}`, as the reference's
    /// `system_one` takes them) about the images whose RGBA pixels `rgba` holds and the mono
    /// 16 kHz clips whose samples `samples` holds, as `layout_json` lays them out
    /// (`{"images": [[width, height]…], "clips": [samples…]}`, clips for d1-Omni-2B only);
    /// returns its plan (`{"model", "images", "clips", "packs", "questions"}`).
    pub fn start(
        &mut self,
        request_json: &str,
        rgba: &[u8],
        samples: &[f32],
        layout_json: &str,
    ) -> Result<String, JsError> {
        let invalid = |problem: serde_json::Error| error(d1::Error::Request(problem.to_string()));
        let request: Request = serde_json::from_str(request_json).map_err(invalid)?;
        let layout: Layout = serde_json::from_str(layout_json).map_err(invalid)?;
        let images = images(rgba, &layout).map_err(error)?;
        let clips = clips(samples, &layout).map_err(error)?;
        let engine = self.shell.engine()?;
        Ok(json(
            &engine.start(&request, &images, &clips).map_err(error)?,
        ))
    }

    /// Runs the next stage; returns `{"stage", "done", "steps", "response"}`, the response
    /// (`{"answers", "usage"}`, the reference's) once every question is answered, else `null`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }
}

/// The images `rgba` holds as `layout` lays them out.
fn images(rgba: &[u8], layout: &Layout) -> Result<Vec<Pixels>, d1::Error> {
    let mut rest = rgba;
    let mut images = Vec::with_capacity(layout.images.len());
    for &[width, height] in &layout.images {
        let (values, after) = rest
            .split_at_checked(width * height * RGBA)
            .ok_or_else(|| {
                d1::Error::Request(format!("too few bytes for a {width}×{height} image"))
            })?;
        rest = after;
        images.push(Pixels {
            size: Size { width, height },
            channels: RGBA,
            values: values.to_vec(),
        });
    }
    if !rest.is_empty() {
        return Err(d1::Error::Request(format!(
            "{} bytes past the images' sizes",
            rest.len()
        )));
    }
    Ok(images)
}

/// The clips `samples` holds as `layout` lays them out.
fn clips(samples: &[f32], layout: &Layout) -> Result<Vec<Vec<f32>>, d1::Error> {
    if layout.clips.iter().sum::<usize>() != samples.len() {
        return Err(d1::Error::Request(format!(
            "{} samples for clips of {:?}",
            samples.len(),
            layout.clips
        )));
    }
    let mut rest = samples;
    let clips = layout.clips.iter().map(|&length| {
        let (clip, after) = rest.split_at(length);
        rest = after;
        clip.to_vec()
    });
    Ok(clips.collect())
}
