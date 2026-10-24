use qwen3_5::session::{Engine, Message};
use qwen3_5::{Pixels, Size, Video};
use runtime::{error, json};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Channels of the pixels the page pushes: RGBA, as a canvas gives them.
const RGBA: usize = 4;

runtime::runtime!(Engine);

/// Where the page's pixels lie: each image's size, then each video's frame size and its frames'
/// times in seconds, every frame and image one after another.
#[derive(Deserialize)]
struct Layout {
    images: Vec<[usize; 2]>,
    videos: Vec<VideoLayout>,
}

#[derive(Deserialize)]
struct VideoLayout {
    width: usize,
    height: usize,
    times: Vec<f64>,
}

#[wasm_bindgen]
impl Runtime {
    /// Starts replying to `message_json` (`{"text", "thinking"?, "max_tokens"?, "sampling"?,
    /// "seed"}`), the conversation's next turn, with the images and videos whose RGBA pixels
    /// `rgba` holds as `layout_json` lays them out (`{"images": [[width, height]…], "videos":
    /// [{"width", "height", "times"}…]}`); returns its plan (`{"media", "prompt_tokens",
    /// "read_tokens", "max_tokens", "sampling"}`).
    pub fn start(
        &mut self,
        message_json: &str,
        rgba: &[u8],
        layout_json: &str,
    ) -> Result<String, JsError> {
        let invalid =
            |problem: serde_json::Error| error(qwen3_5::Error::Request(problem.to_string()));
        let message: Message = serde_json::from_str(message_json).map_err(invalid)?;
        let layout: Layout = serde_json::from_str(layout_json).map_err(invalid)?;
        let (images, videos) = media(rgba, &layout).map_err(error)?;
        Ok(json(
            &self
                .shell
                .engine()?
                .start(&message, &images, &videos)
                .map_err(error)?,
        ))
    }

    /// Runs the next stage; returns `{"stage", "reasoning", "answer", "tokens", "done"}`, the
    /// reply so far.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// Forgets the conversation, to start a new one.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.shell.engine()?.reset();
        Ok(())
    }
}

/// The images and videos `rgba` holds as `layout` lays them out.
fn media(rgba: &[u8], layout: &Layout) -> Result<(Vec<Pixels>, Vec<Video>), qwen3_5::Error> {
    let mut rest = rgba;
    let mut take = |width: usize, height: usize| {
        let (values, after) = rest
            .split_at_checked(width * height * RGBA)
            .ok_or_else(|| {
                qwen3_5::Error::Request(format!("too few bytes for a {width}×{height} picture"))
            })?;
        rest = after;
        Ok(Pixels {
            size: Size { width, height },
            channels: RGBA,
            values: values.to_vec(),
        })
    };
    let images = layout
        .images
        .iter()
        .map(|&[width, height]| take(width, height))
        .collect::<Result<Vec<_>, qwen3_5::Error>>()?;
    let videos = layout
        .videos
        .iter()
        .map(|video| {
            let frames = video.times.iter().map(|_| take(video.width, video.height));
            Ok(Video {
                frames: frames.collect::<Result<_, qwen3_5::Error>>()?,
                times: video.times.clone(),
            })
        })
        .collect::<Result<Vec<_>, qwen3_5::Error>>()?;
    if !rest.is_empty() {
        return Err(qwen3_5::Error::Request(format!(
            "{} bytes past the media's sizes",
            rest.len()
        )));
    }
    Ok((images, videos))
}
