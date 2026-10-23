//! Images and videos as the model reads them, every patch spanning two frames.
//!
//! An image is read as Qwen3.5's processor reads it (`Preprocessing::centered_patches`), its
//! frame repeated, at 65,536 to 1,638,400 pixels. A video arrives as frames already sampled
//! (the reference samples 2 a second, 4 to 32 spread over the whole video) with each frame's time;
//! an odd last frame is repeated, every frame resized alike to at most 200,704 pixels, and each
//! two consecutive frames are one group, read by the tower as an image is and placed in the
//! prompt after its time, the mean of its frames' (`<1.5 seconds>`).

use blocks::vit::preprocessing::{Patches, centered};
use blocks::vit::{Grid, Preprocessing};
use media::image::{Filter, Pixels, Size};
use serde::Serialize;

use crate::Error;
use crate::model::{FRAMES, MERGE, PATCH};

pub const IMAGE: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 65_536,
    max_pixels: 1_638_400,
    filter: Filter::Bicubic,
};
/// A video frame's budget: its merged blocks only shrink it.
pub const VIDEO: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 0,
    max_pixels: 200_704,
    filter: Filter::Bicubic,
};
/// The widest aspect ratio the processor accepts.
const MAX_ASPECT: usize = 200;
/// The most frames a video takes.
pub const MAX_FRAMES: usize = 32;
/// The most language-model tokens a request's videos take.
pub const MAX_VIDEO_TOKENS: usize = 16_384;
/// Values of one frame of one channel of a patch.
const PLANE: usize = PATCH * PATCH;

/// A video's sampled frames, each with its time in seconds.
pub struct Video {
    pub frames: Vec<Pixels>,
    pub times: Vec<f64>,
}

/// Two consecutive frames of a video, as patches, and their time.
pub struct Group {
    pub patches: Patches,
    pub time: f64,
}

/// A request's images and videos, every one as two-frame patches.
pub struct Media {
    pub images: Vec<Patches>,
    pub videos: Vec<Vec<Group>>,
}

/// How a request's images and videos are read: each image's and each video's patch grid, and
/// the videos' groups.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub images: Vec<[usize; 2]>,
    pub videos: Vec<[usize; 3]>,
}

impl Media {
    /// # Errors
    /// When an image is empty or too narrow, or a video's frames are none, too many, of
    /// different sizes or without one time each, or the videos take too many tokens.
    pub fn new(images: &[Pixels], videos: &[Video]) -> Result<Self, Error> {
        let images = images
            .iter()
            .map(|image| {
                IMAGE
                    .centered_patches(image, MAX_ASPECT)
                    .map(|patches| doubled(&patches))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let videos = videos.iter().map(groups).collect::<Result<Vec<_>, _>>()?;
        let video_tokens: usize = videos
            .iter()
            .flatten()
            .map(|group| VIDEO.tokens(group.patches.grid.patches()))
            .sum();
        if video_tokens > MAX_VIDEO_TOKENS {
            return Err(Error::Request(format!(
                "the videos take {video_tokens} input tokens, over the {MAX_VIDEO_TOKENS} a \
                 request may spend on videos; send fewer or shorter videos"
            )));
        }
        Ok(Self { images, videos })
    }

    /// Every image and video group, in prompt order.
    pub fn items(&self) -> impl Iterator<Item = &Patches> {
        let groups = self.videos.iter().flatten().map(|group| &group.patches);
        self.images.iter().chain(groups)
    }

    pub fn plan(&self) -> Plan {
        let grid = |grid: Grid| [grid.rows, grid.columns];
        let videos = self.videos.iter().map(|groups| {
            let [rows, columns] = grid(groups[0].patches.grid);
            [groups.len(), rows, columns]
        });
        Plan {
            images: self.images.iter().map(|image| grid(image.grid)).collect(),
            videos: videos.collect(),
        }
    }
}

/// An image's patches with its frame repeated: each channel's plane twice.
fn doubled(patches: &Patches) -> Patches {
    let values = patches
        .values
        .as_chunks::<PLANE>()
        .0
        .iter()
        .flat_map(|plane| plane.iter().chain(plane))
        .copied()
        .collect();
    Patches {
        grid: patches.grid,
        values,
    }
}

/// A video's frames, two by two, as patches.
fn groups(video: &Video) -> Result<Vec<Group>, Error> {
    let Video { frames, times } = video;
    if frames.is_empty() || frames.len() > MAX_FRAMES || times.len() != frames.len() {
        return Err(Error::Request(format!(
            "a video is 1 to {MAX_FRAMES} frames, each with its time"
        )));
    }
    let Size { width, height } = frames[0].size;
    if frames.iter().any(|frame| frame.size != frames[0].size) {
        return Err(Error::Request(
            "all frames of a video must have the same size".into(),
        ));
    }
    let factor = VIDEO.factor();
    if width < factor || height < factor {
        return Err(Error::Request(format!(
            "a video is {width}×{height}: both sides must be at least {factor} pixels"
        )));
    }
    let size = VIDEO.whole_blocks(height as f64, width as f64);
    let frame = |index: usize| {
        let at = index.min(frames.len() - 1);
        (VIDEO.patches(&frames[at], size, centered), times[at])
    };
    let groups = (0..frames.len().div_ceil(FRAMES)).map(|group| {
        let ((first, start), (second, end)) = (frame(FRAMES * group), frame(FRAMES * group + 1));
        Group {
            patches: paired(&first, &second),
            time: (start + end) / 2.0,
        }
    });
    Ok(groups.collect())
}

/// Two frames' patches as one patch each: per channel, the first frame's plane, then the
/// second's.
fn paired(first: &Patches, second: &Patches) -> Patches {
    let planes = first.values.as_chunks::<PLANE>().0.iter();
    let planes = planes.zip(second.values.as_chunks::<PLANE>().0);
    let values = planes.flat_map(|(one, two)| one.iter().chain(two)).copied();
    Patches {
        grid: first.grid,
        values: values.collect(),
    }
}
