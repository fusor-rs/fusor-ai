//! A turn's images and videos as the model reads them, every patch spanning two frames.
//!
//! An image is read as Qwen3.5's processor reads it (`Preprocessing::centered_patches`), its
//! frame repeated, at 65,536 to 1,638,400 pixels. A video arrives as frames already sampled as the
//! processor samples them (2 a second, 4 to 32 spread over the whole video) with each frame's
//! time; its frames are resized alike as the processor resizes them with each frame's pixels
//! capped (`cap_pixels_per_frame`), an odd last frame is repeated, and each two consecutive
//! frames are one group, read by the tower as an image is and placed in the prompt after its
//! time, the mean of its frames' (`<1.5 seconds>`).

use blocks::vit::preprocessing::{Group, Patches, Video};
use blocks::vit::{Grid, Preprocessing};
use media::image::{Filter, Pixels, Size};
use serde::Serialize;

use crate::Error;
use crate::model::{FRAMES, MERGE, PATCH};

/// The checkpoint's image budget, its 16.8 million pixels lowered to 1,600 tokens.
pub const IMAGE: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 65_536,
    max_pixels: 1_638_400,
    filter: Filter::Bicubic,
};
/// A video's budget over all its frames: the checkpoint's fewest pixels, and its most (25.2
/// million) lowered to 6,422,528, 3,136 tokens.
const VIDEO: Preprocessing = Preprocessing {
    patch: PATCH,
    merge: MERGE,
    min_pixels: 4_096,
    max_pixels: 6_422_528,
    filter: Filter::Bicubic,
};
/// A frame's most pixels however few the frames: 768 tokens (the processor's
/// `max_video_tokens`).
const FRAME_PIXELS: usize = 768 * 32 * 32;
/// A frame's fewest pixels: the processor's `int(1.05 × shortest_edge)`.
const MIN_FRAME_PIXELS: usize = 4_300;
/// The widest aspect ratio the processor accepts.
const MAX_ASPECT: usize = 200;
/// The most frames a video takes.
pub const MAX_FRAMES: usize = 32;
/// Values of one frame of one channel of a patch.
const PLANE: usize = PATCH * PATCH;

/// What a turn's image or video leaves in the conversation: its grids of language-model tokens,
/// a video's one per group with the group's time.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Image(Grid),
    Video(Vec<(Grid, f64)>),
}

/// A turn's images and videos, every one as two-frame patches.
pub struct Media {
    pub images: Vec<Patches>,
    pub videos: Vec<Vec<Group>>,
}

/// How a turn's images and videos are read: each image's and each video's patch grid, and the
/// videos' groups.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub images: Vec<[usize; 2]>,
    pub videos: Vec<[usize; 3]>,
}

impl Media {
    /// # Errors
    /// When an image is empty or too narrow, or a video's frames are too few or too many, of
    /// different sizes, too small or without one time each.
    pub fn new(images: &[Pixels], videos: &[Video]) -> Result<Self, Error> {
        let images = images
            .iter()
            .map(|image| {
                IMAGE
                    .centered_patches(image, MAX_ASPECT)
                    .map(|patches| patches.doubled(PLANE))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let videos = videos.iter().map(groups).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { images, videos })
    }

    /// Every image and video group, in prompt order.
    pub fn items(&self) -> impl Iterator<Item = &Patches> {
        let groups = self.videos.iter().flatten().map(|group| &group.patches);
        self.images.iter().chain(groups)
    }

    /// What the images and videos leave in the conversation, in prompt order.
    pub fn kept(&self) -> Vec<Item> {
        let images = self
            .images
            .iter()
            .map(|image| Item::Image(tokens(image.grid)));
        let videos = self.videos.iter().map(|groups| {
            let groups = groups
                .iter()
                .map(|group| (tokens(group.patches.grid), group.time));
            Item::Video(groups.collect())
        });
        images.chain(videos).collect()
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

/// The grid of language-model tokens of a grid of patches: one per merged block.
fn tokens(grid: Grid) -> Grid {
    Grid {
        rows: grid.rows / MERGE,
        columns: grid.columns / MERGE,
    }
}

/// A video's frames, two by two, as patches.
fn groups(video: &Video) -> Result<Vec<Group>, Error> {
    let Video { frames, times } = video;
    let count = frames.len();
    if !(FRAMES..=MAX_FRAMES).contains(&count) || times.len() != count {
        return Err(Error::Request(format!(
            "a video is {FRAMES} to {MAX_FRAMES} frames, each with its time"
        )));
    }
    let Size { width, height } = frames[0].size;
    if frames.iter().any(|frame| frame.size != frames[0].size) {
        return Err(Error::Request(
            "all frames of a video must have the same size".into(),
        ));
    }
    let factor = VIDEO.factor();
    if width < factor || height < factor || width.max(height) > MAX_ASPECT * width.min(height) {
        return Err(Error::Request(format!(
            "a video is {width}×{height}: both sides must be at least {factor} pixels, and its \
             aspect ratio at most {MAX_ASPECT}"
        )));
    }
    let per_frame = FRAME_PIXELS.min(VIDEO.max_pixels / count);
    let budget = Preprocessing {
        max_pixels: per_frame.max(MIN_FRAME_PIXELS) * count,
        ..VIDEO
    };
    let size = budget.whole_video_blocks(count, height as f64, width as f64);
    Ok(VIDEO.frame_pairs(video, size))
}
