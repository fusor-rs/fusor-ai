//! Text lines from the detector's maps: the page scaled to the detector's long side and padded
//! to whole multiples of 32, the line cores (the core probability over one half) as 4-connected
//! regions labeled in raster order of their first pixel, and per region a box: its columns
//! widened by `CORE_X_SHRINK` of its height each side, its top and bottom the means of its
//! pixels' distances.

use kernels::conv::Grid;
use kernels::numpy::median;
use media::image::{Pixels, Size};
use media::opencv::{resize_area, resize_linear};

use crate::crops::LineBox;
use crate::detector::{INPUT_MULTIPLE, MAPS, OUT_STRIDE};

/// A second pass reads the page larger when at least this many lines come out under
/// `ADAPTIVE_MIN_HEIGHT` input pixels high (median), at the scale that makes them
/// `ADAPTIVE_TARGET`, at most `ADAPTIVE_MAX_SIDE` long.
const ADAPTIVE_MIN_LINES: usize = 8;
const ADAPTIVE_MIN_HEIGHT: f32 = 9.0;
const ADAPTIVE_TARGET: f64 = 12.0;
const ADAPTIVE_MAX_SIDE: f64 = 2048.0;
const CORE_X_SHRINK: f64 = 0.15;
const CORE_THRESHOLD: f32 = 0.5;
/// Regions smaller than this many map pixels are not lines.
const MIN_AREA: usize = 2;
const INK_SCALE: f32 = 255.0;
/// The label of map pixels outside every region.
const BACKGROUND: u32 = 0;

/// A detected line: its box and the mean core probability of its region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub line: LineBox,
    pub score: f32,
}

/// The detector's input: the ink map of `page` scaled to `long_side` on its long side and
/// padded to multiples of `INPUT_MULTIPLE`, its grid, and the scale.
pub fn detector_input(page: &Pixels, long_side: usize) -> (Vec<f32>, Grid, f64) {
    let Size { width, height } = page.size;
    let scale = long_side as f64 / width.max(height) as f64;
    let side = |length: usize| ((length as f64 * scale).round_ties_even() as usize).max(1);
    let size = Size {
        width: side(width),
        height: side(height),
    };
    let scaled = match size == page.size {
        true => page.clone(),
        false if scale < 1.0 => resize_area(page, size),
        false => resize_linear(page, size),
    };
    let grid = Grid {
        images: 1,
        height: size.height.next_multiple_of(INPUT_MULTIPLE),
        width: size.width.next_multiple_of(INPUT_MULTIPLE),
    };
    let mut ink = vec![0.0; grid.pixels()];
    for (row, levels) in scaled.values.chunks_exact(size.width).enumerate() {
        for (out, &level) in ink[row * grid.width..].iter_mut().zip(levels) {
            *out = 1.0 - f32::from(level) / INK_SCALE;
        }
    }
    (ink, grid, scale)
}

/// The scale of the second pass over lines detected at `long_side` (input pixels), if the page
/// needs one: a long side over `long_side`, a multiple of `INPUT_MULTIPLE`.
pub fn adaptive_side(detections: &[Detection], long_side: usize) -> Option<usize> {
    if detections.len() < ADAPTIVE_MIN_LINES {
        return None;
    }
    let heights: Vec<f32> = detections
        .iter()
        .map(|detection| detection.line[3] - detection.line[1])
        .collect();
    let median = median(&heights);
    if median >= ADAPTIVE_MIN_HEIGHT {
        return None;
    }
    let target =
        (long_side as f64 * ADAPTIVE_TARGET / f64::from(median.max(1.0))).min(ADAPTIVE_MAX_SIDE);
    let side = target as usize / INPUT_MULTIPLE * INPUT_MULTIPLE;
    (side > long_side).then_some(side)
}

/// `detections` (input pixels at `scale`) in page pixels, clipped to the page.
pub fn to_page(detections: &mut [Detection], scale: f64, page: Size) {
    let scale = scale as f32;
    let (width, height) = (page.width as f32, page.height as f32);
    for detection in detections {
        let [x0, y0, x1, y1] = detection.line.map(|value| value / scale);
        detection.line = [
            x0.clamp(0.0, width),
            y0.clamp(0.0, height),
            x1.clamp(0.0, width),
            y1.clamp(0.0, height),
        ];
    }
}

/// The lines of the detector's maps (`grid` at stride `OUT_STRIDE`), in input pixels.
pub fn decode(maps: &[f32], grid: Grid) -> Vec<Detection> {
    let probabilities: Vec<f32> = maps
        .as_chunks::<MAPS>()
        .0
        .iter()
        .map(|pixel| 1.0 / (1.0 + (-pixel[0]).exp()))
        .collect();
    let mask: Vec<bool> = probabilities
        .iter()
        .map(|&probability| probability > CORE_THRESHOLD)
        .collect();
    let (labels, regions) = label(&mask, grid.width);
    let stride = OUT_STRIDE as f32;
    let mut sums = vec![[0.0f64; 3]; regions.len()];
    for (pixel, &region) in labels.iter().enumerate() {
        if region == BACKGROUND {
            continue;
        }
        let center = ((pixel / grid.width) as f32 + 0.5) * stride;
        let map = &maps[pixel * MAPS..(pixel + 1) * MAPS];
        let sum = &mut sums[region as usize - 1];
        sum[0] += f64::from(center - map[1]);
        sum[1] += f64::from(center + map[2]);
        sum[2] += f64::from(probabilities[pixel]);
    }
    regions
        .iter()
        .zip(sums)
        .filter(|(region, _)| region.area >= MIN_AREA)
        .map(|(region, [top, bottom, score])| {
            let count = f64::from(region.area as f32);
            let (top, bottom) = (top / count, bottom / count);
            let line_height = (bottom - top).max(1.0);
            let reach = (CORE_X_SHRINK * line_height).min(0.5 * line_height);
            let left = (region.left * OUT_STRIDE) as f64 - reach;
            let right = ((region.left + region.width) * OUT_STRIDE) as f64 + reach;
            Detection {
                line: [left as f32, top as f32, right as f32, bottom as f32],
                score: (score / count) as f32,
            }
        })
        .collect()
}

/// A region's columns and pixel count.
struct Region {
    left: usize,
    width: usize,
    area: usize,
}

/// The 4-connected regions of `mask` (`width` wide): each pixel's region (`BACKGROUND`, or
/// 1 + its index), numbered in raster order of their first pixel.
fn label(mask: &[bool], width: usize) -> (Vec<u32>, Vec<Region>) {
    let mut parents: Vec<usize> = Vec::new();
    let mut provisional = vec![usize::MAX; mask.len()];
    for pixel in (0..mask.len()).filter(|&pixel| mask[pixel]) {
        let above = (pixel >= width)
            .then(|| provisional[pixel - width])
            .filter(|&at| at != usize::MAX);
        let left = (pixel % width > 0)
            .then(|| provisional[pixel - 1])
            .filter(|&at| at != usize::MAX);
        provisional[pixel] = match (above, left) {
            (Some(above), Some(left)) => union(&mut parents, above, left),
            (Some(found), None) | (None, Some(found)) => found,
            (None, None) => {
                parents.push(parents.len());
                parents.len() - 1
            }
        };
    }
    let mut numbers = vec![BACKGROUND; parents.len()];
    let mut regions: Vec<Region> = Vec::new();
    let mut labels = vec![BACKGROUND; mask.len()];
    for (pixel, &at) in provisional
        .iter()
        .enumerate()
        .filter(|&(_, &at)| at != usize::MAX)
    {
        let root = find(&mut parents, at);
        if numbers[root] == BACKGROUND {
            regions.push(Region {
                left: pixel % width,
                width: 0,
                area: 0,
            });
            numbers[root] = regions.len() as u32;
        }
        let region = &mut regions[numbers[root] as usize - 1];
        let column = pixel % width;
        let right = (region.left + region.width).max(column + 1);
        region.left = region.left.min(column);
        region.width = right - region.left;
        region.area += 1;
        labels[pixel] = numbers[root];
    }
    (labels, regions)
}

fn find(parents: &mut [usize], at: usize) -> usize {
    let mut root = at;
    while parents[root] != root {
        root = parents[root];
    }
    let mut node = at;
    while parents[node] != root {
        (parents[node], node) = (root, parents[node]);
    }
    root
}

/// Joins the sets of `first` and `second` under the smaller root; returns it.
fn union(parents: &mut [usize], first: usize, second: usize) -> usize {
    let (first, second) = (find(parents, first), find(parents, second));
    let root = first.min(second);
    parents[first.max(second)] = root;
    root
}
