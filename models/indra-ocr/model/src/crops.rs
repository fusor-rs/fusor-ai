//! Line crops as the recognizers were trained on them: a line box widened by `PAD_X` of its
//! height left and right and `PAD_Y` above and below (a tall window adds the page around it at
//! the same scale), cut from the page with paper past its edges, scaled to the band's height,
//! and made an ink map (`1 − gray/255`, a light-on-dark line inverted); then batches of
//! similar widths, padded to one.

use media::image::{Pixels, Size};
use media::opencv::{resize_area, resize_linear};

/// A line box: x0, y0, x1, y1 in page pixels.
pub type LineBox = [f32; 4];

pub const PAD_X: f64 = 0.3;
const PAD_Y: f64 = 0.2;
/// Crop widths are at least this, and padded batches a multiple of it.
const WIDTH_MULTIPLE: usize = 8;
const MAX_WIDTH: usize = 4096;
const PAPER: u8 = 255;
const DARK_PAPER: u8 = 0;
const MAX_LEVEL: f32 = 255.0;
const LEVELS: usize = 256;
/// The percentiles of a line's gray levels that decide whether it is light on dark.
const PERCENTILES: [f64; 3] = [0.02, 0.5, 0.98];
/// Light on dark: the bright tail longer than twice the dark one plus this.
const DARK_MARGIN: f64 = 8.0;

/// An ink map, `height × width`.
pub struct Crop {
    pub width: usize,
    pub values: Vec<f32>,
}

/// How batches are cut (`width_batches`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Batching {
    pub max_lines: usize,
    /// A batch closes when a line is wider than this times its first line's width.
    pub max_ratio: f64,
    /// Widths under this count as this.
    pub min_width: usize,
    /// Lines times width a batch stays within.
    pub max_area: usize,
}

pub const RECOGNIZER_BATCHING: Batching = Batching {
    max_lines: 64,
    max_ratio: 1.3,
    min_width: 64,
    max_area: 64 * 640,
};

/// Light text on a dark ground: the box's median nearer the dark end than the text is.
pub fn is_dark(page: &Pixels, line: LineBox) -> bool {
    let Size { width, height } = page.size;
    let [x0, y0, x1, y1] = line.map(f64::from);
    let rows = (y0.max(0.0) as usize)..((y1 as usize + 1).min(height));
    let columns = (x0.max(0.0) as usize)..((x1 as usize + 1).min(width));
    if rows.is_empty() || columns.is_empty() {
        return false;
    }
    let mut histogram = [0usize; LEVELS];
    for row in rows {
        for &level in &page.values[row * width + columns.start..row * width + columns.end] {
            histogram[usize::from(level)] += 1;
        }
    }
    let [low, median, high] = percentiles(&histogram);
    high - median > 2.0 * (median - low) + DARK_MARGIN
}

/// `np.percentile` (linear) of the levels counted by `histogram`, at `PERCENTILES`.
fn percentiles(histogram: &[usize; LEVELS]) -> [f64; 3] {
    let mut cumulative = [0usize; LEVELS];
    let mut total = 0;
    for (sum, count) in cumulative.iter_mut().zip(histogram) {
        total += count;
        *sum = total;
    }
    let level_above = |rank: usize| cumulative.partition_point(|&sum| sum <= rank) as f64;
    PERCENTILES.map(|quantile| {
        let rank = quantile * (total - 1) as f64;
        let index = rank.floor();
        let fraction = rank - index;
        let low = level_above(index as usize);
        let high = level_above((index as usize + 1).min(total - 1));
        let span = high - low;
        if fraction >= 0.5 {
            high - span * (1.0 - fraction)
        } else {
            low + span * fraction
        }
    })
}

/// The crop of `line`, `height · window` pixels high, `window` bands of the line's scale with
/// the line in the middle one. `dark`: `is_dark` of the line.
pub fn crop(page: &Pixels, line: LineBox, (height, window): (usize, usize), dark: bool) -> Crop {
    let pad_y = (window as f64 * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let height = height * window;
    let [x0, y0, x1, y1] = line.map(f64::from);
    let line_height = (y1 - y0).max(1.0);
    let left = (x0 - PAD_X * line_height).floor() as i64;
    let top = (y0 - pad_y * line_height).floor() as i64;
    let right = (x1 + PAD_X * line_height).ceil() as i64;
    let bottom = (y1 + pad_y * line_height).ceil() as i64;
    let size = Size {
        width: (right - left).max(1) as usize,
        height: (bottom - top).max(1) as usize,
    };
    let paper = if dark { DARK_PAPER } else { PAPER };
    let mut cut = Pixels {
        size,
        channels: 1,
        values: vec![paper; size.width * size.height],
    };
    let (first, last) = (left.max(0), right.min(page.size.width as i64));
    if first < last {
        let len = (last - first) as usize;
        for row in top.max(0)..bottom.min(page.size.height as i64) {
            let source = row as usize * page.size.width + first as usize;
            let target = (row - top) as usize * size.width + (first - left) as usize;
            cut.values[target..target + len].copy_from_slice(&page.values[source..source + len]);
        }
    }
    let scaled_width =
        (size.width as f64 * height as f64 / size.height as f64).round_ties_even() as usize;
    let target = Size {
        width: scaled_width.clamp(WIDTH_MULTIPLE, MAX_WIDTH),
        height,
    };
    let resized = if size.height > height {
        resize_area(&cut, target)
    } else {
        resize_linear(&cut, target)
    };
    let values = resized.values.iter().map(|&level| {
        let ink = f32::from(level) / MAX_LEVEL;
        if dark { ink } else { 1.0 - ink }
    });
    Crop {
        width: target.width,
        values: values.collect(),
    }
}

/// Crops of one height right-padded with zeros to one width, a multiple of `WIDTH_MULTIPLE`:
/// their values and that width.
pub fn pad(crops: &[&Crop], height: usize) -> (Vec<f32>, usize) {
    let widest = crops
        .iter()
        .map(|crop| crop.width)
        .max()
        .expect("a batch holds at least one line");
    let width = widest.next_multiple_of(WIDTH_MULTIPLE);
    let mut out = vec![0.0; crops.len() * height * width];
    for (index, crop) in crops.iter().enumerate() {
        for row in 0..height {
            let target = (index * height + row) * width;
            out[target..target + crop.width]
                .copy_from_slice(&crop.values[row * crop.width..(row + 1) * crop.width]);
        }
    }
    (out, width)
}

/// Line indices grouped into batches of similar width, narrowest first.
pub fn width_batches(widths: &[usize], batching: Batching) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&index| widths[index]);
    let counted = |index: usize| widths[index].max(batching.min_width);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let fits = batches.last().is_some_and(|batch| {
            batch.len() < batching.max_lines
                && counted(index) as f64 <= batching.max_ratio * counted(batch[0]) as f64
                && (batch.len() + 1) * counted(index) <= batching.max_area
        });
        match batches.last_mut() {
            Some(batch) if fits => batch.push(index),
            _ => batches.push(vec![index]),
        }
    }
    batches
}
