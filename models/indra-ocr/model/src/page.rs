//! The page before detection: its preview, the skew that straightens its text rows, the
//! orientation network's thumbnail, and the page turned upright and straightened.

use kernels::numpy::argmax;
use media::image::{Pixels, Size};
use media::opencv::{Border, resize_area, rotate, rotate_quarters};

use crate::layout::rounded;
use crate::orient::SIDE;

/// The skew search and the orientation thumbnail read a preview of at most this side.
const PREVIEW_SIDE: f64 = 1600.0;
/// The skew search's thumbnail side, and the coarse pass's.
const SKEW_SIDE: f64 = 800.0;
const COARSE_SIDE: f64 = 400.0;
/// A page with less ink than this share of full ink is blank: no rows to straighten.
const BLANK: f64 = 0.003;
/// Degrees of tilt searched, and the coarse search's step.
const SKEW_MAX: f64 = 45.0;
const COARSE_STEP: f64 = 2.0;
/// The refinements: around each start, `REFINE_REACH` degrees in steps of `REFINE_STEP`, then
/// around the best `FINE_REACH` in steps of `FINE_STEP`.
const REFINE_REACH: f64 = 2.0;
const REFINE_STEP: f64 = 0.5;
const FINE_REACH: f64 = 0.5;
const FINE_STEP: f64 = 0.1;
/// Slack that keeps the ranges' last values.
const RANGE_SLACK: f64 = 1e-6;
/// Smaller tilts are left alone.
pub const SKEW_MIN: f64 = 0.3;
/// The orientation class of a page set in vertical columns.
pub const VERTICAL: usize = 4;
const QUARTER_TURNS: usize = 4;
const MAX_LEVEL: u8 = 255;
const INK_SCALE: f32 = 255.0;

/// `image` scaled by `scale` (each side rounded, at least 1) with `INTER_AREA`.
fn scaled(image: &Pixels, scale: f64) -> Pixels {
    let side = |length: usize| ((length as f64 * scale).round_ties_even() as usize).max(1);
    let size = Size {
        width: side(image.size.width),
        height: side(image.size.height),
    };
    resize_area(image, size)
}

fn long_side(image: &Pixels) -> f64 {
    image.size.width.max(image.size.height) as f64
}

/// The page shrunk once to at most `PREVIEW_SIDE` a side, or itself.
pub fn preview(page: &Pixels) -> Pixels {
    let scale = PREVIEW_SIDE / long_side(page);
    if scale >= 1.0 {
        return page.clone();
    }
    scaled(page, scale)
}

/// The ink map (255 = ink) of the page scaled to `side` on its long side.
fn ink_thumbnail(page: &Pixels, side: f64) -> Pixels {
    let mut thumbnail = scaled(page, side / long_side(page));
    thumbnail
        .values
        .iter_mut()
        .for_each(|level| *level = MAX_LEVEL - *level);
    thumbnail
}

/// How sharply the horizontal ink profile changes row to row with the ink turned `degrees`:
/// highest when the text rows are level.
fn row_sharpness(ink: &Pixels, degrees: f64) -> f64 {
    let turned = rotate(ink, degrees, Border::Constant(0));
    let profile: Vec<i64> = turned
        .values
        .chunks_exact(ink.size.width)
        .map(|row| row.iter().map(|&level| i64::from(level)).sum())
        .collect();
    profile
        .windows(2)
        .map(|pair| ((pair[1] - pair[0]) as f64).powi(2))
        .sum()
}

/// `np.arange(start, stop, step)`: `start + i · delta`, `delta` the rounded `start + step −
/// start`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let count = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..count)
        .map(|index| start + index as f64 * delta)
        .collect()
}

/// Degrees counter-clockwise that straighten the page's text rows: the angle whose horizontal
/// ink profile changes most sharply, searched over `±SKEW_MAX` on a coarse thumbnail, then
/// refined around the two best angles and 0 on a finer one.
pub fn skew(page: &Pixels) -> f64 {
    let fine = ink_thumbnail(page, SKEW_SIDE);
    let ink: u64 = fine.values.iter().map(|&level| u64::from(level)).sum();
    if (ink as f64 / fine.values.len() as f64) < BLANK * f64::from(MAX_LEVEL) {
        return 0.0;
    }
    let coarse = ink_thumbnail(page, COARSE_SIDE);
    let grid = arange(-SKEW_MAX, SKEW_MAX + RANGE_SLACK, COARSE_STEP);
    let scores: Vec<f64> = grid
        .iter()
        .map(|&angle| row_sharpness(&coarse, angle))
        .collect();
    let mut ranked: Vec<usize> = (0..grid.len()).collect();
    ranked.sort_by(|&left, &right| scores[left].total_cmp(&scores[right]));
    let mut starts = vec![0.0];
    for &index in &ranked[ranked.len() - 2..] {
        if !starts.contains(&grid[index]) {
            starts.push(grid[index]);
        }
    }
    let near: Vec<f64> = starts
        .iter()
        .flat_map(|&start| {
            arange(
                start - REFINE_REACH,
                start + REFINE_REACH + RANGE_SLACK,
                REFINE_STEP,
            )
        })
        .collect();
    let mut seen: Vec<(f64, f64)> = Vec::new();
    let mut near_scores = Vec::with_capacity(near.len());
    for &angle in &near {
        let score = match seen.iter().find(|&&(scored, _)| scored == angle) {
            Some(&(_, score)) => score,
            None => {
                let score = row_sharpness(&fine, angle);
                seen.push((angle, score));
                score
            }
        };
        near_scores.push(score);
    }
    let refined = near[argmax(&near_scores)];
    let last = arange(
        refined - FINE_REACH,
        refined + FINE_REACH + RANGE_SLACK,
        FINE_STEP,
    );
    let last_scores: Vec<f64> = last
        .iter()
        .map(|&angle| row_sharpness(&fine, angle))
        .collect();
    last[argmax(&last_scores)]
}

/// The orientation network's input: the page's ink (`1 − gray/255`) scaled to `SIDE` on its
/// long side, at the top left of a `SIDE × SIDE` square.
pub fn orientation_input(page: &Pixels) -> Vec<f32> {
    let small = scaled(page, SIDE as f64 / long_side(page));
    let mut out = vec![0.0; SIDE * SIDE];
    for (row, levels) in small.values.chunks_exact(small.size.width).enumerate() {
        for (out, &level) in out[row * SIDE..].iter_mut().zip(levels) {
            *out = 1.0 - f32::from(level) / INK_SCALE;
        }
    }
    out
}

/// The page turned `turn` quarter turns back (a vertical page: one quarter turn
/// counter-clockwise, its columns become lines) and straightened by `degrees` (measured again
/// after a quarter turn, on the preview turned alike): the page and the degrees undone.
pub fn upright(page: &Pixels, preview: &Pixels, turn: usize, degrees: f64) -> (Pixels, f64) {
    let quarters = if turn == VERTICAL {
        1
    } else {
        (QUARTER_TURNS - turn) % QUARTER_TURNS
    };
    let turned = rotate_quarters(page, quarters);
    let degrees = if quarters % 2 == 1 {
        skew(&rotate_quarters(preview, quarters))
    } else {
        degrees
    };
    if degrees.abs() >= SKEW_MIN {
        (rotate(&turned, degrees, Border::Replicate), degrees)
    } else {
        (turned, 0.0)
    }
}

/// The preview straightened by `degrees` when that tilt is worth straightening: what the
/// orientation network reads.
pub fn straightened(preview: &Pixels, degrees: f64) -> Pixels {
    if degrees.abs() >= SKEW_MIN {
        rotate(preview, degrees, Border::Replicate)
    } else {
        preview.clone()
    }
}

/// A box of the page that straightening by `degrees` made, mapped back to the page before it:
/// the bounding box of its four corners turned back, clipped to the page, to one decimal.
pub fn unrotate(bbox: [f64; 4], degrees: f64, (width, height): (usize, usize)) -> [f64; 4] {
    let (width, height) = (width as f64, height as f64);
    let (center_x, center_y) = (width / 2.0, height / 2.0);
    let angle = (-degrees).to_radians();
    let (cos, sin) = (angle.cos(), angle.sin());
    let [x0, y0, x1, y1] = bbox;
    let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
    let turned = corners.map(|(x, y)| {
        let across = cos * x + sin * y + ((1.0 - cos) * center_x - sin * center_y);
        let down = -sin * x + cos * y + (sin * center_x + (1.0 - cos) * center_y);
        (across, down)
    });
    let xs = turned.map(|(x, _)| x);
    let ys = turned.map(|(_, y)| y);
    let low = |values: [f64; 4]| values.into_iter().fold(f64::INFINITY, f64::min);
    let high = |values: [f64; 4]| values.into_iter().fold(f64::NEG_INFINITY, f64::max);
    [
        rounded(low(xs).clamp(0.0, width), 1),
        rounded(low(ys).clamp(0.0, height), 1),
        rounded(high(xs).clamp(0.0, width), 1),
        rounded(high(ys).clamp(0.0, height), 1),
    ]
}
