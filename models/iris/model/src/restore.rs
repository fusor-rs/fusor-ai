//! Iris's restorer around the transformer (`Restorer`): a photo shrunk to a budget, enlarged four
//! times, cut into 1024 × 1024 tiles that the model restores in one step each, fused with a
//! Gaussian window, and given back the enlargement's colours by a wavelet fix.

use media::image::{Filter, Pixels, Size, resize_bicubic, resize_bicubic_antialiased};

use crate::model::{IMAGE_CHANNELS, PATCH, TILE};

/// How many times the restored image is as wide as the budgeted photo.
const SCALE: usize = 4;
/// The variance of the fusion window in tile-normalized coordinates, doubled.
const WINDOW_VARIANCE: f32 = 0.02;
/// The wavelet colour fix's blurs, by the dilation of their 3 × 3 window.
const DILATIONS: [usize; 5] = [1, 2, 4, 8, 16];
/// The binomial window of those blurs.
const KERNEL: [[f32; 3]; 3] = [
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
];

/// Where the tiles go: the work image's size before and after padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Work {
    pub size: Size,
    pub padded: Size,
}

/// `fit_budget`: shrunk with Pillow's Lanczos (never enlarged) so its short side is at most
/// `short` and its long side at most `long`, sides `max(1, round(side · scale))`.
pub fn budget(photo: &Pixels, (short, long): (usize, usize)) -> Pixels {
    let Size { width, height } = photo.size;
    let scale = (short as f64 / width.min(height) as f64)
        .min(long as f64 / width.max(height) as f64)
        .min(1.0);
    if scale >= 1.0 {
        return photo.clone();
    }
    let side = |length: usize| ((length as f64 * scale).round_ties_even() as usize).max(1);
    let size = Size {
        width: side(width),
        height: side(height),
    };
    photo.resize(size, Filter::Lanczos)
}

/// The size of a photo enlarged `SCALE` times.
pub fn enlarged(photo: Size) -> Size {
    Size {
        width: photo.width * SCALE,
        height: photo.height * SCALE,
    }
}

/// The photo enlarged to `output` (`enlarged`) by PyTorch's bicubic, in [0, 1] (`v / 255`),
/// pixel-major RGB.
pub fn upsample(photo: &Pixels, output: Size) -> Vec<f32> {
    let unit: Vec<f32> = photo
        .values
        .iter()
        .map(|&value| f32::from(value) / f32::from(u8::MAX))
        .collect();
    resize_bicubic(&unit, photo.size, output, IMAGE_CHANNELS)
}

/// The work image of an enlargement of `output`: its short side `TILE` when that is at most
/// `TILE` (sides `max(TILE, round(side · TILE / short))`), padded right and down to multiples
/// of `PATCH`; none when a side overflows.
pub fn layout(output: Size) -> Option<Work> {
    let short = output.width.min(output.height);
    let size = if short <= TILE {
        let ratio = TILE as f64 / short as f64;
        let side = |length: usize| TILE.max((length as f64 * ratio).round_ties_even() as usize);
        Size {
            width: side(output.width),
            height: side(output.height),
        }
    } else {
        output
    };
    let padded = Size {
        width: size.width.checked_next_multiple_of(PATCH)?,
        height: size.height.checked_next_multiple_of(PATCH)?,
    };
    Some(Work { size, padded })
}

/// The tiles' input: `reference` (`output`) enlarged to `layout.size` by PyTorch's antialiased
/// bicubic when its short side is at most `TILE`, as `v · 2 − 1`, zero-padded to `layout.padded`.
pub fn work(reference: &[f32], output: Size, layout: &Work) -> Vec<f32> {
    let Work { size, padded } = *layout;
    let enlarged;
    let pixels = if output.width.min(output.height) <= TILE {
        enlarged = resize_bicubic_antialiased(reference, output, size, IMAGE_CHANNELS);
        enlarged.as_slice()
    } else {
        reference
    };
    let mut image = vec![0.0; padded.width * padded.height * IMAGE_CHANNELS];
    let rows = pixels.chunks_exact(size.width * IMAGE_CHANNELS);
    for (target, row) in image
        .chunks_exact_mut(padded.width * IMAGE_CHANNELS)
        .zip(rows)
    {
        for (padded_value, value) in target.iter_mut().zip(row) {
            *padded_value = value * 2.0 - 1.0;
        }
    }
    image
}

/// The top left corners of the tiles of an image of `size`, rows outer: along each axis stride
/// `TILE / 2`, the last tile flush with the far edge, a single one when the side is at most
/// `TILE`.
pub(crate) fn tiles(size: Size) -> Vec<(usize, usize)> {
    let along = |side: usize| -> Vec<usize> {
        if side <= TILE {
            return vec![0];
        }
        let mut starts: Vec<usize> = (0..side - TILE).step_by(TILE / 2).collect();
        starts.push(side - TILE);
        starts
    };
    let columns = along(size.width);
    along(size.height)
        .into_iter()
        .flat_map(|top| columns.iter().map(move |&left| (top, left)))
        .collect()
}

/// The tiles fused into one image of `size` (`Restorer.tiled`): each tile weighted by a Gaussian
/// window, the sum divided by the windows' sum, or the one tile of an image of one tile as it is.
pub(crate) enum Fusion {
    Single(Vec<f32>),
    Windowed(Windowed),
}

pub(crate) struct Windowed {
    width: usize,
    fused: Vec<f32>,
    /// The windows' sum at each pixel.
    weight: Vec<f32>,
    /// The tile's window, row-major.
    window: Vec<f32>,
}

impl Fusion {
    pub(crate) fn new(size: Size) -> Self {
        let fused = vec![0.0; size.width * size.height * IMAGE_CHANNELS];
        if size.width <= TILE && size.height <= TILE {
            return Fusion::Single(fused);
        }
        Fusion::Windowed(Windowed {
            width: size.width,
            fused,
            weight: vec![0.0; size.width * size.height],
            window: window(),
        })
    }

    /// Adds `restored`, a `TILE × TILE` tile whose top left corner is `(top, left)`.
    pub(crate) fn add(&mut self, corner: (usize, usize), restored: &[f32]) {
        match self {
            Fusion::Single(fused) => fused.copy_from_slice(restored),
            Fusion::Windowed(windowed) => windowed.add(corner, restored),
        }
    }

    /// The fused image, pixel-major RGB; the tiles are spent.
    pub(crate) fn finish(&mut self) -> Vec<f32> {
        match self {
            Fusion::Single(fused) => std::mem::take(fused),
            Fusion::Windowed(windowed) => windowed.finish(),
        }
    }
}

impl Windowed {
    fn add(&mut self, (top, left): (usize, usize), restored: &[f32]) {
        for row in 0..TILE {
            for column in 0..TILE {
                let window = self.window[row * TILE + column];
                let pixel = (top + row) * self.width + left + column;
                self.weight[pixel] += window;
                let source = &restored[(row * TILE + column) * IMAGE_CHANNELS..];
                let target = &mut self.fused[pixel * IMAGE_CHANNELS..];
                for (sum, value) in target.iter_mut().zip(source).take(IMAGE_CHANNELS) {
                    *sum += value * window;
                }
            }
        }
    }

    fn finish(&mut self) -> Vec<f32> {
        let mut fused = std::mem::take(&mut self.fused);
        let pixels = fused.as_chunks_mut::<IMAGE_CHANNELS>().0.iter_mut();
        for (pixel, weight) in pixels.zip(&self.weight) {
            for value in pixel {
                *value /= weight;
            }
        }
        fused
    }
}

/// The tiles' fusion weights: `exp(−(i − TILE/2)²/TILE²/0.02) · exp(−(j − (TILE−1)/2)²/TILE²/0.02)`
/// at row `i`, column `j`, in f32.
fn window() -> Vec<f32> {
    let gaussian = |center: f32| -> Vec<f32> {
        (0..TILE)
            .map(|index| {
                let offset = index as f32 - center;
                (-(offset * offset) / (TILE * TILE) as f32 / WINDOW_VARIANCE).exp()
            })
            .collect()
    };
    let rows = gaussian(TILE as f32 / 2.0);
    let columns = gaussian((TILE - 1) as f32 / 2.0);
    rows.iter()
        .flat_map(|row| columns.iter().map(move |column| row * column))
        .collect()
}

/// The fused image (`work.padded`) cropped to the work size, as `(y + 1) / 2`, resized back to
/// `out` by the antialiased bicubic when it differs.
pub fn finish(fused: &[f32], work: &Work, out: Size) -> Vec<f32> {
    let Work { size, padded } = *work;
    let mut image = Vec::with_capacity(size.width * size.height * IMAGE_CHANNELS);
    for row in fused
        .chunks_exact(padded.width * IMAGE_CHANNELS)
        .take(size.height)
    {
        image.extend(
            row[..size.width * IMAGE_CHANNELS]
                .iter()
                .map(|value| (value + 1.0) / 2.0),
        );
    }
    if size == out {
        return image;
    }
    resize_bicubic_antialiased(&image, size, out, IMAGE_CHANNELS)
}

/// `wavelet_color_fix`: the content's high frequencies on the reference's low ones (five à-trous
/// binomial blurs, edges repeated).
pub fn color_fix(content: &[f32], reference: &[f32], size: Size) -> Vec<f32> {
    let (high, _) = split(content, size);
    let (_, low) = split(reference, size);
    high.iter()
        .zip(&low)
        .map(|(high, low)| high + low)
        .collect()
}

/// `(high, low)` frequency parts of an image; they sum back to it.
fn split(image: &[f32], size: Size) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0.0; image.len()];
    let mut low = image.to_vec();
    for dilation in DILATIONS {
        let blurred = blur(&low, size, dilation);
        for ((high, low), blurred) in high.iter_mut().zip(&low).zip(&blurred) {
            *high += low - blurred;
        }
        low = blurred;
    }
    (high, low)
}

/// `image` blurred by `KERNEL` with taps `dilation` apart, each channel alone, the nearest edge
/// pixel standing in past the edges.
fn blur(image: &[f32], size: Size, dilation: usize) -> Vec<f32> {
    let Size { width, height } = size;
    let near = |position: usize, offset: isize, length: usize| {
        let moved = position.saturating_add_signed(offset * dilation as isize);
        moved.min(length - 1)
    };
    let mut out = Vec::with_capacity(image.len());
    for row in 0..height {
        let rows = [-1, 0, 1].map(|offset| near(row, offset, height));
        for column in 0..width {
            let columns = [-1, 0, 1].map(|offset| near(column, offset, width));
            let taps = (rows, columns);
            out.extend((0..IMAGE_CHANNELS).map(|channel| tap_sum(image, width, taps, channel)));
        }
    }
    out
}

/// The sum of `KERNEL` times the `channel` values at the taps' rows and columns, row by row.
fn tap_sum(
    image: &[f32],
    width: usize,
    (rows, columns): ([usize; 3], [usize; 3]),
    channel: usize,
) -> f32 {
    let mut sum = 0.0;
    for (kernel, row) in KERNEL.iter().zip(rows) {
        for (weight, column) in kernel.iter().zip(columns) {
            sum += weight * image[(row * width + column) * IMAGE_CHANNELS + channel];
        }
    }
    sum
}

/// `round(255 · clamp(v, 0, 1))`, half to even, RGB bytes.
pub fn bytes(image: &[f32]) -> Vec<u8> {
    image
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * f32::from(u8::MAX)).round_ties_even() as u8)
        .collect()
}
