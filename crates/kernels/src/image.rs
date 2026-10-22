//! Image resampling as vision models' reference preprocessing does it: Pillow's bilinear and
//! bicubic resizes of 8-bit images, and PyTorch's bilinear and bicubic interpolation of a grid of
//! f32 vectors (a learned position table resampled to an image's patch grid).

/// Width and height, in pixels or grid cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: usize,
    pub height: usize,
}

/// An 8-bit image, row-major, its `channels` interleaved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pixels {
    pub size: Size,
    pub channels: usize,
    pub values: Vec<u8>,
}

const RGB: usize = 3;
const RGBA: usize = 4;
/// Pillow's fixed-point fraction bits for 8-bit images (`PRECISION_BITS` in `Resample.c`).
const PRECISION_BITS: u32 = 32 - 8 - 2;
/// Pillow's bicubic filter's `a`.
const BICUBIC_A: f64 = -0.5;
/// PyTorch's bicubic interpolation's `a` (`upsample_bicubic2d`).
const TORCH_BICUBIC_A: f32 = -0.75;

/// A Pillow resampling filter (`Image.BILINEAR`, `Image.BICUBIC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Bilinear,
    Bicubic,
}

impl Filter {
    /// Its support, in source pixels at scale 1.
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => (1.0 - x).max(0.0),
            Filter::Bicubic if x < 1.0 => ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1.0,
            Filter::Bicubic if x < 2.0 => (((x - 5.0) * x + 8.0) * x - 4.0) * BICUBIC_A,
            Filter::Bicubic => 0.0,
        }
    }
}

impl Pixels {
    /// The image as RGB, as Pillow's `convert("RGB")` makes it: an RGBA image's alpha dropped.
    ///
    /// # Panics
    /// When the image is neither RGB nor RGBA.
    pub fn rgb(&self) -> Pixels {
        assert!([RGB, RGBA].contains(&self.channels), "an RGB or RGBA image");
        let pixels = self.values.chunks_exact(self.channels);
        Pixels {
            size: self.size,
            channels: RGB,
            values: pixels.flat_map(|pixel| &pixel[..RGB]).copied().collect(),
        }
    }

    /// The image resized as Pillow's `Image.resize(size, filter)` resizes an 8-bit image: a
    /// horizontal then a vertical pass, each antialiased when shrinking, with coefficients in
    /// fixed point and every value rounded and clamped to 8 bits after its pass.
    pub fn resize(&self, size: Size, filter: Filter) -> Pixels {
        let horizontal = Weights::new(self.size.width, size.width, filter);
        let vertical = Weights::new(self.size.height, size.height, filter);
        let channels = self.channels;
        let source = &self.values;
        let wide: Vec<u8> = (0..self.size.height * size.width * channels)
            .map(|at| {
                let (row, x, channel) = (
                    at / channels / size.width,
                    at / channels % size.width,
                    at % channels,
                );
                let line = row * self.size.width;
                horizontal.value(x, |column| source[(line + column) * channels + channel])
            })
            .collect();
        let stride = size.width * channels;
        let values = (0..size.height * stride)
            .map(|at| vertical.value(at / stride, |row| wide[row * stride + at % stride]))
            .collect();
        Pixels {
            size,
            channels,
            values,
        }
    }
}

/// One pass's filter: for each output pixel, its first source pixel and its fixed-point weights.
struct Weights {
    first: Vec<usize>,
    weights: Vec<Vec<i32>>,
}

impl Weights {
    /// `precompute_coeffs` and `normalize_coeffs_8bpc` of Pillow's `Resample.c`.
    fn new(source: usize, target: usize, filter: Filter) -> Self {
        let scale = source as f64 / target as f64;
        let filter_scale = scale.max(1.0);
        let support = filter.support() * filter_scale;
        let (mut first, mut weights) = (Vec::with_capacity(target), Vec::with_capacity(target));
        for out in 0..target {
            let center = (out as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(source);
            let taps: Vec<f64> = (start..end)
                .map(|x| filter.weight((x as f64 - center + 0.5) / filter_scale))
                .collect();
            let total: f64 = taps.iter().sum();
            let fixed = taps.iter().map(|tap| {
                let normalized = if total == 0.0 { *tap } else { tap / total };
                let scaled = normalized * f64::from(1u32 << PRECISION_BITS);
                (scaled + 0.5f64.copysign(scaled)) as i32
            });
            first.push(start);
            weights.push(fixed.collect());
        }
        Self { first, weights }
    }

    /// Output pixel `out` from the source values `sample(x)`, as Pillow's 8-bit passes round
    /// and clamp it.
    fn value(&self, out: usize, sample: impl Fn(usize) -> u8) -> u8 {
        let first = self.first[out];
        let taps = self.weights[out].iter().enumerate();
        let sum = taps.fold(1 << (PRECISION_BITS - 1), |sum: i32, (tap, weight)| {
            sum + i32::from(sample(first + tap)) * weight
        });
        (sum >> PRECISION_BITS).clamp(0, i32::from(u8::MAX)) as u8
    }
}

/// `grid` (`from.height × from.width` cells of `width` values each, row-major) resampled to
/// `to` cells as `F.interpolate(mode="bilinear", align_corners=False)` does it, in f32.
pub fn resize_bilinear(grid: &[f32], from: Size, to: Size, width: usize) -> Vec<f32> {
    let rows = Axis::new(from.height, to.height);
    let columns = Axis::new(from.width, to.width);
    let cell = |row: usize, column: usize| &grid[(row * from.width + column) * width..][..width];
    let mut out = Vec::with_capacity(to.height * to.width * width);
    for (top, bottom, down) in &rows.taps {
        for (left, right, across) in &columns.taps {
            let (top_left, top_right) = (cell(*top, *left), cell(*top, *right));
            let (bottom_left, bottom_right) = (cell(*bottom, *left), cell(*bottom, *right));
            out.extend((0..width).map(|i| {
                let upper = (1.0 - across) * top_left[i] + across * top_right[i];
                let lower = (1.0 - across) * bottom_left[i] + across * bottom_right[i];
                (1.0 - down) * upper + down * lower
            }));
        }
    }
    out
}

/// `grid` resampled to `to` cells as `F.interpolate(mode="bicubic", align_corners=False)` does
/// it, in f32: four taps per axis (Keys, `a = −0.75`), the edge cells repeated past the border,
/// and no antialiasing when shrinking.
pub fn resize_bicubic(grid: &[f32], from: Size, to: Size, width: usize) -> Vec<f32> {
    let rows = CubicAxis::new(from.height, to.height);
    let columns = CubicAxis::new(from.width, to.width);
    let mut out = vec![0.0; to.height * to.width * width];
    let mut line = vec![0.0; width];
    for (y, (row_taps, row_weights)) in rows.taps.iter().enumerate() {
        for (x, (column_taps, column_weights)) in columns.taps.iter().enumerate() {
            let cell = &mut out[(y * to.width + x) * width..][..width];
            for (row, row_weight) in row_taps.iter().zip(row_weights) {
                line.fill(0.0);
                for (column, column_weight) in column_taps.iter().zip(column_weights) {
                    let source = &grid[(row * from.width + column) * width..][..width];
                    add_scaled(&mut line, source, *column_weight);
                }
                add_scaled(cell, &line, *row_weight);
            }
        }
    }
    out
}

/// `out += weight · values`, one value at a time.
fn add_scaled(out: &mut [f32], values: &[f32], weight: f32) {
    for (value, sample) in out.iter_mut().zip(values) {
        *value += sample * weight;
    }
}

/// One axis of a bicubic resampling: per output cell, its four source cells and their weights.
struct CubicAxis {
    taps: Vec<([usize; 4], [f32; 4])>,
}

impl CubicAxis {
    fn new(source: usize, target: usize) -> Self {
        let scale = source as f32 / target as f32;
        let last = source as isize - 1;
        let taps = (0..target).map(|out| {
            let position = (out as f32 + 0.5) * scale - 0.5;
            let first = position.floor();
            let offset = position - first;
            let cells =
                std::array::from_fn(|tap| (first as isize + tap as isize - 1).clamp(0, last));
            (cells.map(|cell| cell as usize), cubic_weights(offset))
        });
        Self {
            taps: taps.collect(),
        }
    }
}

/// The weights of the four taps around a position `offset` past the second, as PyTorch's
/// `get_cubic_upsample_coefficients` computes them.
fn cubic_weights(offset: f32) -> [f32; 4] {
    let slope = TORCH_BICUBIC_A;
    let near = |x: f32| ((slope + 2.0) * x - (slope + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((slope * x - 5.0 * slope) * x + 8.0 * slope) * x - 4.0 * slope;
    let rest = 1.0 - offset;
    [far(offset + 1.0), near(offset), near(rest), far(rest + 1.0)]
}

/// One axis of a bilinear resampling: per output cell, the two source cells and the weight of
/// the second.
struct Axis {
    taps: Vec<(usize, usize, f32)>,
}

impl Axis {
    fn new(source: usize, target: usize) -> Self {
        let scale = source as f32 / target as f32;
        let taps = (0..target).map(|out| {
            let position = ((out as f32 + 0.5) * scale - 0.5).max(0.0);
            let first = position as usize;
            let second = (first + 1).min(source - 1);
            (first, second, position - first as f32)
        });
        Self {
            taps: taps.collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Filter, Pixels, Size, resize_bicubic};

    fn image() -> Pixels {
        Pixels {
            size: Size {
                width: 4,
                height: 3,
            },
            channels: 3,
            values: vec![
                0, 255, 10, 200, 30, 40, 90, 90, 90, 255, 0, 255, 10, 20, 30, 250, 250, 0, 5, 128,
                64, 33, 66, 99, 255, 255, 255, 0, 0, 0, 128, 64, 32, 1, 2, 3,
            ],
        }
    }

    // Pillow 12.2.0: `Image.fromarray(image, "RGB").resize(size, filter)`, to 6×4 and to 2×2.
    #[test]
    fn bicubic_resize_matches_pillow_up_and_down() {
        let up = [
            0, 255, 7, 99, 147, 21, 194, 13, 48, 99, 82, 76, 180, 44, 185, 255, 0, 255, 0, 81, 7,
            124, 139, 7, 236, 183, 21, 53, 132, 64, 63, 76, 126, 120, 43, 165, 95, 81, 117, 136,
            133, 57, 150, 172, 1, 62, 119, 42, 17, 71, 54, 7, 43, 53, 255, 255, 255, 126, 131, 139,
            0, 0, 0, 121, 52, 26, 71, 32, 16, 0, 0, 0,
        ];
        let down = [102, 141, 18, 135, 68, 135, 128, 129, 85, 44, 58, 27];
        check_pillow(Filter::Bicubic, &up, &down);
    }

    #[test]
    fn bilinear_resize_matches_pillow_up_and_down() {
        let up = [
            0, 255, 10, 100, 143, 25, 182, 40, 48, 108, 80, 82, 173, 45, 173, 255, 0, 255, 6, 108,
            23, 119, 138, 19, 199, 159, 25, 69, 123, 64, 77, 78, 116, 116, 41, 158, 102, 108, 114,
            129, 132, 57, 139, 148, 9, 69, 112, 43, 36, 73, 58, 21, 42, 63, 255, 255, 255, 128,
            128, 128, 21, 11, 5, 107, 53, 27, 65, 33, 18, 1, 2, 3,
        ];
        let down = [104, 135, 30, 130, 72, 123, 122, 124, 80, 54, 62, 36];
        check_pillow(Filter::Bilinear, &up, &down);
    }

    fn check_pillow(filter: Filter, up: &[u8], down: &[u8]) {
        let sizes = [(6, 4, up), (2, 2, down)];
        for (width, height, want) in sizes {
            let size = Size { width, height };
            assert_eq!(image().resize(size, filter).values, want, "{size:?}");
        }
    }

    #[test]
    fn bicubic_grid_resize_matches_pytorch_up_and_down() {
        let grid = [
            -1.0, 0.0, -0.5, 0.5, 0.0, 1.0, 0.5, 0.25, 1.0, 0.75, 1.5, 0.0,
        ];
        let from = Size {
            width: 3,
            height: 2,
        };
        // PyTorch 2.14.1: `F.interpolate(grid, size, mode="bicubic", align_corners=False)`.
        let up = [
            -1.178_208_7,
            -0.069_701_43,
            -0.972_208_2,
            0.128_486_37,
            -0.630_208_5,
            0.478_298_63,
            -0.288_208_54,
            0.886_704_9,
            -0.082_208_46,
            1.145_222_5,
            -0.298_000_07,
            0.077_000_01,
            -0.091_999_74,
            0.328_000_13,
            0.25,
            0.625,
            0.592,
            0.584_499_95,
            0.798_000_2,
            0.488_000_1,
            0.582_208_16,
            0.223_701_37,
            0.788_208_4,
            0.527_513_8,
            1.130_208,
            0.771_701_34,
            1.472_208,
            0.282_295_1,
            1.678_208_4,
            -0.169_222_09,
        ];
        let down = [-0.154_296_88, 0.242_675_78, 0.654_296_9, 0.545_898_44];
        for (width, height, want) in [(5, 3, up.as_slice()), (2, 1, down.as_slice())] {
            let got = resize_bicubic(&grid, from, Size { width, height }, 2);
            assert_eq!(got.len(), want.len());
            for (got, want) in got.iter().zip(want) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "{width}×{height}: {got} vs {want}"
                );
            }
        }
    }
}
