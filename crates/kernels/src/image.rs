//! Image resampling as vision models' reference preprocessing does it: Pillow's bicubic resize of
//! 8-bit images, and PyTorch's bilinear interpolation of a grid of f32 vectors (a learned
//! position table resampled to an image's patch grid).

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
/// The support of Pillow's bicubic filter, in source pixels at scale 1.
const BICUBIC_SUPPORT: f64 = 2.0;
/// Pillow's bicubic filter's `a`.
const BICUBIC_A: f64 = -0.5;

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

    /// The image resized as Pillow's `Image.resize(size, Image.BICUBIC)` resizes an 8-bit image:
    /// a horizontal then a vertical pass, each antialiased when shrinking, with coefficients in
    /// fixed point and every value rounded and clamped to 8 bits after its pass.
    pub fn resize_bicubic(&self, size: Size) -> Pixels {
        let horizontal = Weights::new(self.size.width, size.width);
        let vertical = Weights::new(self.size.height, size.height);
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
    fn new(source: usize, target: usize) -> Self {
        let scale = source as f64 / target as f64;
        let filter_scale = scale.max(1.0);
        let support = BICUBIC_SUPPORT * filter_scale;
        let (mut first, mut weights) = (Vec::with_capacity(target), Vec::with_capacity(target));
        for out in 0..target {
            let center = (out as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(source);
            let taps: Vec<f64> = (start..end)
                .map(|x| bicubic((x as f64 - center + 0.5) / filter_scale))
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

/// Pillow's bicubic kernel (Keys, `a = −0.5`).
fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * BICUBIC_A
    } else {
        0.0
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
    use super::{Pixels, Size};

    #[test]
    fn bicubic_resize_matches_pillow_up_and_down() {
        let image = Pixels {
            size: Size {
                width: 4,
                height: 3,
            },
            channels: 3,
            values: vec![
                0, 255, 10, 200, 30, 40, 90, 90, 90, 255, 0, 255, 10, 20, 30, 250, 250, 0, 5, 128,
                64, 33, 66, 99, 255, 255, 255, 0, 0, 0, 128, 64, 32, 1, 2, 3,
            ],
        };
        // Pillow 12.2.0: `Image.fromarray(image, "RGB").resize(size, Image.BICUBIC)`.
        let cases: [(Size, Vec<u8>); 2] = [
            (
                Size {
                    width: 6,
                    height: 4,
                },
                vec![
                    0, 255, 7, 99, 147, 21, 194, 13, 48, 99, 82, 76, 180, 44, 185, 255, 0, 255, 0,
                    81, 7, 124, 139, 7, 236, 183, 21, 53, 132, 64, 63, 76, 126, 120, 43, 165, 95,
                    81, 117, 136, 133, 57, 150, 172, 1, 62, 119, 42, 17, 71, 54, 7, 43, 53, 255,
                    255, 255, 126, 131, 139, 0, 0, 0, 121, 52, 26, 71, 32, 16, 0, 0, 0,
                ],
            ),
            (
                Size {
                    width: 2,
                    height: 2,
                },
                vec![102, 141, 18, 135, 68, 135, 128, 129, 85, 44, 58, 27],
            ),
        ];
        for (size, want) in cases {
            assert_eq!(image.resize_bicubic(size).values, want, "{size:?}");
        }
    }
}
