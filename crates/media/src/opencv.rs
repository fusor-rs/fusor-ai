//! OpenCV's operations on 8-bit single-channel images as OpenCV 5.0's arm64 builds compute them
//! (`imgproc/src/resize.cpp`, `warp_kernels.simd.hpp`, `color_rgb.simd.hpp`, and the carotene
//! layer's `resize.cpp` for enlarging): the conversion to grayscale, `resize` with `INTER_AREA`
//! and `INTER_LINEAR`, a rotation about the centre with `warpAffine`, and quarter turns.

use crate::image::{Pixels, Size};

/// `cvtColor`'s RGB → gray weights in 14-bit fixed point, and its rounding.
const GRAY_RED: u32 = 4899;
const GRAY_GREEN: u32 = 9617;
const GRAY_BLUE: u32 = 1868;
const GRAY_SHIFT: u32 = 14;
/// `resize`'s linear coefficients' fixed point (`INTER_RESIZE_COEF_BITS`).
const COEFFICIENT_BITS: u32 = 11;
const COEFFICIENT_SCALE: f32 = (1 << COEFFICIENT_BITS) as f32;
/// The vertical pass's shifts: of each horizontal sum, of each weighed one (to its high half),
/// and of their total.
const SUM_SHIFT: u32 = 4;
const HIGH_HALF: u32 = 16;
const ROUND_SHIFT: u32 = 2;
/// Carotene's linear resize takes images at least this wide (and this wide and high after);
/// its weights are 7-bit.
const CAROTENE_MIN: usize = 8;
const CAROTENE_BITS: u32 = 7;
const CAROTENE_ONE: u16 = 1 << CAROTENE_BITS;
/// Source pixels of a halved image's pixel.
const QUARTER: usize = 4;
/// The share of a source pixel `computeResizeAreaTab` ignores at a cell's edges.
const AREA_EDGE: f64 = 1e-3;

/// What a rotation reads past the image's edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Border {
    /// This value (`BORDER_CONSTANT`).
    Constant(u8),
    /// The nearest edge pixel (`BORDER_REPLICATE`).
    Replicate,
}

impl Pixels {
    /// The image in one channel, as `cvtColor(…, COLOR_RGB2GRAY)` makes it (alpha dropped); a
    /// grayscale image as it is.
    pub fn gray(&self) -> Pixels {
        if self.channels == 1 {
            return self.clone();
        }
        let values = self.values.chunks_exact(self.channels).map(|pixel| {
            let [red, green, blue] = [0, 1, 2].map(|channel| u32::from(pixel[channel]));
            let sum = red * GRAY_RED + green * GRAY_GREEN + blue * GRAY_BLUE;
            ((sum + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT) as u8
        });
        gray(self.size, values.collect())
    }

    fn at(&self, x: usize, y: usize) -> u8 {
        self.values[y * self.size.width + x]
    }
}

fn gray(size: Size, values: Vec<u8>) -> Pixels {
    Pixels {
        size,
        channels: 1,
        values,
    }
}

/// A grayscale `image` resized to `size` with `INTER_AREA`: each pixel the mean of the source
/// pixels its cell covers, partly covered ones weighed by their share (each row's weighed with
/// fused multiply-adds); enlarging falls back to `INTER_LINEAR` with the area's own
/// coefficients.
pub fn resize_area(image: &Pixels, size: Size) -> Pixels {
    let scale_x = image.size.width as f64 / size.width as f64;
    let scale_y = image.size.height as f64 / size.height as f64;
    if scale_x < 1.0 || scale_y < 1.0 {
        return resize_by_taps(image, size, Taps::Area);
    }
    if image.size.width.is_multiple_of(size.width) && image.size.height.is_multiple_of(size.height)
    {
        return resize_blocks(image, size);
    }
    let columns = area_taps(image.size.width, size.width, scale_x);
    let rows = area_taps(image.size.height, size.height, scale_y);
    let mut out = Vec::with_capacity(size.width * size.height);
    let mut line = vec![0.0f32; size.width];
    let mut sum = vec![0.0f32; size.width];
    for taps in rows {
        sum.fill(0.0);
        for (source_row, beta) in taps {
            line.fill(0.0);
            for (column, taps) in columns.iter().enumerate() {
                for &(source, alpha) in taps {
                    let level = f32::from(image.at(source, source_row));
                    line[column] = fused(level, alpha, line[column]);
                }
            }
            for (sum, value) in sum.iter_mut().zip(&line) {
                *sum += beta * value;
            }
        }
        out.extend(sum.iter().map(|&value| saturate(value)));
    }
    gray(size, out)
}

/// `resizeAreaFast_`: each pixel the mean of its whole block, a 2 × 2 one rounded half up as the
/// SIMD kernel does it, larger ones in f32 rounded half to even.
fn resize_blocks(image: &Pixels, size: Size) -> Pixels {
    let (block_x, block_y) = (
        image.size.width / size.width,
        image.size.height / size.height,
    );
    let area = block_x * block_y;
    let mut out = Vec::with_capacity(size.width * size.height);
    for y in 0..size.height {
        for x in 0..size.width {
            let pixels = (0..block_y).flat_map(|dy| (0..block_x).map(move |dx| (dx, dy)));
            let sum: u32 = pixels
                .map(|(dx, dy)| u32::from(image.at(x * block_x + dx, y * block_y + dy)))
                .sum();
            out.push(match area {
                QUARTER => ((sum + 2) >> 2) as u8,
                _ => saturate(sum as f32 * (1.0 / area as f32)),
            });
        }
    }
    gray(size, out)
}

/// Per output pixel of an axis shrunk by `scale`, its source pixels and their weights
/// (`computeResizeAreaTab`).
fn area_taps(source: usize, target: usize, scale: f64) -> Vec<Vec<(usize, f32)>> {
    (0..target)
        .map(|out| {
            let start = out as f64 * scale;
            let end = start + scale;
            let cell = scale.min(source as f64 - start);
            let last = (end.floor() as usize).min(source - 1);
            let first = (start.ceil() as usize).min(last);
            let mut taps = Vec::new();
            if first as f64 - start > AREA_EDGE {
                taps.push((first - 1, ((first as f64 - start) / cell) as f32));
            }
            taps.extend((first..last).map(|index| (index, (1.0 / cell) as f32)));
            if end - last as f64 > AREA_EDGE {
                let share = (end - last as f64).min(1.0).min(cell);
                taps.push((last, (share / cell) as f32));
            }
            taps
        })
        .collect()
}

/// A grayscale `image` resized to `size` with `INTER_LINEAR`. Enlarging, as OpenCV's arm64
/// builds do it through their carotene layer (`resizeLinearOpenCV`): rows blended first, then
/// columns, each blend with 7-bit weights rounded to 8 bits. Otherwise as its generic code: 11-bit
/// weights, and halving both sides exactly is `INTER_AREA`.
pub fn resize_linear(image: &Pixels, size: Size) -> Pixels {
    let ratio = |source: usize, target: usize| (1.0 / (target as f64 / source as f64)) as f32;
    let (width_ratio, height_ratio) = (
        ratio(image.size.width, size.width),
        ratio(image.size.height, size.height),
    );
    let carotene = image.size.width >= CAROTENE_MIN
        && size.width >= CAROTENE_MIN
        && size.height >= CAROTENE_MIN;
    if carotene && width_ratio <= 1.0 && height_ratio <= 1.0 {
        let rows = blend_rows(image, size.height, height_ratio);
        return blend_columns(&rows, size.width, width_ratio);
    }
    if image.size.width == 2 * size.width && image.size.height == 2 * size.height {
        return resize_area(image, size);
    }
    resize_by_taps(image, size, Taps::Linear)
}

/// The source pixel before position `index` scaled by `ratio` (carotene's sampling, `index ·
/// ratio + (ratio − 1)/2` with a fused multiply-add), and the 7-bit weight of that pixel.
fn carotene_tap(index: usize, ratio: f32) -> (i64, u16) {
    let offset = 0.5 * ratio - 0.5;
    let position = fused(index as f32, ratio, offset);
    let truncated = position.trunc();
    let after = if truncated < position {
        truncated + 1.0
    } else {
        truncated
    };
    let weight = ((after - position) * CAROTENE_ONE as f32).max(0.0) as u16;
    (after as i64 - 1, weight)
}

/// `(first · weight + second · (ONE − weight) + ONE/2) >> 7`.
fn carotene_blend(first: u8, second: u8, weight: u16) -> u8 {
    let sum = u16::from(first) * weight + u16::from(second) * (CAROTENE_ONE - weight);
    ((sum + CAROTENE_ONE / 2) >> CAROTENE_BITS) as u8
}

/// The image's rows resampled to `height`.
fn blend_rows(image: &Pixels, height: usize, ratio: f32) -> Pixels {
    let width = image.size.width;
    let last = image.size.height as i64 - 1;
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let (before, weight) = carotene_tap(row, ratio);
        let (first, second) = (
            before.clamp(0, last) as usize,
            (before + 1).min(last) as usize,
        );
        let upper = &image.values[first * width..][..width];
        let lower = &image.values[second * width..][..width];
        out.extend(
            upper
                .iter()
                .zip(lower)
                .map(|(&upper, &lower)| carotene_blend(upper, lower, weight)),
        );
    }
    gray(Size { width, height }, out)
}

/// The image's columns resampled to `width`.
fn blend_columns(image: &Pixels, width: usize, ratio: f32) -> Pixels {
    let last = image.size.width as i64 - 1;
    let taps: Vec<(usize, usize, u16)> = (0..width)
        .map(|column| {
            let (before, weight) = carotene_tap(column, ratio);
            (
                before.clamp(0, last) as usize,
                (before + 1).min(last) as usize,
                weight,
            )
        })
        .collect();
    let mut out = Vec::with_capacity(width * image.size.height);
    for row in image.values.chunks_exact(image.size.width) {
        out.extend(
            taps.iter()
                .map(|&(first, second, weight)| carotene_blend(row[first], row[second], weight)),
        );
    }
    gray(
        Size {
            width,
            height: image.size.height,
        },
        out,
    )
}

/// Which coefficients `resize_by_taps` uses.
#[derive(Clone, Copy)]
enum Taps {
    Linear,
    Area,
}

impl Taps {
    /// The first source pixel and the second's weight for output pixel `out` of an axis scaled
    /// by `scale` (source over target).
    fn tap(self, out: usize, scale: f64, source: usize) -> (usize, f32) {
        let (first, fraction) = match self {
            Taps::Linear => {
                let position = ((out as f64 + 0.5) * scale - 0.5) as f32;
                let first = position.floor();
                (first as isize, position - first)
            }
            Taps::Area => {
                let first = (out as f64 * scale).floor();
                let fraction = ((out + 1) as f64 - (first + 1.0) / scale) as f32;
                let fraction = if fraction <= 0.0 {
                    0.0
                } else {
                    fraction - fraction.floor()
                };
                (first as isize, fraction)
            }
        };
        match usize::try_from(first) {
            Err(_) => (0, 0.0),
            Ok(first) if first + 1 >= source => (source - 1, 0.0),
            Ok(first) => (first, fraction),
        }
    }
}

/// `resize`'s two-tap path for 8-bit images: each source row's horizontal sums in 11-bit fixed
/// point, then the vertical pass of its 8-bit specialization (`VResizeLinear`): each sum
/// shifted down `SUM_SHIFT` bits and weighed keeping the product's high 16 bits, the two added
/// and rounded off their last `ROUND_SHIFT` bits.
fn resize_by_taps(image: &Pixels, size: Size, taps: Taps) -> Pixels {
    let scale_x = image.size.width as f64 / size.width as f64;
    let scale_y = image.size.height as f64 / size.height as f64;
    let fixed = |fraction: f32| {
        let second = (fraction * COEFFICIENT_SCALE).round_ties_even() as i32;
        let first = ((1.0 - fraction) * COEFFICIENT_SCALE).round_ties_even() as i32;
        [first, second]
    };
    let columns: Vec<(usize, [i32; 2])> = (0..size.width)
        .map(|x| {
            let (first, fraction) = taps.tap(x, scale_x, image.size.width);
            (first, fixed(fraction))
        })
        .collect();
    let mut sums = Vec::with_capacity(image.size.height * size.width);
    for row in 0..image.size.height {
        sums.extend(columns.iter().map(|&(first, [left, right])| {
            let next = (first + 1).min(image.size.width - 1);
            i32::from(image.at(first, row)) * left + i32::from(image.at(next, row)) * right
        }));
    }
    let weighed = |sum: i32, weight: i32| (weight * (sum >> SUM_SHIFT)) >> HIGH_HALF;
    let mut out = Vec::with_capacity(size.width * size.height);
    for y in 0..size.height {
        let (first, fraction) = taps.tap(y, scale_y, image.size.height);
        let [top, bottom] = fixed(fraction);
        let upper = &sums[first * size.width..][..size.width];
        let next = (first + 1).min(image.size.height - 1);
        let lower = &sums[next * size.width..][..size.width];
        out.extend(upper.iter().zip(lower).map(|(&upper, &lower)| {
            let sum = weighed(upper, top) + weighed(lower, bottom);
            ((sum + (1 << (ROUND_SHIFT - 1))) >> ROUND_SHIFT) as u8
        }));
    }
    gray(size, out)
}

/// A grayscale `image` rotated `degrees` counter-clockwise about its centre `(width/2,
/// height/2)`, the same size: `warpAffine` of `getRotationMatrix2D(centre, degrees, 1)` with
/// bilinear interpolation, in f32 with fused multiply-adds as OpenCV's vector code does it.
pub fn rotate(image: &Pixels, degrees: f64, border: Border) -> Pixels {
    let Size { width, height } = image.size;
    let (center_x, center_y) = (width as f64 / 2.0, height as f64 / 2.0);
    let (cos, sin) = (degrees.to_radians().cos(), degrees.to_radians().sin());
    let forward = [
        cos,
        sin,
        (1.0 - cos) * center_x - sin * center_y,
        -sin,
        cos,
        sin * center_x + (1.0 - cos) * center_y,
    ];
    let inverse = invert_affine(forward).map(|value| value as f32);
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let row_x = fused(y as f32, inverse[1], inverse[2]);
        let row_y = fused(y as f32, inverse[4], inverse[5]);
        for x in 0..width {
            let source_x = fused(inverse[0], x as f32, row_x);
            let source_y = fused(inverse[3], x as f32, row_y);
            out.push(bilinear(image, source_x, source_y, border));
        }
    }
    gray(image.size, out)
}

/// `factor · other + addend` rounded once, as a fused multiply-add: the product of two f32 is
/// exact in f64.
fn fused(factor: f32, other: f32, addend: f32) -> f32 {
    (f64::from(factor) * f64::from(other) + f64::from(addend)) as f32
}

/// `invertAffineTransform` of a 2 × 3 matrix, row-major.
fn invert_affine(matrix: [f64; 6]) -> [f64; 6] {
    let determinant = matrix[0] * matrix[4] - matrix[1] * matrix[3];
    let inverse = if determinant == 0.0 {
        0.0
    } else {
        1.0 / determinant
    };
    let (a11, a22) = (matrix[4] * inverse, matrix[0] * inverse);
    let (a12, a21) = (-matrix[1] * inverse, -matrix[3] * inverse);
    let b1 = -a11 * matrix[2] - a12 * matrix[5];
    let b2 = -a21 * matrix[2] - a22 * matrix[5];
    [a11, a12, b1, a21, a22, b2]
}

/// The image at `(x, y)` interpolated between its four neighbours, as `warpAffine` reads it.
fn bilinear(image: &Pixels, x: f32, y: f32, border: Border) -> u8 {
    let Size { width, height } = image.size;
    let (left, top) = (x.floor(), y.floor());
    let (across, down) = (x - left, y - top);
    let (left, top) = (left as i64, top as i64);
    let outside = |at: i64, length: usize| at < -1 || at >= length as i64;
    if let Border::Constant(value) = border
        && (outside(left, width) || outside(top, height))
    {
        return value;
    }
    let pixel = |dx: i64, dy: i64| {
        let (x, y) = (left + dx, top + dy);
        let inside = (0..width as i64).contains(&x) && (0..height as i64).contains(&y);
        match border {
            _ if inside => f32::from(image.at(x as usize, y as usize)),
            Border::Constant(value) => f32::from(value),
            Border::Replicate => {
                let x = x.clamp(0, width as i64 - 1) as usize;
                let y = y.clamp(0, height as i64 - 1) as usize;
                f32::from(image.at(x, y))
            }
        }
    };
    let (top_left, top_right) = (pixel(0, 0), pixel(1, 0));
    let (bottom_left, bottom_right) = (pixel(0, 1), pixel(1, 1));
    let upper = fused(across, top_right - top_left, top_left);
    let lower = fused(across, bottom_right - bottom_left, bottom_left);
    saturate(fused(down, lower - upper, upper))
}

/// A grayscale `image` turned `quarter_turns` quarter turns counter-clockwise (`np.rot90`).
pub fn rotate_quarters(image: &Pixels, quarter_turns: usize) -> Pixels {
    let Size { width, height } = image.size;
    match quarter_turns % 4 {
        0 => image.clone(),
        1 => {
            let values = (0..width)
                .flat_map(|row| (0..height).map(move |column| (width - 1 - row, column)))
                .map(|(x, y)| image.at(x, y));
            gray(
                Size {
                    width: height,
                    height: width,
                },
                values.collect(),
            )
        }
        2 => gray(image.size, image.values.iter().rev().copied().collect()),
        _ => {
            let values = (0..width)
                .flat_map(|row| (0..height).map(move |column| (row, height - 1 - column)))
                .map(|(x, y)| image.at(x, y));
            gray(
                Size {
                    width: height,
                    height: width,
                },
                values.collect(),
            )
        }
    }
}

/// `saturate_cast<uchar>` of a float: rounded half to even, clamped.
fn saturate(value: f32) -> u8 {
    value.round_ties_even().clamp(0.0, 255.0) as u8
}
