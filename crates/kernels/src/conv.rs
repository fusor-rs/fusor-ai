//! Products over many rows (the pixels of images, the tokens of long sequences), split over a
//! team by rows, and 2-D convolutions as such products.
//!
//! An image is `height × width` pixels in row-major order, each a row of `channels` values; a
//! batch is images one after another (`Grid`). A convolution's weights are one
//! `[outputs, kernel² · inputs]` matrix whose columns are tap-major (window row, then column),
//! each tap's inputs contiguous, so an output pixel is that matrix times the input rows of its
//! window laid end to end, zeros past the image's edge (the padding is `kernel / 2`).
//!
//! `Conv2d::forward_upsampled` convolves the input repeated twice along both axes (nearest
//! upsampling) without materializing it: the windows read it in place.
//!
//! `CausalConv3d` convolves a window of consecutive frames (the oldest first, a history before a
//! video's first) and a `kernel × kernel` window of each at stride 1: one
//! `[outputs, window · kernel² · inputs]` matrix, columns frame-major, then tap-major, each tap's
//! inputs contiguous.
//!
//! `Depthwise2d` convolves each channel alone (in f32), by output pixels.
//!
//! `project_rows` and the convolutions give each thread whole rows when there are at least as
//! many rows as outputs, so each thread quantizes only its own rows; otherwise every thread takes
//! every row and a share of the output columns. Rows are gathered and quantized `CHUNK_ROWS` at
//! a time.

use std::ops::Range;

use team::Ctx;

use crate::Error;
use crate::linear::{COLUMN_ALIGN, Linear};
use crate::matmul::QuantizedRows;
use crate::tensor::{Table, Vector};

/// Rows a thread gathers and quantizes at a time: its working set.
const CHUNK_ROWS: usize = 64;
/// How many times `forward_upsampled` repeats each pixel along each axis.
const UPSAMPLING: usize = 2;

/// The layout of pixel-major rows: `images` images of `height × width` pixels, one after another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub images: usize,
    pub height: usize,
    pub width: usize,
}

impl Grid {
    pub fn pixels(&self) -> usize {
        self.images * self.height * self.width
    }

    /// Pixels of one image.
    pub fn area(&self) -> usize {
        self.height * self.width
    }

    /// The grid `Conv2d::forward_upsampled` reads its input as: twice the height and width.
    pub fn doubled(&self) -> Self {
        Self {
            height: UPSAMPLING * self.height,
            width: UPSAMPLING * self.width,
            ..*self
        }
    }
}

/// The dimensions of one convolution: `inputs` channels to `outputs` over a `kernel × kernel`
/// window moved by `stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub inputs: usize,
    pub outputs: usize,
    pub kernel: usize,
    pub stride: usize,
}

impl Shape {
    /// A convolution at stride 1, whose output grid is its input's.
    pub fn same(inputs: usize, outputs: usize, kernel: usize) -> Self {
        Self {
            inputs,
            outputs,
            kernel,
            stride: 1,
        }
    }

    /// A convolution at stride 2, which halves the height and the width (rounding up).
    pub fn halving(inputs: usize, outputs: usize, kernel: usize) -> Self {
        Self {
            stride: 2,
            ..Self::same(inputs, outputs, kernel)
        }
    }

    /// The `[rows, cols]` of the convolution's matrix.
    pub fn matrix(&self) -> [usize; 2] {
        [self.outputs, self.kernel * self.kernel * self.inputs]
    }
}

/// A convolution over a `kernel × kernel` window moved by `stride`.
pub struct Conv2d {
    linear: Linear,
    kernel: usize,
    stride: usize,
}

/// One thread's rows on their way into a product: gathered windows, then their int8 form.
#[derive(Default)]
pub struct Patches {
    gathered: Vec<f32>,
    quant: QuantizedRows,
}

/// A depthwise convolution's channels, `height × width` window and stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthwiseShape {
    pub channels: usize,
    pub window: [usize; 2],
    pub stride: usize,
}

/// A depthwise convolution: each channel by its own `height × width` window moved by `stride`
/// along both axes, zeros past the image's edges (the padding is half the window).
pub struct Depthwise2d {
    /// The weights tap-major (window row, then column), each tap's channels contiguous.
    taps: Vec<f32>,
    bias: Vec<f32>,
    window: [usize; 2],
    stride: usize,
}

impl Depthwise2d {
    /// `weight` is `[channels, height · width]`, each channel's taps row-major.
    ///
    /// # Errors
    /// When the weight's columns are not the window's taps, or the bias not one per channel.
    pub fn new(
        weight: &Table,
        bias: Option<&Vector>,
        window: [usize; 2],
        stride: usize,
    ) -> Result<Self, Error> {
        let matrix = weight.matrix();
        let (channels, taps) = (matrix.rows, matrix.cols);
        let bias = bias.map_or_else(|| vec![0.0; channels], |bias| bias.values().to_vec());
        if taps != window[0] * window[1] || bias.len() != channels || stride == 0 {
            return Err(Error::Format(format!(
                "a depthwise convolution of {channels}×{taps} weights over a {window:?} window"
            )));
        }
        let values = weight.dequantize();
        let taps = (0..taps * channels)
            .map(|at| values[(at % channels) * taps + at / channels])
            .collect();
        Ok(Self {
            taps,
            bias,
            window,
            stride,
        })
    }

    pub fn channels(&self) -> usize {
        self.bias.len()
    }

    pub fn output_grid(&self, grid: Grid) -> Grid {
        let side =
            |length: usize, window: usize| (length + 2 * (window / 2) - window) / self.stride + 1;
        Grid {
            images: grid.images,
            height: side(grid.height, self.window[0]),
            width: side(grid.width, self.window[1]),
        }
    }

    /// `store(pixel, out)`: this thread's share of the output pixels, each pixel's channels into
    /// `out` (the convolution of `input`, pixel-major rows of `grid`, plus the bias).
    pub fn forward(
        &self,
        ctx: &Ctx,
        (input, grid): (&[f32], Grid),
        out: &mut Vec<f32>,
        mut store: impl FnMut(usize, &[f32]),
    ) {
        let channels = self.channels();
        let output = self.output_grid(grid);
        let [height, width] = self.window;
        out.resize(channels, 0.0);
        for pixel in ctx.split(output.pixels(), 1) {
            let image = pixel / output.area();
            let (row, column) = (pixel / output.width % output.height, pixel % output.width);
            out.copy_from_slice(&self.bias);
            for tap in 0..height * width {
                let y = (row * self.stride + tap / width).checked_sub(height / 2);
                let x = (column * self.stride + tap % width).checked_sub(width / 2);
                let Some((y, x)) = y.zip(x).filter(|&(y, x)| y < grid.height && x < grid.width)
                else {
                    continue;
                };
                let at = ((image * grid.height + y) * grid.width + x) * channels;
                let weights = &self.taps[tap * channels..(tap + 1) * channels];
                for ((out, value), weight) in out.iter_mut().zip(&input[at..]).zip(weights) {
                    *out += value * weight;
                }
            }
            store(pixel, out);
        }
    }
}

/// Where the rows of a product come from.
enum Rows<'input> {
    /// Consecutive rows of the matrix's width.
    Dense(&'input [f32]),
    /// The windows of a convolution over `input`, pixel-major rows of `grid` repeated `scale`
    /// times along both axes.
    Windows {
        conv: &'input Conv2d,
        input: &'input [f32],
        grid: Grid,
        scale: usize,
    },
    /// The windows of a causal convolution over `frames`, each one image of `grid`.
    Frames {
        conv: &'input CausalConv3d,
        frames: &'input [&'input [f32]],
        grid: Grid,
    },
}

impl Conv2d {
    /// # Errors
    /// When the weights' columns are not `kernel²` taps of whole inputs.
    pub fn new(linear: Linear, kernel: usize, stride: usize) -> Result<Self, Error> {
        let cols = linear.matrix().cols;
        if kernel == 0 || stride == 0 || !cols.is_multiple_of(kernel * kernel) {
            return Err(Error::Format(format!(
                "a convolution of {cols} columns with a {kernel}-wide window at stride {stride}"
            )));
        }
        Ok(Self {
            linear,
            kernel,
            stride,
        })
    }

    pub fn outputs(&self) -> usize {
        self.linear.matrix().rows
    }

    pub fn inputs(&self) -> usize {
        self.linear.matrix().cols / (self.kernel * self.kernel)
    }

    pub fn output_grid(&self, grid: Grid) -> Grid {
        let padding = self.kernel / 2;
        let side = |length: usize| (length + 2 * padding - self.kernel) / self.stride + 1;
        Grid {
            images: grid.images,
            height: side(grid.height),
            width: side(grid.width),
        }
    }

    /// `store(pixel, channel, value)` of the convolution of `input` (pixel-major rows of
    /// `grid`, `inputs()` values each) plus its bias, for this thread's share of the output.
    pub fn forward(
        &self,
        ctx: &Ctx,
        (input, grid): (&[f32], Grid),
        patches: &mut Patches,
        store: impl FnMut(usize, usize, f32),
    ) {
        let rows = self.output_grid(grid).pixels();
        let source = if self.kernel == 1 && self.stride == 1 {
            Rows::Dense(&input[..rows * self.inputs()])
        } else {
            Rows::Windows {
                conv: self,
                input,
                grid,
                scale: 1,
            }
        };
        project_chunks(ctx, &self.linear, (&source, rows), patches, store);
    }

    /// `forward` over `input` upsampled to twice its height and width, each pixel repeated
    /// (nearest); the output's grid is `output_grid` of that doubled grid.
    pub fn forward_upsampled(
        &self,
        ctx: &Ctx,
        (input, grid): (&[f32], Grid),
        patches: &mut Patches,
        store: impl FnMut(usize, usize, f32),
    ) {
        let source = Rows::Windows {
            conv: self,
            input,
            grid,
            scale: UPSAMPLING,
        };
        let rows = self.output_grid(grid.doubled()).pixels();
        project_chunks(ctx, &self.linear, (&source, rows), patches, store);
    }

    /// The windows of output pixels `pixels`, end to end, into `out`, over `input` (rows of
    /// `grid`) repeated `scale` times along both axes.
    fn gather(
        &self,
        (input, grid, scale): (&[f32], Grid, usize),
        pixels: Range<usize>,
        out: &mut Vec<f32>,
    ) {
        let output = self.output_grid(Grid {
            height: scale * grid.height,
            width: scale * grid.width,
            ..grid
        });
        out.clear();
        for pixel in pixels {
            let image = pixel / output.area();
            let (row, column) = (pixel / output.width % output.height, pixel % output.width);
            let window = (self.kernel, self.stride, self.inputs());
            push_window(out, (input, grid, scale), (image, row, column), window);
        }
    }
}

/// One output pixel's `kernel × kernel` window over `input` (rows of `grid` repeated `scale`
/// times along both axes) laid end to end into `out`, zeros past the edges (the padding is
/// `kernel / 2`).
fn push_window(
    out: &mut Vec<f32>,
    (input, grid, scale): (&[f32], Grid, usize),
    (image, row, column): (usize, usize, usize),
    (kernel, stride, channels): (usize, usize, usize),
) {
    let (height, width) = (scale * grid.height, scale * grid.width);
    let padding = kernel / 2;
    for tap in 0..kernel * kernel {
        let source_row = (row * stride + tap / kernel).checked_sub(padding);
        let source_column = (column * stride + tap % kernel).checked_sub(padding);
        match source_row.zip(source_column) {
            Some((y, x)) if y < height && x < width => {
                let (y, x) = (y / scale, x / scale);
                let at = ((image * grid.height + y) * grid.width + x) * channels;
                out.extend_from_slice(&input[at..at + channels]);
            }
            _ => out.resize(out.len() + channels, 0.0),
        }
    }
}

/// The dimensions of one causal 3-D convolution: `inputs` channels to `outputs` over `window`
/// frames and a `kernel × kernel` window of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalShape {
    pub inputs: usize,
    pub outputs: usize,
    pub window: usize,
    pub kernel: usize,
}

impl CausalShape {
    /// The `[rows, cols]` of the convolution's matrix.
    pub fn matrix(&self) -> [usize; 2] {
        [
            self.outputs,
            self.window * self.kernel * self.kernel * self.inputs,
        ]
    }
}

/// A convolution over `window` consecutive frames and a `kernel × kernel` window of each, at
/// stride 1.
pub struct CausalConv3d {
    linear: Linear,
    window: usize,
    kernel: usize,
}

impl CausalConv3d {
    /// # Errors
    /// When the weights' columns are not `window · kernel²` taps of whole inputs.
    pub fn new(linear: Linear, window: usize, kernel: usize) -> Result<Self, Error> {
        let cols = linear.matrix().cols;
        let taps = window * kernel * kernel;
        if taps == 0 || !cols.is_multiple_of(taps) {
            return Err(Error::Format(format!(
                "a convolution of {cols} columns over {window} frames of {kernel}-wide windows"
            )));
        }
        Ok(Self {
            linear,
            window,
            kernel,
        })
    }

    pub fn outputs(&self) -> usize {
        self.linear.matrix().rows
    }

    pub fn inputs(&self) -> usize {
        self.linear.matrix().cols / (self.window * self.kernel * self.kernel)
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// `store(pixel, channel, value)` for this thread's share of the output: output frame `f`
    /// (of `frames.len() − window + 1`) from `frames[f..f + window]`, each one image of `grid`
    /// (pixel-major rows of `inputs()` values); `pixel` counts the output frames' pixels in order.
    ///
    /// # Panics
    /// When `grid` is not one frame, or there are fewer frames than the window.
    pub fn forward(
        &self,
        ctx: &Ctx,
        (frames, grid): (&[&[f32]], Grid),
        patches: &mut Patches,
        store: impl FnMut(usize, usize, f32),
    ) {
        assert_eq!(grid.images, 1, "`grid` is one frame");
        let rows = (frames.len() + 1 - self.window) * grid.area();
        let source = Rows::Frames {
            conv: self,
            frames,
            grid,
        };
        project_chunks(ctx, &self.linear, (&source, rows), patches, store);
    }

    /// The windows of output pixels `pixels`, end to end, into `out`, over `frames`.
    fn gather(&self, (frames, grid): (&[&[f32]], Grid), pixels: Range<usize>, out: &mut Vec<f32>) {
        let window = (self.kernel, 1, self.inputs());
        out.clear();
        for pixel in pixels {
            let frame = pixel / grid.area();
            let (row, column) = (pixel / grid.width % grid.height, pixel % grid.width);
            for input in &frames[frame..frame + self.window] {
                push_window(out, (input, grid, 1), (0, row, column), window);
            }
        }
    }
}

impl Rows<'_> {
    /// Rows `range`, gathered into `out` unless they lie in the input as they are.
    fn chunk<'out>(
        &'out self,
        width: usize,
        range: Range<usize>,
        out: &'out mut Vec<f32>,
    ) -> &'out [f32] {
        match self {
            Rows::Dense(values) => &values[range.start * width..range.end * width],
            Rows::Windows {
                conv,
                input,
                grid,
                scale,
            } => {
                conv.gather((input, *grid, *scale), range, out);
                out
            }
            Rows::Frames { conv, frames, grid } => {
                conv.gather((frames, *grid), range, out);
                out
            }
        }
    }
}

/// `store(row, col, input[row] · W[col] + b[col])` for this thread's share of `rows` rows of
/// `input` (rows of the linear's width) and every output column.
pub fn project_rows(
    ctx: &Ctx,
    linear: &Linear,
    (input, rows): (&[f32], usize),
    patches: &mut Patches,
    store: impl FnMut(usize, usize, f32),
) {
    let width = linear.matrix().cols;
    let source = Rows::Dense(&input[..rows * width]);
    project_chunks(ctx, linear, (&source, rows), patches, store);
}

fn project_chunks(
    ctx: &Ctx,
    linear: &Linear,
    (source, rows): (&Rows, usize),
    patches: &mut Patches,
    mut store: impl FnMut(usize, usize, f32),
) {
    let matrix = linear.matrix();
    let outputs = matrix.rows;
    let by_rows = rows >= outputs || !outputs.is_multiple_of(COLUMN_ALIGN);
    let (mine, cols) = if by_rows {
        (ctx.split(rows, 1), 0..outputs)
    } else {
        (0..rows, ctx.split(outputs, COLUMN_ALIGN))
    };
    let Patches { gathered, quant } = patches;
    for start in mine.clone().step_by(CHUNK_ROWS) {
        let chunk = start..(start + CHUNK_ROWS).min(mine.end);
        let values = source.chunk(matrix.cols, chunk, gathered);
        let input = quant.input(&matrix, values);
        linear.project(&input, cols.clone(), |row, col, value| {
            store(start + row, col, value);
        });
    }
}

#[cfg(test)]
mod tests {
    use team::{SharedBuf, Team};

    use super::{CausalConv3d, Conv2d, Grid, Patches};
    use crate::linear::Linear;
    use crate::tensor::{DType, Table, Tensor, Vector};

    /// Two 3 × 3 images of 2 channels, value `100·image + 10·y + x + channel / 2`, through a
    /// 3 × 3 window at stride 2: output 0 sums channel 1 over the taps inside the image, output 1
    /// is the centre tap's channel 0 plus a bias of 0.25.
    #[test]
    fn strided_windows_are_zero_padded_at_the_edges() {
        let (inputs, outputs, taps) = (2, 2, 9);
        let mut weights = vec![0.0; outputs * taps * inputs];
        for tap in 0..taps {
            weights[tap * inputs + 1] = 1.0;
        }
        weights[taps * inputs + 4 * inputs] = 1.0;
        let weight = Tensor::encode(DType::F32, vec![outputs, taps * inputs], &weights).unwrap();
        let bias = Tensor::encode(DType::F32, vec![outputs], &[0.0, 0.25]).unwrap();
        let bias = Vector::new(bias).unwrap();
        let linear = Linear::new(Table::new(weight).unwrap(), Some(bias)).unwrap();
        let conv = Conv2d::new(linear, 3, 2).unwrap();
        let grid = Grid {
            images: 2,
            height: 3,
            width: 3,
        };
        let input: Vec<f32> = (0..grid.pixels() * inputs)
            .map(|at| {
                let (pixel, channel) = (at / inputs, at % inputs);
                let (image, y, x) = (pixel / 9, pixel / 3 % 3, pixel % 3);
                (100 * image + 10 * y + x) as f32 + channel as f32 / 2.0
            })
            .collect();
        let output = conv.output_grid(grid);
        let got = SharedBuf::new(output.pixels() * outputs);
        Team::global().run(|ctx| {
            let mut patches = Patches::default();
            conv.forward(
                ctx,
                (&input, grid),
                &mut patches,
                |pixel, channel, value| {
                    got.set(pixel * outputs + channel, value);
                },
            );
        });
        let want = [
            24.0, 0.25, 28.0, 2.25, 64.0, 20.25, 68.0, 22.25, //
            424.0, 100.25, 428.0, 102.25, 464.0, 120.25, 468.0, 122.25,
        ];
        assert_eq!(
            output,
            Grid {
                images: 2,
                height: 2,
                width: 2
            }
        );
        assert_eq!(got.read(), want);
    }

    /// A window of 2 frames of a 3 × 3 kernel over 1 channel at a single pixel: column 4 is frame
    /// 0's centre tap, column 13 frame 1's, weighted 1 and 100 with a bias of 0.5.
    #[test]
    fn causal_frames_are_read_oldest_first_then_tap_by_tap() {
        let (window, kernel) = (2, 3);
        let columns = window * kernel * kernel;
        let mut weights = vec![0.0; columns];
        weights[4] = 1.0;
        weights[13] = 100.0;
        let weight = Tensor::encode(DType::F32, vec![1, columns], &weights).unwrap();
        let bias = Tensor::encode(DType::F32, vec![1], &[0.5]).unwrap();
        let bias = Vector::new(bias).unwrap();
        let linear = Linear::new(Table::new(weight).unwrap(), Some(bias)).unwrap();
        let conv = CausalConv3d::new(linear, window, kernel).unwrap();
        let grid = Grid {
            images: 1,
            height: 1,
            width: 1,
        };
        let frames: [&[f32]; 3] = [&[3.0], &[5.0], &[7.0]];
        let got = SharedBuf::new(2);
        Team::global().run(|ctx| {
            let mut patches = Patches::default();
            conv.forward(ctx, (&frames, grid), &mut patches, |pixel, _, value| {
                got.set(pixel, value);
            });
        });
        assert_eq!(got.read(), [503.5, 705.5]);
    }
}
