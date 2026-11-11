//! The decoder and the first frame's encoder of Wan2.1's causal video autoencoder (Wan2.1's and
//! Motif-Video's). The decoder turns latents into frames: a pointwise linear (`post_quant`, into
//! which a compiler folds the latents' denormalization), a causal 3-D convolution, a middle block
//! (a residual block, a single-head self-attention over each frame's pixels added back, a
//! residual block), then per level `blocks` residual blocks and, but after the last level, a
//! resample: nearest ×2 and a 3 × 3 convolution halving the channels, preceded on the levels
//! marked `temporal` by a causal time convolution doubling the frames (each output frame's two
//! halves of its channels). Last a norm, SiLU and a causal convolution to RGB. Norms are per
//! pixel: `x / ‖x‖ · √channels · γ`.
//!
//! Every 3-D convolution is causal over the frames (`kernels::conv::CausalConv3d`): it keeps its
//! last two input frames (`Buffers`, zeros at a video's start), so latent frames decoded one by one
//! give the frames of the whole video. Latent frame 0 skips the time convolutions and gives one
//! frame; every later one gives `2^temporal levels` frames.
//!
//! The decoder runs the steps up to the middle block's over whole latent frames
//! (`Decoder::decode_latents`), and the levels and the head over horizontal strips of them
//! (`Decoder::decode_strip`), so the levels' activations and histories span a strip, not a
//! frame. A strip's level computes its rows and a halo on either side (`Shape::halos`), which the
//! next strip computes again. Rows near a window's cut edge read zeros where the frame has
//! values, and each convolution moves that error one row inward, in this frame and, through the
//! histories, in the next ones; the halos cover every convolution after a level's input, so a
//! strip's own rows equal the whole frame's bit for bit.
//!
//! Tensors under a prefix `p`: `p.post_quant` (a linear), `p.conv_in`, `p.middle.{first,
//! attention, second}`, `p.up.{l}.resnets.{r}`, `p.up.{l}.resample` (a `Conv2d`) and on temporal
//! levels `p.up.{l}.time`, `p.head.{norm, conv}`; a residual block's `norm1`, `conv1`, `norm2`,
//! `conv2` and, when its channels change, `shortcut` (a linear); the attention's `norm`, `qkv` and
//! `out`. Norms are vectors; convolutions with frames are `CausalConv3d`s.
//!
//! The encoder (`Encoder`) reads a video's first frame alone, as the autoencoder's first chunk
//! does: every causal convolution sees two zero frames before it, so only its kernel's last frame
//! touches the input (`CausalConv3d`s of one frame), and the time convolutions are skipped. It
//! mirrors the decoder: a convolution into the last level's channels, the levels from the decoder's
//! last to its first, `blocks − 1` residual blocks each, every level but the last halving the
//! height and width (`Shape::encoder_resample`), the middle block, a norm, SiLU and a convolution
//! to `2 · latent` channels, and a pointwise linear (`quant`): each latent pixel's means, then its
//! log-variances. Tensors under `p.encoder`: `conv_in`, `down.{l}.resnets.{r}`,
//! `down.{l}.resample` (a `Conv2d`), `middle.{first, attention, second}`, `head.{norm, conv}`,
//! `quant`.
//!
//! `decode_latents`, `decode_strip` and `encode` run inside a team run.

mod attention;
mod buffers;
mod encoder;
mod head;
mod level;
mod middle;
mod resnet;
mod walk;

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::Linear;
use kernels::conv::{CausalConv3d, CausalShape, Grid};
use team::Ctx;

pub use buffers::{Buffers, Capacity};
use head::Head;
use level::Level;
use middle::Middle;
use walk::{Walk, into};

pub use encoder::Encoder;

/// The dimensions of one decoder, which its encoder mirrors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Channels of a latent pixel.
    pub latent: usize,
    /// Channels of each level's residual blocks, from the latents' resolution on.
    pub channels: &'static [usize],
    /// Residual blocks per level.
    pub blocks: usize,
    /// Per level but the last: whether its resample doubles the frames too.
    pub temporal: &'static [bool],
    /// Channels of an output pixel.
    pub outputs: usize,
    /// The side of the convolutions' cubes: the frames a causal convolution reads (the output's
    /// and the ones before it) and the pixels of its window along each side.
    pub kernel: usize,
}

impl Shape {
    const fn levels(&self) -> usize {
        self.channels.len()
    }

    /// Channels a level's first residual block reads: the middle's, or the previous level's
    /// halved by its resample.
    fn level_inputs(&self, level: usize) -> usize {
        match level.checked_sub(1) {
            Some(above) => self.channels[above] / 2,
            None => self.channels[0],
        }
    }

    /// A causal convolution of `inputs` channels to `outputs` over `frames` frames of the
    /// shape's `kernel × kernel` windows.
    fn causal(&self, (inputs, outputs): (usize, usize), frames: usize) -> CausalShape {
        CausalShape {
            inputs,
            outputs,
            window: frames,
            kernel: self.kernel,
        }
    }

    /// The window of the encoder's stride-2 resample. Wan pads only after the last row and column
    /// (`ZeroPad2d((0, 1, 0, 1))`) before a `kernel × kernel` window at stride 2; a window of
    /// `2 · kernel − 1` centred on row `2r` holds that window in its last `kernel` rows and
    /// columns, so a compiler writes it there, zeros before, and `Conv2d`'s centred padding gives
    /// the same output.
    pub const fn encoder_resample(&self) -> usize {
        2 * self.kernel - 1
    }

    /// Whether level `level` doubles the frames with a causal time convolution.
    pub const fn has_time(&self, level: usize) -> bool {
        level < self.temporal.len() && self.temporal[level]
    }

    /// Frames level `level` works on in a call after the first: every time convolution above it
    /// doubled them (`levels()` counts the output's).
    const fn level_frames(&self, level: usize) -> usize {
        let mut frames = 1;
        let mut above = 0;
        while above < level {
            if self.has_time(above) {
                frames *= 2;
            }
            above += 1;
        }
        frames
    }

    /// Output frames of latent frame `index`: 1 for the first, `2^temporal levels` after.
    pub const fn frames(&self, index: usize) -> usize {
        if index == 0 {
            1
        } else {
            self.level_frames(self.levels())
        }
    }

    /// Output pixels per latent pixel along each side: `2^(levels − 1)`.
    pub const fn scale(&self) -> usize {
        1 << (self.levels() - 1)
    }

    /// Per level, the rows it computes beyond a strip's on either side, at its resolution: the
    /// rows its convolutions read around those the next level's resample reads (the head's on
    /// the last level), rounded out to whole rows of the level below on all levels but the
    /// first.
    fn halos(&self) -> Vec<usize> {
        let radius = self.kernel / 2;
        let blocks = radius * 2 * self.blocks;
        let mut halos = vec![0; self.levels()];
        let mut needed = blocks + radius;
        for level in (1..self.levels()).rev() {
            let below = (needed + radius).div_ceil(2);
            halos[level] = 2 * below;
            needed = below + blocks;
        }
        halos[0] = needed;
        halos
    }
}

pub struct Decoder {
    shape: Shape,
    halos: Vec<usize>,
    post_quant: Linear,
    conv_in: CausalConv3d,
    middle: Middle,
    up: Vec<Level>,
    head: Head,
}

impl Decoder {
    /// The output frame's grid for latents of `grid` (one frame).
    pub fn output_grid(&self, grid: Grid) -> Grid {
        Grid {
            height: self.shape.scale() * grid.height,
            width: self.shape.scale() * grid.width,
            ..grid
        }
    }

    /// Latent frame `index` of a video (`buffers.latent`, one frame of `grid`) through the steps
    /// up to the middle block's, kept for `decode_strip`. Latent frames are decoded in order from
    /// 0.
    ///
    /// # Panics
    /// When the buffers do not hold the latents.
    pub fn decode_latents(&self, ctx: &Ctx, buffers: &Buffers, (grid, index): (Grid, usize)) {
        let Capacity { pixels, video, .. } = buffers.capacity;
        let area = grid.area();
        assert!(
            grid.images == 1 && area <= pixels && (index + 1) * area <= video,
            "the buffers hold the video's latent frames"
        );
        let Shape {
            latent, channels, ..
        } = self.shape;
        let mut walk = Walk::new(ctx, buffers, (grid, index), &buffers.latent_histories);
        walk.project(
            &self.post_quant,
            &buffers.latent,
            into(&buffers.planes[1], latent),
        );
        walk.causal(&self.conv_in, 1, into(&buffers.planes[0], channels[0]));
        self.middle.forward(&mut walk);
        walk.finish();
        let values = channels[0] * area;
        let mine = ctx.split(values, 1);
        let kept = index * values + mine.start..index * values + mine.end;
        buffers
            .middles
            .part(kept)
            .copy_from_slice(&buffers.planes[walk.current].read()[mine]);
        ctx.barrier();
    }

    /// `buffers.frames` = the output rows of latent rows `rows` of the `shape.frames(index)`
    /// frames of latent frame `index` (one frame of `grid`), from what `decode_latents` kept of
    /// it. A strip's latent frames are decoded in order from 0.
    ///
    /// # Panics
    /// When the buffers do not hold the strip.
    pub fn decode_strip(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        (grid, index): (Grid, usize),
        rows: Range<usize>,
    ) {
        let Capacity { width, strip, .. } = buffers.capacity;
        assert!(
            grid.width <= width && rows.len() <= strip && rows.end <= grid.height,
            "the buffers hold the strip"
        );
        let windows: Vec<_> = (0..self.shape.levels())
            .map(|level| self.window(grid.height, &rows, level))
            .collect();
        let window = Grid {
            height: windows[0].len(),
            ..grid
        };
        self.load_window(ctx, buffers, (grid, index), &windows[0]);
        let mut walk = Walk::new(ctx, buffers, (window, index), &buffers.strip_histories);
        for (level, up) in self.up.iter().enumerate() {
            let kept = windows.get(level + 1).map_or(0..0, |next| {
                next.start / 2 - windows[level].start..next.end / 2 - windows[level].start
            });
            up.forward(&mut walk, kept);
        }
        let scale = self.shape.scale();
        let last = &windows[windows.len() - 1];
        let output = rows.start * scale - last.start..rows.end * scale - last.start;
        let store = self.store_rows(buffers, walk.grid, output);
        self.head.forward(&mut walk, store);
        walk.finish();
    }

    /// Rows `window` of latent frame `index`'s middle-block output (frames of `grid`) into
    /// plane 0, then a barrier.
    fn load_window(
        &self,
        ctx: &Ctx,
        buffers: &Buffers,
        (grid, index): (Grid, usize),
        window: &Range<usize>,
    ) {
        let row = self.shape.channels[0] * grid.width;
        let first = index * row * grid.height + window.start * row;
        let mine = ctx.split(window.len() * row, 1);
        let source = &buffers.middles.read()[first + mine.start..first + mine.end];
        buffers.planes[0].part(mine).copy_from_slice(source);
        ctx.barrier();
    }

    /// The store of the head's output over frames of `grid` into `buffers.frames`: rows
    /// `output` of each frame, the rest dropped.
    fn store_rows<'out>(
        &self,
        buffers: &'out Buffers,
        grid: Grid,
        output: Range<usize>,
    ) -> impl FnMut(usize, usize, f32) + 'out {
        let Grid { height, width, .. } = grid;
        let outputs = self.shape.outputs;
        move |pixel, channel, value| {
            let (frame, row, column) = (
                pixel / (height * width),
                pixel / width % height,
                pixel % width,
            );
            if output.contains(&row) {
                let at = (frame * output.len() + row - output.start) * width + column;
                buffers.frames.set(at * outputs + channel, value);
            }
        }
    }

    /// The rows level `level` computes for latent rows `rows` of a frame `height` latent rows
    /// tall.
    fn window(&self, height: usize, rows: &Range<usize>, level: usize) -> Range<usize> {
        let halo = self.halos[level];
        (rows.start << level).saturating_sub(halo)
            ..((rows.end << level) + halo).min(height << level)
    }
}

impl Load for Decoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            latent,
            channels,
            outputs,
            ..
        } = shape;
        if shape.temporal.len() + 1 != shape.levels() {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {} levels and {} temporal flags",
                shape.levels(),
                shape.temporal.len()
            )));
        }
        let last = channels[shape.levels() - 1];
        let up = (0..shape.levels())
            .map(|level| Level::load(store, &format!("{prefix}.up.{level}"), &shape, level))
            .collect::<Result<Vec<_>, _>>()?;
        let middle = Middle::load(store, prefix, (shape, shape.kernel))?;
        let causal = |inputs, outputs| shape.causal((inputs, outputs), shape.kernel);
        let head = Head::load(store, prefix, (last, causal(last, outputs)))?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            shape,
            halos: shape.halos(),
            post_quant: scope.load("post_quant", [latent; 2])?,
            conv_in: scope.load("conv_in", causal(latent, channels[0]))?,
            middle,
            up,
            head,
        })
    }
}
