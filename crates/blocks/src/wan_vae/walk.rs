//! The planes of one decoder or encoder call and the causal convolutions' histories it advances:
//! which plane holds the current activations, how many frames of which grid, and which history is
//! next.

use std::ops::Range;

use kernels::Linear;
use kernels::Vector;
use kernels::conv::{CausalConv3d, Conv2d, Grid, project_rows};
use team::{Ctx, SharedBuf};

use super::Buffers;

/// `F.normalize`'s floor for a pixel's norm.
const NORM_FLOOR: f32 = 1e-12;

/// The store of a product into `plane`, pixel-major rows of `width` values.
pub(super) fn into(plane: &SharedBuf, width: usize) -> impl FnMut(usize, usize, f32) + '_ {
    move |pixel, channel, value| plane.set(pixel * width + channel, value)
}

pub(super) struct Walk<'call> {
    pub(super) ctx: &'call Ctx<'call>,
    pub(super) buffers: &'call Buffers,
    /// The plane holding the activations: `frames` frames of `grid`.
    pub(super) current: usize,
    pub(super) frames: usize,
    pub(super) grid: Grid,
    /// The latent frame of the video: 0 starts every history from zeros.
    index: usize,
    /// The histories the call advances, and the next causal convolution's.
    histories: &'call [SharedBuf],
    next: usize,
}

impl<'call> Walk<'call> {
    /// A walk over one frame of `grid` (latents, or the encoder's video frame) in plane 0, of
    /// latent frame `index`, advancing `histories`.
    pub(super) fn new(
        ctx: &'call Ctx<'call>,
        buffers: &'call Buffers,
        (grid, index): (Grid, usize),
        histories: &'call [SharedBuf],
    ) -> Self {
        Self {
            ctx,
            buffers,
            current: 0,
            frames: 1,
            grid,
            index,
            histories,
            next: 0,
        }
    }

    /// The two planes that are not the current one.
    pub(super) fn free(&self) -> [usize; 2] {
        let planes = self.buffers.planes.len();
        [(self.current + 1) % planes, (self.current + 2) % planes]
    }

    /// Every history was advanced exactly once.
    pub(super) fn finish(&self) {
        assert_eq!(
            self.next,
            self.histories.len(),
            "a history for every causal convolution"
        );
    }

    fn pixels(&self) -> usize {
        self.frames * self.grid.area()
    }

    /// `store` of `linear` over the current frames' rows of `input`, then a barrier.
    pub(super) fn project(
        &self,
        linear: &Linear,
        input: &SharedBuf,
        store: impl FnMut(usize, usize, f32),
    ) {
        let patches = self.buffers.patches.get(self.ctx);
        project_rows(
            self.ctx,
            linear,
            (input.read(), self.pixels()),
            patches,
            store,
        );
        self.ctx.barrier();
    }

    /// `to = activation(x / ‖x‖ · √channels · gamma)` per pixel of `from`; `to` may be `from`.
    pub(super) fn norm(
        &self,
        (from, to): (usize, usize),
        gamma: &Vector,
        activation: impl Fn(f32) -> f32,
    ) {
        let channels = gamma.values().len();
        let scale = (channels as f32).sqrt();
        let (source, target) = (&self.buffers.planes[from], &self.buffers.planes[to]);
        for pixel in self.ctx.split(self.pixels(), 1) {
            let range = pixel * channels..(pixel + 1) * channels;
            let row = target.part(range.clone());
            if from != to {
                row.copy_from_slice(&source.read()[range]);
            }
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            let inverse = scale / norm.max(NORM_FLOOR);
            for (value, weight) in row.iter_mut().zip(gamma.values()) {
                *value = activation(*value * inverse * weight);
            }
        }
        self.ctx.barrier();
    }

    /// `store` of `conv` over the frames in plane `from` and the history (zeros before latent
    /// frame 0), which then holds the last frames of that timeline; a convolution of one frame
    /// keeps no history.
    pub(super) fn causal(
        &mut self,
        conv: &CausalConv3d,
        from: usize,
        store: impl FnMut(usize, usize, f32),
    ) {
        let buffers = self.buffers;
        let len = self.grid.area() * conv.inputs();
        let remembered = conv.window() - 1;
        let input = &buffers.planes[from].read()[..self.frames * len];
        let patches = buffers.patches.get(self.ctx);
        if remembered == 0 {
            let timeline: Vec<&[f32]> = input.chunks_exact(len).collect();
            conv.forward(self.ctx, (&timeline, self.grid), patches, store);
            self.ctx.barrier();
            return;
        }
        let history = &self.histories[self.next];
        if self.index == 0 {
            self.clear(history, remembered * len);
        }
        let timeline: Vec<&[f32]> = (history.read()[..remembered * len].chunks_exact(len))
            .chain(input.chunks_exact(len))
            .collect();
        conv.forward(self.ctx, (&timeline, self.grid), patches, store);
        self.ctx.barrier();
        for at in self.ctx.split(len, 1) {
            for kept in 0..remembered {
                let source = self.frames + kept;
                let value = match source.checked_sub(remembered) {
                    Some(frame) => input[frame * len + at],
                    None => history.get(source * len + at),
                };
                history.set(kept * len + at, value);
            }
        }
        self.ctx.barrier();
        self.next += 1;
    }

    /// The frames doubled by a time convolution: output channel `j` of frame `f` is channel
    /// `j % channels` of frame `2f + j / channels`. Latent frame 0 skips it, its history zeros.
    pub(super) fn double_frames(&mut self, time: &CausalConv3d) {
        if self.index == 0 {
            let values = (time.window() - 1) * self.grid.area() * time.inputs();
            self.clear(&self.histories[self.next], values);
            self.next += 1;
            return;
        }
        let channels = time.inputs();
        let area = self.grid.area();
        let target = self.free()[0];
        let output = &self.buffers.planes[target];
        self.causal(time, self.current, |pixel, channel, value| {
            let frame = 2 * (pixel / area) + channel / channels;
            output.set(
                (frame * area + pixel % area) * channels + channel % channels,
                value,
            );
        });
        self.current = target;
        self.frames *= 2;
    }

    /// Zeros the first `values` of `history`, then a barrier.
    fn clear(&self, history: &SharedBuf, values: usize) {
        history.part(self.ctx.split(values, 1)).fill(0.0);
        self.ctx.barrier();
    }

    /// Rows `rows` of every frame at twice the height and width: nearest ×2, then `resample`.
    pub(super) fn double_grid(&mut self, resample: &Conv2d, rows: Range<usize>) {
        let buffers = self.buffers;
        let target = self.free()[0];
        let output = &buffers.planes[target];
        let row = self.grid.width * resample.inputs();
        let kept = Grid {
            images: 1,
            height: rows.len(),
            ..self.grid
        };
        let doubled = kept.doubled();
        let outputs = resample.outputs();
        let patches = buffers.patches.get(self.ctx);
        for frame in 0..self.frames {
            let first = (frame * self.grid.height + rows.start) * row;
            let input = &buffers.planes[self.current].read()[first..first + rows.len() * row];
            let offset = frame * doubled.area();
            resample.forward_upsampled(
                self.ctx,
                (input, kept),
                patches,
                |pixel, channel, value| {
                    output.set((offset + pixel) * outputs + channel, value);
                },
            );
        }
        self.ctx.barrier();
        self.current = target;
        self.grid = doubled;
    }

    /// The frames at half the height and width: `resample` at stride 2.
    pub(super) fn halve_grid(&mut self, resample: &Conv2d) {
        let buffers = self.buffers;
        let target = self.free()[0];
        let grid = Grid {
            images: self.frames,
            ..self.grid
        };
        let input = (buffers.planes[self.current].read(), grid);
        let patches = buffers.patches.get(self.ctx);
        let store = into(&buffers.planes[target], resample.outputs());
        resample.forward(self.ctx, input, patches, store);
        self.ctx.barrier();
        self.current = target;
        self.grid = resample.output_grid(self.grid);
    }
}
