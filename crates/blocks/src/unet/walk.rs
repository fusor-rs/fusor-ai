//! `Unet::forward` phase by phase: the time embedding, the down path, the middle block, the up
//! path and the output.

use kernels::conv::{Grid, project_rows};
use kernels::ops::silu;
use team::SharedBuf;

use super::Unet;
use super::level::Middle;
use super::pass::Pass;
use crate::SINUSOID_BASE;

impl Unet {
    /// `activated = silu(time_out(silu(time_in(sinusoid(timestep)))))`.
    pub(super) fn embed_time(&self, pass: &Pass, timestep: f32) {
        let (ctx, buffers) = (pass.ctx, pass.buffers);
        if ctx.is_first() {
            sinusoid(timestep, buffers.sinusoid.part(0..self.shape.channels[0]));
        }
        ctx.barrier();
        let stages = [
            (&self.time_in, &buffers.sinusoid, &buffers.time),
            (&self.time_out, &buffers.time, &buffers.activated),
        ];
        for (linear, input, output) in stages {
            let patches = buffers.patches.get(ctx);
            project_rows(ctx, linear, (input.read(), 1), patches, |_, col, value| {
                output.set(col, silu(value));
            });
            ctx.barrier();
        }
    }

    /// `conv_in` and the down path, every block's output into its skip buffer; returns the grid
    /// at the bottom.
    pub(super) fn down_path(&self, pass: &Pass, grid: Grid, layer: &mut usize) -> Grid {
        let (ctx, buffers) = (pass.ctx, pass.buffers);
        let skips = &buffers.skips;
        pass.convolve(&self.conv_in, (&buffers.sample, grid), &skips[0]);
        let (mut grid, mut skip) = (grid, 0);
        for level in &self.down {
            for (block, resnet) in level.resnets.iter().enumerate() {
                let (input, output) = (&skips[skip], &skips[skip + 1]);
                let planes = pass.resnet(input, output, &buffers.normed);
                resnet.forward(ctx, &planes, grid, &pass.shared);
                if let Some(transformer) = level.transformers.get(block) {
                    pass.transform(transformer, (output, grid), layer);
                }
                skip += 1;
            }
            if let Some(downsample) = &level.resample {
                pass.convolve(downsample, (&skips[skip], grid), &skips[skip + 1]);
                grid = downsample.output_grid(grid);
                skip += 1;
            }
        }
        grid
    }

    /// The middle block, from the last skip into `hidden[0]`.
    pub(super) fn middle(&self, pass: &Pass, grid: Grid, layer: &mut usize) {
        let (ctx, buffers) = (pass.ctx, pass.buffers);
        let (last, hidden) = (buffers.skips.last().expect("a level"), &buffers.hidden[0]);
        let Middle {
            first,
            transformer,
            second,
        } = &self.middle;
        first.forward(
            ctx,
            &pass.resnet(last, hidden, &buffers.normed),
            grid,
            &pass.shared,
        );
        pass.transform(transformer, (hidden, grid), layer);
        second.forward(
            ctx,
            &pass.resnet(hidden, hidden, &buffers.normed),
            grid,
            &pass.shared,
        );
    }

    /// The up path from `hidden[0]`: each block's input is the hidden state and the next skip
    /// side by side. Returns the grid at the top and the hidden buffer that holds the result.
    pub(super) fn up_path<'call>(
        &self,
        pass: &Pass<'call>,
        grid: Grid,
        layer: &mut usize,
    ) -> (Grid, &'call SharedBuf) {
        let (ctx, buffers) = (pass.ctx, pass.buffers);
        let (mut grid, mut current, mut skip) = (grid, 0, buffers.skips.len());
        for level in self.up.iter().rev() {
            for (block, resnet) in level.resnets.iter().enumerate() {
                skip -= 1;
                let hidden = &buffers.hidden[current];
                let skip_width = buffers.skip_widths[skip];
                let widths = (resnet.inputs() - skip_width, skip_width);
                pass.concatenate((hidden, &buffers.skips[skip]), widths, grid);
                let planes = pass.resnet(&buffers.wide, hidden, &buffers.wide);
                resnet.forward(ctx, &planes, grid, &pass.shared);
                if let Some(transformer) = level.transformers.get(block) {
                    pass.transform(transformer, (hidden, grid), layer);
                }
            }
            if let Some(upsample) = &level.resample {
                let (input, output) = (&buffers.hidden[current], &buffers.hidden[1 - current]);
                let outputs = upsample.outputs();
                let patches = buffers.patches.get(ctx);
                upsample.forward_upsampled(
                    ctx,
                    (input.read(), grid),
                    patches,
                    |pixel, channel, value| {
                        output.set(pixel * outputs + channel, value);
                    },
                );
                ctx.barrier();
                grid = grid.doubled();
                current = 1 - current;
            }
        }
        (grid, &buffers.hidden[current])
    }

    /// `prediction = conv_out(silu(norm_out(hidden)))`.
    pub(super) fn out(&self, pass: &Pass, grid: Grid, hidden: &SharedBuf) {
        let (ctx, buffers) = (pass.ctx, pass.buffers);
        let normed = (&buffers.normed, &buffers.statistics);
        self.norm_out.apply(ctx, (hidden, grid), normed, silu);
        ctx.barrier();
        pass.convolve(&self.conv_out, (&buffers.normed, grid), &buffers.prediction);
    }
}

/// The timestep's sinusoids, `cos(t · fᵢ)` then `sin(t · fᵢ)` for `fᵢ = 10000^(−i / half)`, in
/// f32 like `get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)`.
fn sinusoid(timestep: f32, out: &mut [f32]) {
    let half = out.len() / 2;
    let (cosines, sines) = out.split_at_mut(half);
    for (i, (cosine, sine)) in cosines.iter_mut().zip(sines).enumerate() {
        let exponent = -SINUSOID_BASE.ln() * i as f32 / half as f32;
        let angle = timestep * exponent.exp();
        *cosine = angle.cos();
        *sine = angle.sin();
    }
}
