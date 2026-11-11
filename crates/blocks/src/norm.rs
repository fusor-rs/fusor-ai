//! LayerNorm with its affine parameters, and GroupNorm over images.

use std::ops::Range;

use bundle::{Load, Scope, Store};
use kernels::Vector;
use kernels::conv::Grid;
use kernels::ops::layer_norm;
use team::{Ctx, SharedBuf};

pub struct LayerNorm {
    weight: Vector,
    bias: Vector,
}

impl LayerNorm {
    /// The LayerNorm of every row of `x` (rows of the norm's width).
    pub fn apply(&self, x: &[f32], eps: f32, out: &mut [f32]) {
        layer_norm(x, self.weight.values(), self.bias.values(), eps, out);
    }
}

/// `p.weight` and `p.bias`, f32 vectors of the norm's width.
impl Load for LayerNorm {
    type Shape = usize;

    fn load(store: &mut Store, prefix: &str, width: usize) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            weight: scope.load("weight", width)?,
            bias: scope.load("bias", width)?,
        })
    }
}

/// The dimensions of one GroupNorm: `channels` in `groups` equal runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Groups {
    pub channels: usize,
    pub groups: usize,
    pub eps: f32,
}

/// GroupNorm over pixel-major images (`kernels::conv::Grid`): per image, each run of channels is
/// normalized over all its values (with the biased variance, as `nn.GroupNorm`), then scaled and
/// shifted per channel.
pub struct GroupNorm {
    groups: Groups,
    weight: Vector,
    bias: Vector,
}

impl GroupNorm {
    /// Values of the `statistics` buffer `apply` needs for `images` images.
    pub fn statistics(&self, images: usize) -> usize {
        2 * images * self.groups.groups
    }

    /// `out = activation(norm(input))` for every pixel of `grid` (`out` may be `input`). Two
    /// phases: each thread's (image, group) statistics into `statistics`, a barrier, then its
    /// pixels; the caller barriers before `out` is read.
    pub fn apply(
        &self,
        ctx: &Ctx,
        (input, grid): (&SharedBuf, Grid),
        (out, statistics): (&SharedBuf, &SharedBuf),
        activation: impl Fn(f32) -> f32,
    ) {
        let Groups {
            channels, groups, ..
        } = self.groups;
        let values = &input.read()[..grid.pixels() * channels];
        let units = ctx.split(grid.images * groups, 1);
        for image in units.start / groups..units.end.div_ceil(groups) {
            let first = image * groups;
            let mine = units.start.max(first)..units.end.min(first + groups);
            let pixels = &values[image * grid.area() * channels..][..grid.area() * channels];
            let moments = statistics.part(2 * mine.start..2 * mine.end);
            self.moments(pixels, mine.start - first..mine.end - first, moments);
        }
        ctx.barrier();
        let moments = statistics.read();
        let in_place = std::ptr::eq(input, out);
        for pixel in ctx.split(grid.pixels(), 1) {
            let first = pixel / grid.area() * groups;
            let range = pixel * channels..(pixel + 1) * channels;
            let row = out.part(range.clone());
            // In place, `input` is this thread's to write: it is not read as a whole.
            if !in_place {
                row.copy_from_slice(&input.read()[range]);
            }
            self.normalize(row, &moments[2 * first..2 * (first + groups)], &activation);
        }
    }

    /// `row`, one pixel's channels, normalized by its image's `moments` (a mean and an inverse
    /// standard deviation per group), scaled, shifted and activated, a group's run at a time.
    fn normalize(&self, row: &mut [f32], moments: &[f32], activation: impl Fn(f32) -> f32) {
        let width = self.groups.channels / self.groups.groups;
        let affine = self.weight.values().chunks_exact(width);
        let affine = affine.zip(self.bias.values().chunks_exact(width));
        let runs = row.chunks_exact_mut(width).zip(affine);
        for ((run, (scales, shifts)), &[mean, inverse]) in runs.zip(moments.as_chunks::<2>().0) {
            for ((value, scale), shift) in run.iter_mut().zip(scales).zip(shifts) {
                *value = activation((*value - mean) * inverse * scale + shift);
            }
        }
    }

    /// The mean and the inverse standard deviation of the groups `groups` of one image
    /// (`pixels`, its rows) into `moments` (their pairs), summed in f64: the groups together, a
    /// pass over the pixels for their sums and one for their squared deviations, each group's
    /// sum in the order of its pixels.
    fn moments(&self, pixels: &[f32], groups: Range<usize>, moments: &mut [f32]) {
        let Groups { channels, eps, .. } = self.groups;
        let width = channels / self.groups.groups;
        let span = groups.start * width..groups.end * width;
        let count = (pixels.len() / channels * width) as f64;
        let pairs = moments.as_chunks_mut::<2>().0;
        // `Iterator::sum` starts from −0.0; so do these, each the same sum pixel by pixel.
        let mut sums = vec![-0.0_f64; groups.len()];
        for pixel in pixels.chunks_exact(channels) {
            for (sum, run) in sums.iter_mut().zip(pixel[span.clone()].chunks_exact(width)) {
                *sum += f64::from(run.iter().sum::<f32>());
            }
        }
        for ([mean, _], sum) in pairs.iter_mut().zip(&mut sums) {
            *mean = (*sum / count) as f32;
            *sum = -0.0;
        }
        for pixel in pixels.chunks_exact(channels) {
            let runs = pixel[span.clone()].chunks_exact(width);
            for ((sum, run), [mean, _]) in sums.iter_mut().zip(runs).zip(&*pairs) {
                *sum += f64::from(run.iter().map(|value| (value - mean).powi(2)).sum::<f32>());
            }
        }
        for ([_, inverse], squares) in pairs.iter_mut().zip(sums) {
            *inverse = (1.0 / (squares / count + f64::from(eps)).sqrt()) as f32;
        }
    }
}

/// `p.weight` and `p.bias`, f32 vectors of the channels.
impl Load for GroupNorm {
    type Shape = Groups;

    fn load(store: &mut Store, prefix: &str, groups: Groups) -> Result<Self, bundle::Error> {
        if groups.groups == 0 || !groups.channels.is_multiple_of(groups.groups) {
            return Err(bundle::Error::Format(format!(
                "{prefix}: {} channels in {} groups",
                groups.channels, groups.groups
            )));
        }
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            groups,
            weight: scope.load("weight", groups.channels)?,
            bias: scope.load("bias", groups.channels)?,
        })
    }
}
