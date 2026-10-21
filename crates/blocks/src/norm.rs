//! LayerNorm with its affine parameters, and GroupNorm over images.

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
        for unit in ctx.split(grid.images * groups, 1) {
            let (mean, inverse) = self.moments(values, grid, unit);
            statistics.set(2 * unit, mean);
            statistics.set(2 * unit + 1, inverse);
        }
        ctx.barrier();
        let moments = statistics.read();
        let (scales, shifts) = (self.weight.values(), self.bias.values());
        let width = channels / groups;
        let in_place = std::ptr::eq(input, out);
        for pixel in ctx.split(grid.pixels(), 1) {
            let first = pixel / grid.area() * groups;
            let range = pixel * channels..(pixel + 1) * channels;
            let row = out.part(range.clone());
            // In place, `input` is this thread's to write: it is not read as a whole.
            if !in_place {
                row.copy_from_slice(&input.read()[range]);
            }
            for (channel, value) in row.iter_mut().enumerate() {
                let unit = first + channel / width;
                let (mean, inverse) = (moments[2 * unit], moments[2 * unit + 1]);
                *value = activation((*value - mean) * inverse * scales[channel] + shifts[channel]);
            }
        }
    }

    /// The mean and the inverse standard deviation of `unit` (image-major, then group), summed
    /// in f64.
    fn moments(&self, values: &[f32], grid: Grid, unit: usize) -> (f32, f32) {
        let Groups {
            channels,
            groups,
            eps,
        } = self.groups;
        let width = channels / groups;
        let (image, group) = (unit / groups, unit % groups);
        let image_values = grid.area() * channels;
        let runs = values[image * image_values..(image + 1) * image_values]
            .chunks_exact(channels)
            .map(|pixel| &pixel[group * width..(group + 1) * width]);
        let count = (grid.area() * width) as f64;
        let sum: f64 = runs
            .clone()
            .map(|run| f64::from(run.iter().sum::<f32>()))
            .sum();
        let mean = (sum / count) as f32;
        let squares: f64 = runs
            .map(|run| f64::from(run.iter().map(|value| (value - mean).powi(2)).sum::<f32>()))
            .sum();
        let inverse = 1.0 / (squares / count + f64::from(eps)).sqrt();
        (mean, inverse as f32)
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
