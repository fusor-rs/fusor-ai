//! The activations of one `Unet::forward`, sized for the largest call by the shape.

use kernels::conv::Patches;
use team::{PerThread, SharedBuf};

use super::Shape;
use crate::attention;

impl Shape {
    /// Channels of every down block's output, in the order the up path reads them back.
    fn skip_channels(&self) -> Vec<(usize, usize)> {
        let mut skips = vec![(self.channels[0], 0)];
        for (level, &channels) in self.channels.iter().enumerate() {
            skips.extend(std::iter::repeat_n((channels, level), self.blocks));
            if level + 1 < self.levels() {
                skips.push((channels, level + 1));
            }
        }
        skips
    }

    /// Activations for calls over up to `images` latents of up to `pixels` pixels each.
    pub fn buffers(&self, images: usize, pixels: usize, threads: usize) -> Buffers {
        let plane = |channels: usize, level: usize| images * channels * (pixels >> (2 * level));
        let neighbours = |level: usize| {
            let level = level as isize;
            let around = [level - 1, level, level + 1].map(|at| self.level_channels(at));
            around.into_iter().fold(0, usize::max)
        };
        let largest =
            |size: &dyn Fn(usize) -> usize| (0..self.levels()).map(size).fold(0, usize::max);
        let hidden = largest(&|level| plane(neighbours(level), level));
        let inner = largest(&|level| plane(self.channels[level], level));
        let skips = self.skip_channels();
        let new = SharedBuf::new;
        Buffers {
            images,
            pixels,
            sample: new(images * pixels * self.latent),
            prediction: new(images * pixels * self.latent),
            skip_widths: skips.iter().map(|&(channels, _)| channels).collect(),
            skips: (skips.into_iter())
                .map(|(channels, level)| new(plane(channels, level)))
                .collect(),
            hidden: [new(hidden), new(hidden)],
            wide: new(2 * hidden),
            normed: new(hidden),
            inner: new(inner),
            tokens: new(inner),
            qkv: new(3 * inner),
            attention: new(inner),
            feed_forward: new(self.feed_forward * inner),
            statistics: new(2 * images * self.groups),
            sinusoid: new(self.channels[0]),
            time: new(self.time_width()),
            activated: new(self.time_width()),
            projected: new(largest(&|level| self.channels[level])),
            patches: PerThread::new(threads, Patches::default),
            attention_scratch: PerThread::new(threads, attention::Scratch::default),
        }
    }
}

/// Activations of one `forward`.
pub struct Buffers {
    pub(super) images: usize,
    pub(super) pixels: usize,
    /// The latents in, `images × pixels × latent`, pixel-major.
    pub sample: SharedBuf,
    /// The predicted noise, as `sample`.
    pub prediction: SharedBuf,
    /// Every down block's output, and its channels.
    pub(super) skips: Vec<SharedBuf>,
    pub(super) skip_widths: Vec<usize>,
    pub(super) hidden: [SharedBuf; 2],
    pub(super) wide: SharedBuf,
    pub(super) normed: SharedBuf,
    pub(super) inner: SharedBuf,
    pub(super) tokens: SharedBuf,
    pub(super) qkv: SharedBuf,
    pub(super) attention: SharedBuf,
    pub(super) feed_forward: SharedBuf,
    pub(super) statistics: SharedBuf,
    pub(super) sinusoid: SharedBuf,
    pub(super) time: SharedBuf,
    pub(super) activated: SharedBuf,
    pub(super) projected: SharedBuf,
    pub(super) patches: PerThread<Patches>,
    pub(super) attention_scratch: PerThread<attention::Scratch>,
}
