//! The activations of one `Decoder::decode`, sized by the steps it runs.

use kernels::conv::Patches;
use team::{PerThread, SharedBuf};

use super::Shape;
use crate::attention;

impl Shape {
    /// The channels and the doublings of the latents' resolution of each step's output, in the
    /// order `decode` runs them: `conv_in`, the middle block's residual blocks, then each
    /// level's residual blocks and its resample. The steps alternate between the two planes.
    fn steps(&self) -> Vec<(usize, u32)> {
        let bottom = (self.bottom(), 0);
        let mut steps = Vec::from([bottom, bottom, bottom]);
        for (scale, level) in (0..self.levels()).rev().enumerate() {
            let scale = scale as u32;
            let channels = self.channels[level];
            steps.extend(std::iter::repeat_n((channels, scale), self.blocks + 1));
            if level > 0 {
                steps.push((channels, scale + 1));
            }
        }
        steps
    }

    /// Activations for decoding one image of up to `pixels` latent pixels.
    pub fn buffers(&self, pixels: usize, threads: usize) -> Buffers {
        let size = |(channels, scale): (usize, u32)| channels * (pixels << (2 * scale));
        let mut planes = [0; 2];
        for (step, output) in self.steps().into_iter().enumerate() {
            planes[step % 2] = planes[step % 2].max(size(output));
        }
        let resnets = (0..self.levels()).map(|level| {
            let scale = self.scale() - level as u32;
            size((self.channels[level], scale))
        });
        let new = SharedBuf::new;
        Buffers {
            pixels,
            latents: new(pixels * self.latent),
            quantized: new(pixels * self.latent),
            planes: planes.map(new),
            inner: new(resnets.fold(0, usize::max)),
            qkv: new(3 * pixels * self.bottom()),
            attention: new(pixels * self.bottom()),
            statistics: new(2 * self.groups),
            image: new(self.outputs * (pixels << (2 * self.scale()))),
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, attention::Scratch::default),
        }
    }
}

/// Activations of one `decode`.
pub struct Buffers {
    pub(super) pixels: usize,
    /// The latents in, `pixels × latent`, pixel-major.
    pub latents: SharedBuf,
    /// The latents through `post_quant`.
    pub(super) quantized: SharedBuf,
    pub(super) planes: [SharedBuf; 2],
    pub(super) inner: SharedBuf,
    pub(super) qkv: SharedBuf,
    pub(super) attention: SharedBuf,
    pub(super) statistics: SharedBuf,
    /// The image out, pixel-major, `outputs` channels of every pixel.
    pub image: SharedBuf,
    pub(super) patches: PerThread<Patches>,
    pub(super) scratch: PerThread<attention::Scratch>,
}
