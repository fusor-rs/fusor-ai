//! A video's latents (frame-major, row-major, channels last) and its tokens: patches one frame
//! deep and `patch × patch` latent pixels wide, frame-major then row-major (Wan's and
//! Motif-Video's transformers). A side that is not a whole number of patches is padded with zeros
//! past its right or bottom edge (P-JiT's images): `gather` reads zeros there and `scatter` drops
//! those pixels.

use team::SharedBuf;

use crate::rotary::Position;

/// A video's latents: `frames` frames of `height × width` latent pixels, frame-major, each pixel
/// its channels (channels last).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Video {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl Video {
    pub fn pixels(&self) -> usize {
        self.frames * self.height * self.width
    }
}

/// A video cut into tokens of `patch × patch` pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Patching {
    pub video: Video,
    pub patch: usize,
}

impl Patching {
    /// Patch rows and patch columns of one frame.
    fn grid(&self) -> (usize, usize) {
        (
            self.video.height.div_ceil(self.patch),
            self.video.width.div_ceil(self.patch),
        )
    }

    pub fn tokens(&self) -> usize {
        let (rows, columns) = self.grid();
        self.video.frames * rows * columns
    }

    /// Token `token`'s frame, patch row and patch column.
    pub fn position(&self, token: usize) -> Position {
        let (rows, columns) = self.grid();
        Position {
            temporal: token / (rows * columns),
            height: token / columns % rows,
            width: token % columns,
        }
    }

    /// The latent pixel at `offset` (row-major) inside token `token`'s patch, or `None` past the
    /// video's edges.
    fn pixel(&self, token: usize, offset: usize) -> Option<usize> {
        let position = self.position(token);
        let row = position.height * self.patch + offset / self.patch;
        let column = position.width * self.patch + offset % self.patch;
        if row >= self.video.height || column >= self.video.width {
            return None;
        }
        Some((position.temporal * self.video.height + row) * self.video.width + column)
    }

    /// Token `token`'s patch of `latents` (`channels` per pixel) into `out`: its pixels
    /// row-major, a pixel's channels contiguous.
    pub fn gather(&self, (latents, channels): (&[f32], usize), token: usize, out: &mut [f32]) {
        for (offset, pixel_values) in out.chunks_exact_mut(channels).enumerate() {
            match self.pixel(token, offset) {
                Some(pixel) => {
                    pixel_values
                        .copy_from_slice(&latents[pixel * channels..(pixel + 1) * channels]);
                }
                None => pixel_values.fill(0.0),
            }
        }
    }

    /// The inverse: `values` (a patch as `gather` lays it out) into their pixels of `latents`.
    pub fn scatter(&self, (values, channels): (&[f32], usize), token: usize, latents: &SharedBuf) {
        for (offset, pixel_values) in values.chunks_exact(channels).enumerate() {
            let Some(pixel) = self.pixel(token, offset) else {
                continue;
            };
            let target = latents.part(pixel * channels..(pixel + 1) * channels);
            target.copy_from_slice(pixel_values);
        }
    }
}

#[cfg(test)]
mod tests {
    use team::SharedBuf;

    use super::{Patching, Video};

    #[test]
    fn padding_reads_zeros_past_the_edges_and_scatter_drops_them() {
        let patching = Patching {
            video: Video {
                frames: 1,
                height: 3,
                width: 3,
            },
            patch: 2,
        };
        assert_eq!(patching.tokens(), 4);
        let latents: Vec<f32> = (1..=9).map(|value| value as f32).collect();
        let mut patch = [7.0; 4];
        patching.gather((&latents, 1), 3, &mut patch);
        assert_eq!(patch, [9.0, 0.0, 0.0, 0.0]);
        let target = SharedBuf::new(9);
        patching.scatter((&[5.0, 6.0, 7.0, 8.0], 1), 3, &target);
        assert_eq!(
            target.read(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
        );
    }
}
