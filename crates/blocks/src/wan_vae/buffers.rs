//! The activations of `Decoder`'s and `Encoder`'s calls and the causal convolutions' histories,
//! sized by the steps a call runs.

use kernels::conv::Patches;
use team::{PerThread, SharedBuf};

use super::Shape;
use crate::attention;

/// What a decoder's buffers hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// Latent pixels of a frame.
    pub pixels: usize,
    /// Latent columns of a frame.
    pub width: usize,
    /// Latent pixels of every latent frame of a video.
    pub video: usize,
    /// Latent rows of a strip.
    pub strip: usize,
}

impl Shape {
    /// Pixels level `level` computes for a strip at most: the strip's rows and the halo on either
    /// side, at most a frame's.
    fn window_pixels(&self, capacity: Capacity, halos: &[usize], level: usize) -> usize {
        let rows = (capacity.strip << level) + 2 * halos[level];
        (capacity.pixels << (2 * level)).min((capacity.width << level) * rows)
    }

    /// Values a decoder's plane holds at most: the middle's attention's fused projection over a
    /// frame, then every level's step over a strip's window, the time convolutions' doubled
    /// channels included.
    fn plane_values(&self, capacity: Capacity, halos: &[usize]) -> usize {
        let window = |level| self.window_pixels(capacity, halos, level) * self.level_frames(level);
        let mut largest = 3 * self.channels[0] * capacity.pixels;
        for level in 0..self.levels() {
            let channels = self.channels[level];
            let time = if self.has_time(level) {
                2 * channels
            } else {
                channels
            };
            largest = largest.max(self.level_inputs(level).max(time) * window(level));
            if level + 1 < self.levels() {
                largest = largest.max(channels / 2 * window(level + 1));
            }
        }
        largest
    }

    /// Values an encoder's plane holds at most: each level's widest step over the frame at its
    /// resolution, and the middle's attention's fused projection.
    fn encoder_plane_values(&self, pixels: usize) -> usize {
        let last = self.levels() - 1;
        let level_values = |level: usize| {
            let read = self.channels[(level + 1).min(last)];
            self.channels[level].max(read) * (pixels << (2 * level))
        };
        (0..self.levels())
            .map(level_values)
            .fold(3 * self.channels[0] * pixels, usize::max)
    }

    /// The values of the histories `Decoder::decode_latents` advances: the convolution in's, then
    /// the middle's four.
    fn latent_history_values(&self, pixels: usize) -> Vec<usize> {
        let history = |inputs: usize| (self.kernel - 1) * inputs * pixels;
        let mut values = vec![history(self.latent)];
        values.extend([history(self.channels[0]); 4]);
        values
    }

    /// The values of the histories `Decoder::decode_strip` advances over a strip's windows, in
    /// the order it runs them: each level's, then the head's.
    fn strip_history_values(&self, capacity: Capacity, halos: &[usize]) -> Vec<usize> {
        let history = |inputs: usize, level: usize| {
            (self.kernel - 1) * inputs * self.window_pixels(capacity, halos, level)
        };
        let mut values = Vec::new();
        for level in 0..self.levels() {
            let channels = self.channels[level];
            for block in 0..self.blocks {
                let inputs = if block == 0 {
                    self.level_inputs(level)
                } else {
                    channels
                };
                values.extend([history(inputs, level), history(channels, level)]);
            }
            if self.has_time(level) {
                values.push(history(channels, level));
            }
        }
        let last = self.levels() - 1;
        values.push(history(self.channels[last], last));
        values
    }

    /// A decoder's activations and histories for `capacity`.
    pub fn decoder_buffers(&self, capacity: Capacity, threads: usize) -> Buffers {
        let halos = self.halos();
        let last = self.levels() - 1;
        let strip = (capacity.pixels << (2 * last))
            .min((capacity.width << last) * (capacity.strip << last));
        let shared = |values: Vec<usize>| values.into_iter().map(SharedBuf::new).collect();
        Buffers {
            capacity,
            latent: SharedBuf::new(capacity.pixels * self.latent),
            frames: SharedBuf::new(self.outputs * strip * self.level_frames(self.levels())),
            middles: SharedBuf::new(capacity.video * self.channels[0]),
            planes: std::array::from_fn(|_| SharedBuf::new(self.plane_values(capacity, &halos))),
            latent_histories: shared(self.latent_history_values(capacity.pixels)),
            strip_histories: shared(self.strip_history_values(capacity, &halos)),
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, attention::Scratch::default),
        }
    }

    /// An encoder's activations for first frames of up to `pixels` latent pixels; they hold no
    /// latent frames to decode.
    pub fn encoder_buffers(&self, pixels: usize, threads: usize) -> Buffers {
        let last = self.levels() - 1;
        Buffers {
            capacity: Capacity {
                pixels,
                width: 0,
                video: 0,
                strip: 0,
            },
            latent: SharedBuf::new(2 * pixels * self.latent),
            frames: SharedBuf::new(self.outputs * (pixels << (2 * last))),
            middles: SharedBuf::new(0),
            planes: std::array::from_fn(|_| SharedBuf::new(self.encoder_plane_values(pixels))),
            latent_histories: Vec::new(),
            strip_histories: Vec::new(),
            patches: PerThread::new(threads, Patches::default),
            scratch: PerThread::new(threads, attention::Scratch::default),
        }
    }
}

/// Activations of a decoder's or an encoder's calls, and what the decoder's next calls remember.
pub struct Buffers {
    pub(super) capacity: Capacity,
    /// One latent frame in (decoding, `latent` channels), or the first frame's moments out
    /// (encoding, `2 · latent`: the means, then the log-variances), pixel-major.
    pub latent: SharedBuf,
    /// A strip of the frames out (decoding: frame-major, the strip's rows, pixel-major,
    /// `outputs` channels, unclamped), or the first frame in (encoding: pixel-major RGB in
    /// [−1, 1]).
    pub frames: SharedBuf,
    /// Each latent frame's middle-block output, which every strip reads.
    pub(super) middles: SharedBuf,
    /// Three pixel-major planes: a step reads one or two and writes another.
    pub(super) planes: [SharedBuf; 3],
    /// Every causal convolution's last two input frames: at the latents' resolution over the
    /// frame, and of the levels and the head over a strip's windows.
    pub(super) latent_histories: Vec<SharedBuf>,
    pub(super) strip_histories: Vec<SharedBuf>,
    pub(super) patches: PerThread<Patches>,
    pub(super) scratch: PerThread<attention::Scratch>,
}
