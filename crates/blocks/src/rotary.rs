//! Rotary position embeddings whose frequencies turn by different axes of a position: the
//! sequence position of a text model, the temporal, height and width positions of Qwen2-VL's
//! multimodal RoPE, or the row and column of a vision patch.
//!
//! A text row of a multimodal model has the same value on every axis, which turns every
//! frequency as a one-dimensional RoPE would.

use kernels::ops::rope_inv_freq;

/// A row's coordinates: temporal (or its sequence position), height and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl Position {
    /// A text row's: `index` on every axis.
    pub fn uniform(index: usize) -> Self {
        Self {
            temporal: index,
            height: index,
            width: index,
        }
    }

    fn along(self, axis: Axis) -> usize {
        match axis {
            Axis::Temporal => self.temporal,
            Axis::Height => self.height,
            Axis::Width => self.width,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Temporal,
    Height,
    Width,
}

/// Which axis turns each rotary frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    /// Every frequency `base^(−2i/head_dim)` by the sequence position.
    Sequence,
    /// `Sequence`'s frequencies, only the first this many turning and the others 0: the tail of
    /// each half of a head stays as it is (LFM3's partial rotary embedding).
    Partial(usize),
    /// Qwen2-VL's multimodal RoPE: the frequencies of `Sequence` in three consecutive sections
    /// of these lengths, turned by the temporal, height and width positions.
    Multimodal([usize; 3]),
    /// Qwen3-VL's interleaved multimodal RoPE: frequency `i` turned by the temporal, height and
    /// width positions in turn (`i % 3`), the height and width ones only within the first
    /// `3 · sections[1]` and `3 · sections[2]` frequencies, the temporal one elsewhere.
    MultimodalInterleaved([usize; 3]),
    /// A vision patch's: the first half of the frequencies by its row, the second by its column,
    /// each half `base^(−2i/(head_dim/2))`.
    Grid,
}

/// The frequencies of one head and the axis each turns by.
#[derive(Debug, Clone)]
pub struct Rotary {
    inv_freq: Vec<f32>,
    axes: Vec<Axis>,
}

impl Rotary {
    /// The frequencies of the first `head_dim` elements of a head (all of it, or the rotated part
    /// of a partial rotary embedding).
    ///
    /// # Panics
    /// When a multimodal rotation's sections do not add up to `head_dim / 2`.
    pub fn new(rotation: Rotation, base: f32, head_dim: usize) -> Self {
        let half = head_dim / 2;
        let (inv_freq, axes) = match rotation {
            Rotation::Sequence => (rope_inv_freq(base, head_dim), vec![Axis::Temporal; half]),
            Rotation::Partial(turning) => {
                let mut inv_freq = rope_inv_freq(base, head_dim);
                inv_freq[turning..].fill(0.0);
                (inv_freq, vec![Axis::Temporal; half])
            }
            Rotation::Multimodal(sections) => {
                assert_eq!(
                    sections.iter().sum::<usize>(),
                    half,
                    "the sections cover every frequency"
                );
                let axes = [Axis::Temporal, Axis::Height, Axis::Width]
                    .into_iter()
                    .zip(sections)
                    .flat_map(|(axis, len)| std::iter::repeat_n(axis, len));
                (rope_inv_freq(base, head_dim), axes.collect())
            }
            Rotation::MultimodalInterleaved([_, height, width]) => {
                let axes = (0..half).map(|i| match i % 3 {
                    1 if i < 3 * height => Axis::Height,
                    2 if i < 3 * width => Axis::Width,
                    _ => Axis::Temporal,
                });
                (rope_inv_freq(base, head_dim), axes.collect())
            }
            Rotation::Grid => {
                let quarter = rope_inv_freq(base, half);
                let axes = [vec![Axis::Height; half / 2], vec![Axis::Width; half / 2]].concat();
                ([quarter.as_slice(), &quarter].concat(), axes)
            }
        };
        Self { inv_freq, axes }
    }

    /// Cos and sin of each frequency's angle at `position` (angles in f32).
    pub fn angles(&self, position: Position, cos: &mut [f32], sin: &mut [f32]) {
        let turns = self.inv_freq.iter().zip(&self.axes);
        for ((frequency, &axis), (cosine, sine)) in turns.zip(cos.iter_mut().zip(sin)) {
            let angle = position.along(axis) as f32 * frequency;
            *cosine = angle.cos();
            *sine = angle.sin();
        }
    }
}
