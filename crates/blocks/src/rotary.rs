//! Rotary position embeddings whose frequencies turn by different axes of a position: the
//! sequence position of a text model, the temporal, height and width positions of Qwen2-VL's
//! multimodal RoPE, the row and column of a vision patch, or a video patch's frame, row and
//! column.
//!
//! A text row of a multimodal model has the same value on every axis, which turns every
//! frequency as a one-dimensional RoPE would. Height and width positions may be scaled (Position
//! Interpolation).

use kernels::ops::rope_inv_freq;

/// An error unless `hidden` splits into `heads` and the three axes' `sections` cover half a head.
pub(crate) fn check_axes(
    prefix: &str,
    (hidden, heads): (usize, usize),
    sections: [usize; 3],
) -> Result<(), bundle::Error> {
    let covered: usize = sections.iter().sum();
    if hidden.is_multiple_of(heads) && covered == hidden / heads / 2 {
        return Ok(());
    }
    Err(bundle::Error::Format(format!(
        "{prefix}: {heads} heads and rotary sections {sections:?} for a width of {hidden}"
    )))
}

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
    /// `Sequence`'s frequencies divided by this factor: positions interpolated linearly
    /// (`rope_type="linear"`, Gemma 3's and T5Gemma2's full attention layers).
    Linear(usize),
    /// `Sequence`'s frequencies, only the first this many turning and the others 0: the tail of
    /// each half of a head stays as it is (LFM3's partial rotary embedding).
    Partial(usize),
    /// Qwen2-VL's multimodal RoPE: the frequencies of `Sequence` in three consecutive sections
    /// of these lengths, turned by the temporal, height and width positions.
    Multimodal([usize; 3]),
    /// Three consecutive sections of these many frequencies, each its own rotary embedding over
    /// `2 · len` elements (`base^(−2i/(2·len))`), turned by the temporal, height and width
    /// positions (the 3-D rotary embeddings of Wan's and Motif-Video's video transformers).
    Axes([usize; 3]),
    /// Qwen3-VL's interleaved multimodal RoPE: frequency `i` turned by the temporal, height and
    /// width positions in turn (`i % 3`), the height and width ones only within the first
    /// `3 · sections[1]` and `3 · sections[2]` frequencies, the temporal one elsewhere.
    MultimodalInterleaved([usize; 3]),
    /// A vision patch's: the first half of the frequencies by its row, the second by its column,
    /// each half `base^(−2i/(head_dim/2))`.
    Grid,
}

/// The axis of each of `half` frequencies split in consecutive sections of these lengths: the
/// temporal, height and width positions turn one section each.
fn section_axes(sections: [usize; 3], half: usize) -> Vec<Axis> {
    assert_eq!(
        sections.iter().sum::<usize>(),
        half,
        "the sections cover every frequency"
    );
    [Axis::Temporal, Axis::Height, Axis::Width]
        .into_iter()
        .zip(sections)
        .flat_map(|(axis, len)| std::iter::repeat_n(axis, len))
        .collect()
}

/// The frequencies of one head and the axis each turns by; the default has none, for a stack
/// whose layers have no rotary embeddings.
#[derive(Debug, Clone, Default)]
pub struct Rotary {
    inv_freq: Vec<f32>,
    axes: Vec<Axis>,
}

impl Rotary {
    /// The frequencies of the first `head_dim` elements of a head (all of it, or the rotated part
    /// of a partial rotary embedding).
    ///
    /// # Panics
    /// When a multimodal or axes rotation's sections do not add up to `head_dim / 2`.
    pub fn new(rotation: Rotation, base: f32, head_dim: usize) -> Self {
        let half = head_dim / 2;
        let (inv_freq, axes) = match rotation {
            Rotation::Sequence => (rope_inv_freq(base, head_dim), vec![Axis::Temporal; half]),
            Rotation::Linear(factor) => {
                let inv_freq = rope_inv_freq(base, head_dim)
                    .into_iter()
                    .map(|frequency| frequency / factor as f32)
                    .collect();
                (inv_freq, vec![Axis::Temporal; half])
            }
            Rotation::Partial(turning) => {
                let mut inv_freq = rope_inv_freq(base, head_dim);
                inv_freq[turning..].fill(0.0);
                (inv_freq, vec![Axis::Temporal; half])
            }
            Rotation::Multimodal(sections) => {
                (rope_inv_freq(base, head_dim), section_axes(sections, half))
            }
            Rotation::Axes(sections) => {
                let inv_freq = sections
                    .iter()
                    .flat_map(|&len| rope_inv_freq(base, 2 * len))
                    .collect();
                (inv_freq, section_axes(sections, half))
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
                return Self::new(Rotation::Axes([0, half / 2, half / 2]), base, head_dim);
            }
        };
        Self { inv_freq, axes }
    }

    /// The same rotation with the height and width positions multiplied by `scale` (Position
    /// Interpolation: P-JiT's 1024-pixel grid turns at half its positions, as its 512-pixel grid
    /// did).
    pub fn spatially_scaled(mut self, scale: f32) -> Self {
        for (frequency, axis) in self.inv_freq.iter_mut().zip(&self.axes) {
            if matches!(axis, Axis::Height | Axis::Width) {
                *frequency *= scale;
            }
        }
        self
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

#[cfg(test)]
mod tests {
    use super::{Position, Rotary, Rotation};

    #[test]
    fn axes_turn_each_section_by_its_own_frequencies() {
        let rotary = Rotary::new(Rotation::Axes([2, 1, 1]), 10_000.0, 8);
        let position = Position {
            temporal: 1,
            height: 2,
            width: 3,
        };
        let (mut cos, mut sin) = ([0.0; 4], [0.0; 4]);
        rotary.angles(position, &mut cos, &mut sin);
        let want_cos = [0.540_302_3, 0.999_95, -0.416_146_84, -0.989_992_5];
        let want_sin = [0.841_470_96, 0.009_999_833, 0.909_297_4, 0.141_120_01];
        for (got, want) in cos.iter().chain(&sin).zip(want_cos.iter().chain(&want_sin)) {
            assert!((got - want).abs() < 1e-6, "{got} against {want}");
        }
    }

    #[test]
    fn spatial_scale_turns_height_and_width_at_scaled_positions() {
        let rotary = Rotary::new(Rotation::Axes([2, 1, 1]), 10_000.0, 8).spatially_scaled(0.5);
        let (mut cos, mut sin) = ([0.0; 4], [0.0; 4]);
        let position = Position {
            temporal: 1,
            height: 2,
            width: 3,
        };
        rotary.angles(position, &mut cos, &mut sin);
        let want_cos = [0.540_302_3, 0.999_95, 0.540_302_3, 0.070_737_2];
        let want_sin = [0.841_470_96, 0.009_999_833, 0.841_470_96, 0.997_495];
        for (got, want) in cos.iter().chain(&sin).zip(want_cos.iter().chain(&want_sin)) {
            assert!((got - want).abs() < 1e-6, "{got} against {want}");
        }
    }

    #[test]
    fn linear_rotation_divides_every_frequency() {
        let rotary = Rotary::new(Rotation::Linear(8), 10_000.0, 4);
        let (mut cos, mut sin) = ([0.0; 2], [0.0; 2]);
        rotary.angles(Position::uniform(4), &mut cos, &mut sin);
        let want_cos = [0.877_582_56, 0.999_987_5];
        let want_sin = [0.479_425_55, 0.004_999_979];
        for (got, want) in cos.iter().chain(&sin).zip(want_cos.iter().chain(&want_sin)) {
            assert!((got - want).abs() < 1e-6, "{got} against {want}");
        }
    }
}
