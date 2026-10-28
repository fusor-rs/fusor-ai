//! The gated short convolution of LFM2 and LFM3, and the gated 2-D convolution of LFM3-VL's
//! vision tower: the input projection's three parts, an input gate `b`, an output gate `c` and
//! the values `x`, then `c · conv(b · x)`, a depthwise convolution whose `Reach` is a causal one
//! over each channel's last `kernel` rows, carried from call to call, or the `kernel × kernel`
//! patches of an image's grid around each.
//!
//! Tensors under a layer's prefix `p`: `p.input` (`b`, `c` then `x`, each `hidden` wide),
//! `p.conv` (each channel's taps, row-major over a 2-D kernel) and `p.out`.
//!
//! `mix` is two phases: the projection by columns; the convolution by channels.

use bundle::{Scope, Store};
use kernels::{COLUMN_ALIGN, Input, Linear, QUAD, Table};
use team::SharedBuf;

use crate::layer::Pass;
use crate::vit::Grid;

/// The parts of the input projection, in its order, each `hidden` wide.
#[derive(Clone, Copy)]
enum Part {
    InputGate,
    OutputGate,
    Value,
}

const PARTS: [Part; 3] = [Part::InputGate, Part::OutputGate, Part::Value];

/// The input projection's width.
pub(crate) fn input_width(hidden: usize) -> usize {
    PARTS.len() * hidden
}

pub(crate) struct Weights {
    pub(crate) input: Linear,
    /// The convolution's taps, `hidden × taps_per_channel`.
    taps: Vec<f32>,
    pub(crate) out: Linear,
    hidden: usize,
    taps_per_channel: usize,
}

/// The inputs a conv layer of `kernel` taps keeps for the next call, per channel.
pub(super) fn history(kernel: usize) -> usize {
    kernel - 1
}

/// The inputs each output of a call's convolution sees.
pub(crate) enum Reach<'call> {
    /// Its channel's last `taps_per_channel` rows, its own included: the call's, and before
    /// them layer `slot`'s of `kept`, every causal layer's last `taps_per_channel − 1` inputs
    /// (channel-major, oldest first).
    Causal { kept: &'call SharedBuf, slot: usize },
    /// The patches of a `kernel × kernel` square of the call's rows, row-major over `Grid`,
    /// centred on its own, zero past the grid's edges (`taps_per_channel` is `kernel²`).
    Grid(Grid),
}

impl Weights {
    pub(crate) fn load(
        store: &mut Store,
        prefix: &str,
        hidden: usize,
        taps_per_channel: usize,
    ) -> Result<Self, bundle::Error> {
        let mut scope = Scope::new(store, prefix);
        let taps: Table = scope.load("conv", [hidden, taps_per_channel])?;
        Ok(Self {
            input: scope.load("input", [input_width(hidden), hidden])?,
            taps: taps.dequantize(),
            out: scope.load("out", [hidden, hidden])?,
            hidden,
            taps_per_channel,
        })
    }

    /// `buffers.attention`: the gated convolution of the normed rows `input` (prepared for
    /// `input`).
    pub(crate) fn mix(&self, pass: &Pass, input: &Input, reach: &Reach) {
        let width = input_width(self.hidden);
        let projected = &pass.buffers.qkv;
        let cols = pass.ctx.split(width, COLUMN_ALIGN);
        self.input.project_quads(input, cols, |row, first, values| {
            let at = row * width + first;
            projected.part(at..at + QUAD).copy_from_slice(&values);
        });
        pass.ctx.barrier();
        match *reach {
            Reach::Causal { kept, slot } => self.convolve(pass, kept, slot),
            Reach::Grid(grid) => self.convolve_grid(pass, grid),
        }
        pass.ctx.barrier();
    }

    /// Part `part` of channel `channel` of row `row` of the projection.
    fn part(&self, projected: &[f32], row: usize, part: Part, channel: usize) -> f32 {
        projected[row * input_width(self.hidden) + part as usize * self.hidden + channel]
    }

    /// `b · x` of channel `channel` of row `row` of the projection.
    fn gated(&self, projected: &[f32], row: usize, channel: usize) -> f32 {
        self.part(projected, row, Part::InputGate, channel)
            * self.part(projected, row, Part::Value, channel)
    }

    /// This thread's channels of `buffers.attention`: `c` times the causal convolution of
    /// `b · x` over the kept inputs and the call's rows; then the last inputs kept.
    fn convolve(&self, pass: &Pass, kept: &SharedBuf, slot: usize) {
        let (hidden, kernel) = (self.hidden, self.taps_per_channel);
        let history = history(kernel);
        let mine = pass.ctx.split(hidden, 1);
        let projected = pass.buffers.qkv.read();
        let first = slot * hidden * history;
        let kept = kept.part(first + mine.start * history..first + mine.end * history);
        let input = |kept: &[f32], channel: usize, at: usize| match at.checked_sub(history) {
            Some(row) => self.gated(projected, row, channel),
            None => kept[at],
        };
        for row in 0..pass.rows {
            let at = row * hidden;
            let out = pass.buffers.attention.part(at + mine.start..at + mine.end);
            for (offset, channel) in mine.clone().enumerate() {
                let taps = &self.taps[channel * kernel..(channel + 1) * kernel];
                let kept = &kept[offset * history..(offset + 1) * history];
                let sum = (0..kernel).map(|tap| taps[tap] * input(kept, channel, row + tap));
                let gate = self.part(projected, row, Part::OutputGate, channel);
                out[offset] = gate * sum.sum::<f32>();
            }
        }
        for (offset, channel) in mine.enumerate() {
            let kept = &mut kept[offset * history..(offset + 1) * history];
            for slot in 0..history {
                kept[slot] = input(kept, channel, pass.rows + slot);
            }
        }
    }

    /// This thread's channels of `buffers.attention`: `c` times the convolution of `b · x` over
    /// the square of patches around each patch of `grid`.
    fn convolve_grid(&self, pass: &Pass, grid: Grid) {
        let hidden = self.hidden;
        let mine = pass.ctx.split(hidden, 1);
        let projected = pass.buffers.qkv.read();
        for row in 0..grid.patches() {
            let at = row * hidden;
            let out = pass.buffers.attention.part(at + mine.start..at + mine.end);
            for (offset, channel) in mine.clone().enumerate() {
                let gate = self.part(projected, row, Part::OutputGate, channel);
                out[offset] = gate * self.grid_sum(projected, grid, row, channel);
            }
        }
    }

    /// Channel `channel`'s convolution of `b · x` over the square of patches around patch `row`
    /// of `grid`, those past its edges zero.
    fn grid_sum(&self, projected: &[f32], grid: Grid, row: usize, channel: usize) -> f32 {
        let kernel = self.taps_per_channel.isqrt();
        let reach = kernel / 2;
        let taps = &self.taps[channel * self.taps_per_channel..][..self.taps_per_channel];
        let (top, left) = (row / grid.columns, row % grid.columns);
        let columns = left.saturating_sub(reach)..(left + reach + 1).min(grid.columns);
        let mut sum = 0.0;
        for y in top.saturating_sub(reach)..(top + reach + 1).min(grid.rows) {
            let tap_row = &taps[(y + reach - top) * kernel..][..kernel];
            for x in columns.clone() {
                let input = self.gated(projected, y * grid.columns + x, channel);
                sum += tap_row[x + reach - left] * input;
            }
        }
        sum
    }
}
