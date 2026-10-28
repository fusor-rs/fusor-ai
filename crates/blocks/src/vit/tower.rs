//! A NaViT vision tower over one image's patches: each patch's embedding (a linear over its
//! pixels) plus the learned position grid resampled to the image's grid, the encoder, and, where
//! the model has one, a final LayerNorm.
//!
//! Tensors under a prefix `p`: `p.patch` (a linear), `p.positions` (`side² × hidden`, row-major),
//! the encoder's (`vit`) and, where it ends with one, `p.norm`.

use bundle::{Load, Scope, Store};
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows, Table};
use media::image::{Filter, Size, resize_bicubic, resize_bilinear, resize_bilinear_corners};
use team::{Ctx, PerThread};

use super::{Grid, Vit};
use crate::norm::LayerNorm;

/// What the tower ends with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// The encoder's last layer, as it is.
    Encoder,
    /// A LayerNorm over the encoder's output (the encoder's `norm_eps`).
    Norm,
}

/// How the position grid is resampled to an image's grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// PyTorch's interpolation of that name, its corner cells within the image's.
    Interpolate(Filter),
    /// Bilinearly, its corner cells on the image's corner patches (Qwen3-VL's).
    Corners,
}

/// The dimensions of one tower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub encoder: super::Shape,
    /// Values of one patch: its pixels, every channel.
    pub patch_values: usize,
    /// Side of the learned position grid, in patches.
    pub position_grid: usize,
    pub resample: Resample,
    pub ending: Ending,
}

pub struct Tower {
    shape: Shape,
    patch: Linear,
    /// The position grid, `position_grid² × hidden`.
    positions: Vec<f32>,
    encoder: Vit,
    norm: Option<LayerNorm>,
}

/// Activations of one image of up to `patches` patches.
pub struct Buffers {
    /// The patches' rows, `patches × hidden`: the encoder's input, then the tower's output.
    pub encoder: super::Buffers,
    pub quant: PerThread<QuantizedRows>,
    /// A row on its way through the final norm.
    row: PerThread<Vec<f32>>,
}

impl Buffers {
    pub fn new(shape: &Shape, patches: usize, threads: usize) -> Self {
        Self::around(shape.encoder.buffers(patches, threads), threads)
    }

    /// The buffers of a tower whose encoder's activations are `encoder`.
    pub(crate) fn around(encoder: super::Buffers, threads: usize) -> Self {
        Self {
            encoder,
            quant: PerThread::new(threads, QuantizedRows::default),
            row: PerThread::new(threads, Vec::new),
        }
    }

    /// This thread's rows of the first `patches` of `encoder.hidden`, each through `norm`, in
    /// place; then a barrier.
    pub(crate) fn norm_rows(&self, ctx: &Ctx, patches: usize, norm: impl Fn(&[f32], &mut [f32])) {
        let width = self.encoder.hidden_width;
        let rows = ctx.split(patches, 1);
        let hidden = self
            .encoder
            .hidden
            .part(rows.start * width..rows.end * width);
        let copy = self.row.get(ctx);
        for row in hidden.chunks_exact_mut(width) {
            copy.clear();
            copy.extend_from_slice(row);
            norm(copy, row);
        }
        ctx.barrier();
    }
}

impl Tower {
    /// The position grid resampled to `grid`, `grid.patches() × hidden`.
    pub fn positions(&self, grid: Grid) -> Vec<f32> {
        let side = Size {
            width: self.shape.position_grid,
            height: self.shape.position_grid,
        };
        let to = Size {
            width: grid.columns,
            height: grid.rows,
        };
        let resize = match self.shape.resample {
            Resample::Interpolate(Filter::Bilinear) => resize_bilinear,
            Resample::Interpolate(Filter::Bicubic) => resize_bicubic,
            Resample::Corners => resize_bilinear_corners,
        };
        resize(&self.positions, side, to, self.shape.encoder.hidden)
    }

    /// `encoder.hidden` = each patch's embedding plus its position, for this thread's columns.
    pub fn embed(&self, ctx: &Ctx, buffers: &Buffers, patches: &[f32], positions: &[f32]) {
        let width = self.shape.encoder.hidden;
        let input = buffers.quant.get(ctx).input(&self.patch.matrix(), patches);
        let cols = ctx.split(width, COLUMN_ALIGN);
        let hidden = &buffers.encoder.hidden;
        self.patch.project(&input, cols, |row, col, value| {
            let at = row * width + col;
            hidden.set(at, value + positions[at]);
        });
    }

    /// The encoder over `encoder.hidden`, then the final norm of this thread's rows, in place.
    pub fn encode(&self, ctx: &Ctx, buffers: &Buffers, grid: Grid) {
        self.encoder.forward(ctx, &buffers.encoder, grid);
        let Some(norm) = &self.norm else {
            return;
        };
        let eps = self.shape.encoder.norm_eps;
        buffers.norm_rows(ctx, grid.patches(), |row, out| norm.apply(row, eps, out));
    }
}

impl Load for Tower {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let width = shape.encoder.hidden;
        let mut scope = Scope::new(store, prefix);
        let positions: Table = scope.load(
            "positions",
            [shape.position_grid * shape.position_grid, width],
        )?;
        let patch = scope.load("patch", [width, shape.patch_values])?;
        let norm = match shape.ending {
            Ending::Encoder => None,
            Ending::Norm => Some(scope.load("norm", width)?),
        };
        Ok(Self {
            shape,
            patch,
            positions: positions.dequantize(),
            encoder: Vit::load(store, prefix, shape.encoder)?,
            norm,
        })
    }
}
