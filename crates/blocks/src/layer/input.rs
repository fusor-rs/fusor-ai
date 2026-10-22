//! A projection's input: the rows it reads, normed or as they are, prepared for its weights by
//! every thread's share and read back once all are in.

use kernels::{Input, Linear, MatrixStorage, Quantized, quantize_rows};
use team::SharedBuf;

use super::{Norm, Pass};

/// The rows a projection reads: the residual stream through a norm, or a buffer as it is.
#[derive(Clone, Copy)]
pub(super) enum Source<'layer> {
    Normed(&'layer Norm),
    Buffer(&'layer SharedBuf),
}

/// This thread's share of `source` prepared as `linear`'s input, into the shared buffers:
/// quantized for quantized weights, else normed (a buffer is its own f32 input). The share is
/// of the quantization groups of all the rows (whole rows for f32 weights), so that one row is
/// shared out as evenly as thousands; a row's norm is computed by every thread holding part of
/// it.
pub(super) fn prepare(pass: &Pass, source: Source, linear: &Linear) {
    let (buffers, matrix) = (pass.buffers, linear.matrix());
    let width = matrix.cols;
    let unit = match matrix.storage {
        MatrixStorage::F32(_) => width,
        MatrixStorage::Quant { format, .. } => format.group(),
    };
    let per_row = width / unit;
    let share = pass.ctx.split(pass.rows * per_row, 1);
    let scratch = buffers.scratch.get(pass.ctx);
    scratch.normed.resize(width, 0.0);
    for row in share.start / per_row..share.end.div_ceil(per_row) {
        let units = share.start.max(row * per_row)..share.end.min((row + 1) * per_row);
        let values = units.start * unit..units.end * unit;
        let within = values.start - row * width..values.end - row * width;
        let source_row: &[f32] = match source {
            Source::Normed(norm) => {
                let hidden = &buffers.hidden.read()[row * width..(row + 1) * width];
                norm.apply(hidden, &mut scratch.normed);
                &scratch.normed
            }
            Source::Buffer(buffer) => &buffer.read()[row * width..(row + 1) * width],
        };
        match (matrix.storage, source) {
            (MatrixStorage::F32(_), Source::Normed(_)) => {
                buffers
                    .normed
                    .part(values)
                    .copy_from_slice(&source_row[within]);
            }
            (MatrixStorage::F32(_), Source::Buffer(_)) => {}
            (MatrixStorage::Quant { format, .. }, _) => quantize_rows(
                &source_row[within],
                format,
                buffers.quants.part(values),
                buffers.scales.part(units.clone()),
                buffers.corrections.part(units),
            ),
        }
    }
}

/// The input `prepare` made for `linear` over the call's rows, once every thread's share is in
/// (after a barrier).
pub(super) fn input<'pass>(
    pass: &Pass<'pass>,
    source: Source<'pass>,
    linear: &Linear,
) -> Input<'pass> {
    let (buffers, matrix) = (pass.buffers, linear.matrix());
    let len = pass.rows * matrix.cols;
    match (matrix.storage, source) {
        (MatrixStorage::F32(_), Source::Normed(_)) => Input::F32(&buffers.normed.read()[..len]),
        (MatrixStorage::F32(_), Source::Buffer(buffer)) => Input::F32(&buffer.read()[..len]),
        (MatrixStorage::Quant { format, .. }, _) => {
            let groups = len / format.group();
            Input::Quant(Quantized {
                format,
                quants: &buffers.quants.read()[..len],
                scales: &buffers.scales.read()[..groups],
                corrections: &buffers.corrections.read()[..groups],
            })
        }
    }
}
