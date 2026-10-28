//! The causal convolution of the streaming audio blocks (`seanet`, `code2wav`): output row `t`
//! is one matrix times the input rows `t − (window − 1) · dilation ..= t` spaced `dilation`
//! apart, oldest first (the rows before the first are the history, zeros at the start).
//!
//! A transposed conv with kernel `w · s` and stride `s` is such a conv with window `w` whose
//! output row holds the next `s` time steps (its `s` phases), so a row's outputs are final as
//! soon as it is computed: no overlap-add, no pending partial sums. The compiler lays out each
//! matrix as `[phases · out, window · in]` (rows phase-major, columns oldest input first).

use std::ops::Range;

use kernels::{COLUMN_ALIGN, Linear, QuantizedRows};
use team::{Ctx, SharedBuf};

/// A causal convolution as one matrix: output row `t` holds `phases` time steps of `outputs`
/// channels, from `window` input rows of `inputs` channels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CausalConv {
    pub(crate) window: usize,
    pub(crate) phases: usize,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    /// Input rows between one tap and the next.
    pub(crate) dilation: usize,
}

impl CausalConv {
    /// Values of one output row.
    pub(crate) fn row(&self) -> usize {
        self.phases * self.outputs
    }

    /// Input rows the conv keeps between calls: the history.
    pub(crate) fn kept(&self) -> usize {
        (self.window - 1) * self.dilation
    }
}

/// One thread's working memory for `project`.
#[derive(Default)]
pub(crate) struct Scratch {
    window: Vec<f32>,
    quant: QuantizedRows,
}

/// One conv's call: `rows` input rows of `input` after its `history`, the results into `target`
/// (`conv.row()` values per row).
pub(crate) struct Product<'call> {
    pub(crate) conv: &'call CausalConv,
    pub(crate) linear: &'call Linear,
    pub(crate) rows: usize,
    pub(crate) history: &'call SharedBuf,
    pub(crate) input: &'call SharedBuf,
    pub(crate) target: &'call SharedBuf,
}

/// This thread's share of a conv: split by output columns while there are fewer rows than
/// columns (whole units of `COLUMN_ALIGN`) and by rows after. Writes the results of the rows and
/// columns it returns.
pub(crate) fn project(
    ctx: &Ctx,
    product: &Product,
    scratch: &mut Scratch,
) -> (Range<usize>, Range<usize>) {
    let conv = product.conv;
    let target = product.target;
    project_windows(
        ctx,
        (product.linear, product.rows),
        scratch,
        |row, window| gather(product, row, window),
        |row, col, value| target.set(row * conv.row() + col, value),
    )
}

/// The product of `linear` over `rows` windows (`gather(row, window)` fills a row's), this
/// thread's share of it, stored by `store(row, col, value)`. Returns the rows and columns the
/// thread stored.
pub(crate) fn project_windows(
    ctx: &Ctx,
    (linear, rows): (&Linear, usize),
    scratch: &mut Scratch,
    gather: impl Fn(usize, &mut [f32]),
    store: impl Fn(usize, usize, f32),
) -> (Range<usize>, Range<usize>) {
    let matrix = linear.matrix();
    let by_columns = rows < matrix.rows && matrix.rows.is_multiple_of(COLUMN_ALIGN);
    let (mine, cols) = if by_columns {
        (0..rows, ctx.split(matrix.rows, COLUMN_ALIGN))
    } else {
        (ctx.split(rows, 1), 0..matrix.rows)
    };
    scratch.window.resize(mine.len() * matrix.cols, 0.0);
    for (row, window) in mine
        .clone()
        .zip(scratch.window.chunks_exact_mut(matrix.cols))
    {
        gather(row, window);
    }
    let input = scratch.quant.input(&matrix, &scratch.window);
    let first = mine.start;
    linear.project(&input, cols.clone(), |row, col, value| {
        store(first + row, col, value);
    });
    (mine, cols)
}

/// The input rows a conv's output row `row` reads, oldest first, from its history and its input.
fn gather(product: &Product, row: usize, window: &mut [f32]) {
    let conv = product.conv;
    let (history, input) = (product.history.read(), product.input.read());
    let kept = conv.kept();
    for (tap, values) in window.chunks_exact_mut(conv.inputs).enumerate() {
        let source = row + tap * conv.dilation;
        let rows = if source < kept {
            &history[source * conv.inputs..]
        } else {
            &input[(source - kept) * conv.inputs..]
        };
        values.copy_from_slice(&rows[..conv.inputs]);
    }
}

/// Makes the last `conv.kept()` rows of the history followed by the first `rows` rows of `input`
/// the conv's history.
pub(crate) fn keep(conv: &CausalConv, input: &SharedBuf, rows: usize, history: &SharedBuf) {
    let kept = conv.kept() * conv.inputs;
    let end = rows * conv.inputs;
    let history = history.part(0..kept);
    let input = &input.read()[..end];
    match end.checked_sub(kept) {
        Some(skipped) => history.copy_from_slice(&input[skipped..]),
        None => {
            history.copy_within(end.., 0);
            history[kept - end..].copy_from_slice(input);
        }
    }
}
