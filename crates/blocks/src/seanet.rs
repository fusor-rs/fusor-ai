//! SEANet's streaming decoder (EnCodec, Mimi): a causal conv, then per stage an ELU, a
//! transposed conv upsampling by the stage's ratio (kernel `2 · ratio`) and one residual unit
//! (ELU, conv, ELU, 1-wide conv, plus the unit's input), then an ELU and a final conv.
//!
//! Activations are time-major rows of channels. Every conv is a `CausalConv`: output row `t` is
//! one matrix times the input rows `t − window + 1 ..= t` (the rows before the first are the
//! history, zeros at the start). A transposed conv with kernel `2s` and stride `s` is such a conv
//! with window 2 whose output row holds the next `s` time steps (its `s` phases), so a row's
//! outputs are final as soon as it is computed: no overlap-add, no pending partial sums. The
//! compiler lays out each matrix as `[phases · out, window · in]` (rows phase-major, columns
//! oldest input first).
//!
//! `forward` runs inside a team run: one phase per conv, split by output columns while a conv
//! has fewer rows than columns and by rows after, then one phase that moves every conv's history.
//!
//! Tensors under a prefix `p`: `p.init`, `p.stages.{i}.{upsample, residual1, residual2}` and
//! `p.final`, each a linear layer.

use std::ops::Range;

use bundle::{Load, Store};
use kernels::ops::elu;
use kernels::{COLUMN_ALIGN, Linear, QuantizedRows};
use team::{Ctx, PerThread, SharedBuf};

/// The dimensions of one decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Channels of the input rows.
    pub dimension: usize,
    /// Channels of the last stage; stage `i` of `n` starts at `filters · 2^(n − i)`.
    pub filters: usize,
    /// Upsampling ratio of each stage.
    pub ratios: &'static [usize],
    pub kernel: usize,
    pub last_kernel: usize,
    pub residual_kernel: usize,
    /// How much a residual unit narrows its channels.
    pub compress: usize,
    /// Channels of the output.
    pub channels: usize,
}

/// A causal convolution as one matrix (see the module docs): output row `t` holds `phases` time
/// steps of `outputs` channels, from input rows `t − window + 1 ..= t` of `inputs` channels.
#[derive(Debug, Clone, Copy)]
struct CausalConv {
    window: usize,
    phases: usize,
    inputs: usize,
    outputs: usize,
}

impl CausalConv {
    /// Values of one output row.
    fn row(&self) -> usize {
        self.phases * self.outputs
    }
}

/// Where a conv's results go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Output {
    /// ELU of the result into the next conv's input.
    Next,
    /// The result into the skip buffer of `stage` as well.
    NextAndSkip { stage: usize },
    /// The skip buffer of `stage` added to the result first (the end of a residual unit).
    NextPlusSkip { stage: usize },
    /// The decoder's output.
    Last,
}

struct Step {
    conv: CausalConv,
    linear: Linear,
    output: Output,
}

pub struct Seanet {
    steps: Vec<Step>,
}

/// One stream's decoding state and activations, for calls of up to `rows` input rows.
pub struct State {
    rows: usize,
    /// The last `window − 1` input rows of each conv.
    histories: Vec<SharedBuf>,
    /// The input rows of every conv after the first.
    inputs: Vec<SharedBuf>,
    /// Per stage, the output of its upsampling conv, which its residual unit adds back.
    skips: Vec<SharedBuf>,
    /// Output rows of the last call, `channels` values each.
    pub output: SharedBuf,
    scratch: PerThread<Scratch>,
}

#[derive(Default)]
struct Scratch {
    window: Vec<f32>,
    quant: QuantizedRows,
}

/// One conv's rows and buffers within a call.
struct Pass<'call> {
    step: &'call Step,
    rows: usize,
    history: &'call SharedBuf,
    input: &'call SharedBuf,
    target: &'call SharedBuf,
    skips: &'call [SharedBuf],
}

impl Seanet {
    /// The output rows of a call of `rows` input rows.
    pub fn output_rows(&self, rows: usize) -> usize {
        rows * self
            .steps
            .iter()
            .map(|step| step.conv.phases)
            .product::<usize>()
    }

    /// The input rows of each step for a call of `rows` input rows.
    fn step_rows(&self, rows: usize) -> impl Iterator<Item = usize> {
        self.steps.iter().scan(rows, |rows, step| {
            let current = *rows;
            *rows *= step.conv.phases;
            Some(current)
        })
    }

    /// Runs every conv over `rows` input rows of `input` (`dimension` values each); the result
    /// lands in `state.output`.
    ///
    /// # Panics
    /// With fewer rows than `kernel − 1`, which the histories need.
    pub fn forward(&self, ctx: &Ctx, state: &State, input: &SharedBuf, rows: usize) {
        assert!(rows <= state.rows, "the state holds the call");
        for ((index, step), step_rows) in self.steps.iter().enumerate().zip(self.step_rows(rows)) {
            let pass = Pass {
                step,
                rows: step_rows,
                history: &state.histories[index],
                input: state.input(index, input),
                target: state.inputs.get(index).unwrap_or(&state.output),
                skips: &state.skips,
            };
            convolve(ctx, state, &pass);
            ctx.barrier();
        }
        self.keep_history(ctx, state, input, rows);
        ctx.barrier();
    }

    /// Moves the last `window − 1` input rows of every conv into its history.
    fn keep_history(&self, ctx: &Ctx, state: &State, input: &SharedBuf, rows: usize) {
        let mine = ctx.split(self.steps.len(), 1);
        let steps = self.steps.iter().zip(self.step_rows(rows)).enumerate();
        for (index, (step, step_rows)) in steps.filter(|(index, _)| mine.contains(index)) {
            let conv = &step.conv;
            assert!(
                step_rows + 1 >= conv.window,
                "a call fills every conv's history"
            );
            let kept = (conv.window - 1) * conv.inputs;
            let end = step_rows * conv.inputs;
            let source = &state.input(index, input).read()[end - kept..end];
            state.histories[index].part(0..kept).copy_from_slice(source);
        }
    }
}

/// This thread's share of one conv: the windows of its rows, the product, the output.
fn convolve(ctx: &Ctx, state: &State, pass: &Pass) {
    let conv = &pass.step.conv;
    let (rows, cols) = if pass.rows < conv.row() {
        (0..pass.rows, ctx.split(conv.row(), COLUMN_ALIGN))
    } else {
        (ctx.split(pass.rows, 1), 0..conv.row())
    };
    let scratch = state.scratch.get(ctx);
    let width = conv.window * conv.inputs;
    scratch.window.resize(rows.len() * width, 0.0);
    for (row, window) in rows.clone().zip(scratch.window.chunks_exact_mut(width)) {
        gather(pass, row, window);
    }
    let linear = &pass.step.linear;
    let input = scratch.quant.input(&linear.matrix(), &scratch.window);
    let first = rows.start;
    linear.project(&input, cols.clone(), |row, col, value| {
        pass.target.set((first + row) * conv.row() + col, value);
    });
    for row in rows {
        let span = row * conv.row() + cols.start..row * conv.row() + cols.end;
        finish(pass, span);
    }
}

impl State {
    pub fn new(seanet: &Seanet, rows: usize, threads: usize) -> Self {
        let mut histories = Vec::new();
        let mut inputs = Vec::new();
        let mut skips = Vec::new();
        for (step, step_rows) in seanet.steps.iter().zip(seanet.step_rows(rows)) {
            let conv = &step.conv;
            histories.push(SharedBuf::new((conv.window - 1) * conv.inputs));
            inputs.push(SharedBuf::new(step_rows * conv.row()));
            if let Output::NextAndSkip { .. } = step.output {
                skips.push(SharedBuf::new(step_rows * conv.row()));
            }
        }
        let output = inputs.pop().expect("a decoder has convs");
        Self {
            rows,
            histories,
            inputs,
            skips,
            output,
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    /// The input of step `index`: the decoder's input for the first.
    fn input<'call>(&'call self, index: usize, first: &'call SharedBuf) -> &'call SharedBuf {
        match index {
            0 => first,
            _ => &self.inputs[index - 1],
        }
    }

    /// Forgets the stream: every history back to zeros.
    pub fn reset(&mut self) {
        for history in &mut self.histories {
            history.get_mut().fill(0.0);
        }
    }
}

/// The input rows `row − window + 1 ..= row` of a conv, oldest first, from its history and its
/// input.
fn gather(pass: &Pass, row: usize, window: &mut [f32]) {
    let conv = &pass.step.conv;
    let (history, input) = (pass.history.read(), pass.input.read());
    let kept = conv.window - 1;
    for (tap, values) in window.chunks_exact_mut(conv.inputs).enumerate() {
        let source = row + tap;
        let rows = if source < kept {
            &history[source * conv.inputs..]
        } else {
            &input[(source - kept) * conv.inputs..]
        };
        values.copy_from_slice(&rows[..conv.inputs]);
    }
}

/// Applies a conv's output rule to the results in `span` of its target.
fn finish(pass: &Pass, span: Range<usize>) {
    let values = pass.target.part(span.clone());
    match pass.step.output {
        Output::Last => return,
        Output::Next => {}
        Output::NextAndSkip { stage } => pass.skips[stage].part(span).copy_from_slice(values),
        Output::NextPlusSkip { stage } => {
            for (value, saved) in values.iter_mut().zip(pass.skips[stage].part(span).iter()) {
                *value += saved;
            }
        }
    }
    values.iter_mut().for_each(|value| *value = elu(*value));
}

impl Shape {
    /// Every conv in order, with its tensor name under `prefix` and where its results go.
    fn layouts(&self, prefix: &str) -> Vec<(String, CausalConv, Output)> {
        let stages = self.ratios.len();
        let conv = |window, inputs, outputs| CausalConv {
            window,
            phases: 1,
            inputs,
            outputs,
        };
        let top = self.filters << stages;
        let mut layouts = vec![(
            format!("{prefix}.init"),
            conv(self.kernel, self.dimension, top),
            Output::Next,
        )];
        for (stage, &ratio) in self.ratios.iter().enumerate() {
            let wide = top >> stage;
            let (narrow, inner) = (wide / 2, wide / 2 / self.compress);
            let name = |part: &str| format!("{prefix}.stages.{stage}.{part}");
            let upsample = CausalConv {
                phases: ratio,
                ..conv(2, wide, narrow)
            };
            layouts.push((name("upsample"), upsample, Output::NextAndSkip { stage }));
            let residual1 = conv(self.residual_kernel, narrow, inner);
            layouts.push((name("residual1"), residual1, Output::Next));
            let residual2 = conv(1, inner, narrow);
            layouts.push((name("residual2"), residual2, Output::NextPlusSkip { stage }));
        }
        let last = conv(self.last_kernel, self.filters, self.channels);
        layouts.push((format!("{prefix}.final"), last, Output::Last));
        layouts
    }
}

/// Each conv a linear layer of `[phases · out, window · in]`.
impl Load for Seanet {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let steps = shape
            .layouts(prefix)
            .into_iter()
            .map(|(name, conv, output)| {
                let matrix = [conv.row(), conv.window * conv.inputs];
                Ok(Step {
                    linear: Linear::load(store, &name, matrix)?,
                    conv,
                    output,
                })
            })
            .collect::<Result<_, bundle::Error>>()?;
        Ok(Self { steps })
    }
}
