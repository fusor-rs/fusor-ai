//! SEANet's streaming decoder or encoder (EnCodec, Mimi). The decoder is a causal conv, then per
//! stage an ELU, a transposed conv upsampling by the stage's ratio (kernel `2 · ratio`) and one
//! residual unit (ELU, conv, ELU, 1-wide conv, plus the unit's input), then an ELU and a final
//! conv; the encoder is a causal conv, then per stage one residual unit, an ELU and a conv
//! downsampling by the stage's ratio (kernel `2 · ratio`), then an ELU and a final conv.
//!
//! Activations are time-major rows of channels. Every conv is a `CausalConv` (`causal`): an
//! encoder's downsampling conv is a window-2 conv over rows of `ratio` time steps, so its
//! matrices are laid out as the decoder's `[phases · out, window · in]`.
//!
//! `forward` runs inside a team run: one phase per conv, then one phase that moves every conv's
//! history.
//!
//! Tensors under a prefix `p`: `p.init`, `p.stages.{i}.{upsample, residual1, residual2}` (the
//! encoder's `upsample` is `downsample`) and `p.final`, each a linear layer.

use std::ops::Range;

use bundle::{Load, Store};
use kernels::Linear;
use kernels::ops::elu;
use team::{Ctx, PerThread, SharedBuf};

use crate::causal::{self, CausalConv, Product, Scratch};

/// Which way a SEANet converts: samples to a latent, or a latent to samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Decode,
    Encode,
}

/// The dimensions of one decoder or encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub direction: Direction,
    /// Channels of the latent: the decoder's input, the encoder's output.
    pub dimension: usize,
    /// Channels of the stage nearest the samples; the others double with each stage away.
    pub filters: usize,
    /// Ratio of each stage, in the order applied.
    pub ratios: &'static [usize],
    pub kernel: usize,
    pub last_kernel: usize,
    pub residual_kernel: usize,
    /// How much a residual unit narrows its channels.
    pub compress: usize,
    /// Channels of the samples.
    pub channels: usize,
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
    /// The network's output.
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

/// One stream's state and activations, for calls of up to `rows` input rows.
pub struct State {
    rows: usize,
    /// The last `kept()` input rows of each conv.
    histories: Vec<SharedBuf>,
    /// The input rows of every conv after the first.
    inputs: Vec<SharedBuf>,
    /// Per stage, the raw result of the conv that opens its residual unit's block, which the
    /// unit adds back.
    skips: Vec<SharedBuf>,
    /// Output rows of the last call, `channels` values each (`dimension` for an encoder).
    pub output: SharedBuf,
    scratch: PerThread<Scratch>,
}

/// One conv's rows and buffers within a call.
struct Pass<'call> {
    product: Product<'call>,
    output: Output,
    skips: &'call [SharedBuf],
}

impl Seanet {
    /// The output rows of a call of `rows` input rows.
    pub fn output_rows(&self, rows: usize) -> usize {
        let last = self.steps.last().expect("a SEANet has convs");
        let last_rows = self.step_rows(rows).last().expect("a SEANet has convs");
        last_rows * last.conv.phases
    }

    /// The input rows of each step for a call of `rows` input rows.
    fn step_rows(&self, rows: usize) -> impl Iterator<Item = usize> {
        let first = self.steps.first().expect("a SEANet has convs");
        let values = rows * first.conv.inputs;
        self.steps.iter().scan(values, |values, step| {
            let current = *values / step.conv.inputs;
            *values = current * step.conv.row();
            Some(current)
        })
    }

    /// Runs every conv over `rows` input rows of `input` (the first conv's `inputs` values
    /// each); the result lands in `state.output`.
    pub fn forward(&self, ctx: &Ctx, state: &State, input: &SharedBuf, rows: usize) {
        assert!(rows <= state.rows, "the state holds the call");
        for ((index, step), step_rows) in self.steps.iter().enumerate().zip(self.step_rows(rows)) {
            let pass = Pass {
                product: Product {
                    conv: &step.conv,
                    linear: &step.linear,
                    rows: step_rows,
                    history: &state.histories[index],
                    input: state.input(index, input),
                    target: state.inputs.get(index).unwrap_or(&state.output),
                },
                output: step.output,
                skips: &state.skips,
            };
            convolve(ctx, state, &pass);
            ctx.barrier();
        }
        self.keep_history(ctx, state, input, rows);
        ctx.barrier();
    }

    /// Moves the last `kept()` input rows of every conv into its history.
    fn keep_history(&self, ctx: &Ctx, state: &State, input: &SharedBuf, rows: usize) {
        let mine = ctx.split(self.steps.len(), 1);
        let steps = self.steps.iter().zip(self.step_rows(rows)).enumerate();
        for (index, (step, step_rows)) in steps.filter(|(index, _)| mine.contains(index)) {
            let source = state.input(index, input);
            causal::keep(&step.conv, source, step_rows, &state.histories[index]);
        }
    }
}

/// This thread's share of one conv: the product, then the output rule on its results.
fn convolve(ctx: &Ctx, state: &State, pass: &Pass) {
    let (rows, cols) = causal::project(ctx, &pass.product, state.scratch.get(ctx));
    let row_values = pass.product.conv.row();
    for row in rows {
        finish(
            pass,
            row * row_values + cols.start..row * row_values + cols.end,
        );
    }
}

impl State {
    pub fn new(seanet: &Seanet, rows: usize, threads: usize) -> Self {
        let mut histories = Vec::new();
        let mut inputs = Vec::new();
        let mut skips = Vec::new();
        for (step, step_rows) in seanet.steps.iter().zip(seanet.step_rows(rows)) {
            let conv = &step.conv;
            histories.push(SharedBuf::new(conv.kept() * conv.inputs));
            inputs.push(SharedBuf::new(step_rows * conv.row()));
            if let Output::NextAndSkip { .. } = step.output {
                skips.push(SharedBuf::new(step_rows * conv.row()));
            }
        }
        let output = inputs.pop().expect("a SEANet has convs");
        Self {
            rows,
            histories,
            inputs,
            skips,
            output,
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    /// The input of step `index`: the network's input for the first.
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

/// Applies a conv's output rule to the results in `span` of its target.
fn finish(pass: &Pass, span: Range<usize>) {
    let values = pass.product.target.part(span.clone());
    match pass.output {
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
        match self.direction {
            Direction::Decode => self.decoder_layouts(prefix),
            Direction::Encode => self.encoder_layouts(prefix),
        }
    }

    fn conv(window: usize, inputs: usize, outputs: usize) -> CausalConv {
        CausalConv {
            window,
            phases: 1,
            inputs,
            outputs,
            dilation: 1,
        }
    }

    fn decoder_layouts(&self, prefix: &str) -> Vec<(String, CausalConv, Output)> {
        let stages = self.ratios.len();
        let top = self.filters << stages;
        let mut layouts = vec![(
            format!("{prefix}.init"),
            Self::conv(self.kernel, self.dimension, top),
            Output::Next,
        )];
        for (stage, &ratio) in self.ratios.iter().enumerate() {
            let wide = top >> stage;
            let (narrow, inner) = (wide / 2, wide / 2 / self.compress);
            let name = |part: &str| format!("{prefix}.stages.{stage}.{part}");
            let upsample = CausalConv {
                phases: ratio,
                ..Self::conv(2, wide, narrow)
            };
            layouts.push((name("upsample"), upsample, Output::NextAndSkip { stage }));
            let residual1 = Self::conv(self.residual_kernel, narrow, inner);
            layouts.push((name("residual1"), residual1, Output::Next));
            let residual2 = Self::conv(1, inner, narrow);
            layouts.push((name("residual2"), residual2, Output::NextPlusSkip { stage }));
        }
        let last = Self::conv(self.last_kernel, self.filters, self.channels);
        layouts.push((format!("{prefix}.final"), last, Output::Last));
        layouts
    }

    fn encoder_layouts(&self, prefix: &str) -> Vec<(String, CausalConv, Output)> {
        let stages = self.ratios.len();
        let mut layouts = vec![(
            format!("{prefix}.init"),
            Self::conv(self.kernel, self.channels, self.filters),
            Output::NextAndSkip { stage: 0 },
        )];
        for (stage, &ratio) in self.ratios.iter().enumerate() {
            let narrow = self.filters << stage;
            let inner = narrow / self.compress;
            let name = |part: &str| format!("{prefix}.stages.{stage}.{part}");
            let residual1 = Self::conv(self.residual_kernel, narrow, inner);
            layouts.push((name("residual1"), residual1, Output::Next));
            let residual2 = Self::conv(1, inner, narrow);
            layouts.push((name("residual2"), residual2, Output::NextPlusSkip { stage }));
            let downsample = Self::conv(2, ratio * narrow, 2 * narrow);
            let output = if stage + 1 < stages {
                Output::NextAndSkip { stage: stage + 1 }
            } else {
                Output::Next
            };
            layouts.push((name("downsample"), downsample, output));
        }
        let wide = self.filters << stages;
        let last = Self::conv(self.last_kernel, wide, self.dimension);
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
