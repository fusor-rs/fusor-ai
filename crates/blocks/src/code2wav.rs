//! Qwen3-Omni's Code2Wav vocoder, streaming (Qwen3-TTS uses it unchanged): per upsampling ratio a
//! transposed conv and a ConvNeXt block, then a causal conv, per rate a SnakeBeta, a transposed
//! conv and residual units (SnakeBeta, dilated conv, SnakeBeta, 1-wide conv, plus the unit's
//! input), and a last SnakeBeta, conv and clamp to [−1, 1]: latent rows into samples.
//!
//! Every conv is a `CausalConv` (`causal`) and runs as one phase; a stream keeps each conv's last
//! `kept()` input rows between calls, so frames decoded in any number of calls give the samples
//! of one call. The model warms a stream with `history_frames` frames before the first it keeps.
//!
//! Tensors under a prefix `p`: `p.upsample.{i}.{transpose, convnext}` (a linear and a ConvNeXt
//! block, see `convnext`), `p.input`, `p.blocks.{b}.{snake, upsample}`,
//! `p.blocks.{b}.units.{u}.{snake1, conv1, snake2, conv2}`, `p.snake` and `p.output`; a snake
//! `s` is the vectors `s.alpha` and `s.beta`, both the logarithms of its parameters.

mod convnext;

use std::ops::Range;

use bundle::{Load, Store};
use kernels::{Linear, Vector};
use team::{Ctx, PerThread, SharedBuf};

use self::convnext::{Buffers, Call, ConvNext};
use crate::causal::{self, CausalConv, Product, Scratch};

/// SnakeBeta's `no_div_by_zero`.
const SNAKE_EPSILON: f32 = 1e-9;

/// The dimensions of one vocoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Channels of the input rows.
    pub latent: usize,
    /// Ratio of each transposed conv ahead of the decoder.
    pub upsampling: &'static [usize],
    /// Channels of the decoder's first conv; each block halves them.
    pub decoder: usize,
    /// Ratio of each block's transposed conv.
    pub rates: &'static [usize],
    /// Dilation of each of a block's residual units.
    pub dilations: &'static [usize],
    /// Taps of the ConvNeXt blocks' depthwise convs, the decoder's first and last conv and the
    /// residual units' first.
    pub kernel: usize,
    /// How much wider a ConvNeXt block's feed-forward is than its channels.
    pub expansion: usize,
    /// Channels of the samples.
    pub channels: usize,
    pub norm_eps: f32,
}

impl Shape {
    /// Samples one input row becomes.
    pub fn frame_samples(&self) -> usize {
        let ratios = self.upsampling.iter().chain(self.rates);
        ratios.product::<usize>()
    }

    /// The input rows before a row that its samples depend on: the receptive field in samples,
    /// whole frames of it.
    pub fn history_frames(&self) -> usize {
        let frame_samples = self.frame_samples();
        let mut rows_per_frame = 1;
        let mut field = 0;
        for stage in self.layouts("").stages {
            field += stage.conv.kept() * (frame_samples / rows_per_frame);
            rows_per_frame *= stage.conv.phases;
        }
        field.div_ceil(frame_samples)
    }
}

/// What a conv does with its results, in order: adds a skip, keeps the sum as a skip, then
/// applies an activation.
#[derive(Debug, Clone, Copy)]
struct Output {
    add_skip: Option<usize>,
    save_skip: Option<usize>,
    activation: Activation,
}

impl Output {
    const fn to(activation: Activation) -> Self {
        Self {
            add_skip: None,
            save_skip: None,
            activation,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    /// The result is the next conv's input as it is.
    Raw,
    /// The snake of this index.
    Snake(usize),
    /// The vocoder's output.
    Clamp,
}

/// A SnakeBeta: `x + sin²(x · frequency) / inverse` per channel, from the checkpoint's `alpha`
/// and `beta`, the logarithms of its parameters.
struct Snake {
    frequency: Vec<f32>,
    inverse: Vec<f32>,
}

impl Snake {
    fn load(store: &mut Store, name: &str, channels: usize) -> Result<Self, bundle::Error> {
        let alpha = Vector::load(store, &format!("{name}.alpha"), channels)?;
        let beta = Vector::load(store, &format!("{name}.beta"), channels)?;
        Ok(Self {
            frequency: alpha.values().iter().map(|log| log.exp()).collect(),
            inverse: beta
                .values()
                .iter()
                .map(|log| 1.0 / (log.exp() + SNAKE_EPSILON))
                .collect(),
        })
    }

    fn apply(&self, channel: usize, value: f32) -> f32 {
        let wave = (value * self.frequency[channel]).sin();
        value + self.inverse[channel] * wave * wave
    }
}

enum Kind {
    Conv { linear: Linear, output: Output },
    ConvNext(Box<ConvNext>),
}

struct Stage {
    conv: CausalConv,
    kind: Kind,
}

pub struct Code2wav {
    shape: Shape,
    stages: Vec<Stage>,
    snakes: Vec<Snake>,
}

/// One stream's state and activations, for calls of up to `frames` input rows.
pub struct Stream {
    frames: usize,
    /// The last `kept()` input rows of each stage.
    histories: Vec<SharedBuf>,
    /// The input rows of every stage after the first.
    inputs: Vec<SharedBuf>,
    /// Per block, the input of its residual unit, which the unit adds back.
    skips: Vec<SharedBuf>,
    /// The samples of the last call.
    pub output: SharedBuf,
    convnext: Buffers,
    scratch: PerThread<Scratch>,
}

impl Stream {
    /// The input of stage `index`: the vocoder's input for the first.
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

/// One stage's rows and buffers within a call.
struct Pass<'call> {
    product: Product<'call>,
    output: Output,
    skips: &'call [SharedBuf],
    snakes: &'call [Snake],
}

impl Code2wav {
    /// A stream decoding up to `frames` input rows per call.
    pub fn stream(&self, frames: usize, threads: usize) -> Stream {
        let mut histories = Vec::new();
        let mut inputs = Vec::new();
        let mut skips = Vec::new();
        let mut convnext_rows = 0;
        for (stage, rows) in self.stages.iter().zip(self.stage_rows(frames)) {
            let conv = &stage.conv;
            histories.push(SharedBuf::new(conv.kept() * conv.inputs));
            inputs.push(SharedBuf::new(rows * conv.row()));
            match &stage.kind {
                Kind::ConvNext(_) => convnext_rows = convnext_rows.max(rows),
                Kind::Conv { output, .. } if output.save_skip == Some(skips.len()) => {
                    skips.push(SharedBuf::new(rows * conv.row()));
                }
                Kind::Conv { .. } => {}
            }
        }
        let output = inputs.pop().expect("a vocoder has convs");
        Stream {
            frames,
            histories,
            inputs,
            skips,
            output,
            convnext: Buffers::new(&self.shape.convnext(), convnext_rows, threads),
            scratch: PerThread::new(threads, Scratch::default),
        }
    }

    /// The input rows of each stage for a call of `frames` input rows.
    fn stage_rows(&self, frames: usize) -> impl Iterator<Item = usize> {
        self.stages.iter().scan(frames, |rows, stage| {
            let current = *rows;
            *rows *= stage.conv.phases;
            Some(current)
        })
    }

    /// Runs every stage over `frames` rows of `input` (`latent` values each) inside a team run;
    /// the samples land in `stream.output`, `frame_samples` for each.
    pub fn forward(&self, ctx: &Ctx, stream: &Stream, input: &SharedBuf, frames: usize) {
        assert!(frames <= stream.frames, "the stream holds the call");
        let stages = self.stages.iter().enumerate().zip(self.stage_rows(frames));
        for ((index, stage), rows) in stages {
            let source = stream.input(index, input);
            let target = stream.inputs.get(index).unwrap_or(&stream.output);
            let history = &stream.histories[index];
            match &stage.kind {
                Kind::Conv { linear, output } => {
                    let pass = Pass {
                        product: Product {
                            conv: &stage.conv,
                            linear,
                            rows,
                            history,
                            input: source,
                            target,
                        },
                        output: *output,
                        skips: &stream.skips,
                        snakes: &self.snakes,
                    };
                    convolve(ctx, stream, &pass);
                }
                Kind::ConvNext(convnext) => {
                    let call = Call {
                        history,
                        input: source,
                        target,
                        rows,
                    };
                    convnext.forward(ctx, &stream.convnext, &call);
                }
            }
            ctx.barrier();
        }
        self.keep_history(ctx, stream, input, frames);
        ctx.barrier();
    }

    /// Moves the last `kept()` input rows of every stage into its history.
    fn keep_history(&self, ctx: &Ctx, stream: &Stream, input: &SharedBuf, frames: usize) {
        let mine = ctx.split(self.stages.len(), 1);
        let stages = self.stages.iter().zip(self.stage_rows(frames)).enumerate();
        for (index, (stage, rows)) in stages.filter(|(index, _)| mine.contains(index)) {
            let source = stream.input(index, input);
            causal::keep(&stage.conv, source, rows, &stream.histories[index]);
        }
    }
}

/// This thread's share of one conv: the product, then the output rule on its results.
fn convolve(ctx: &Ctx, stream: &Stream, pass: &Pass) {
    let scratch = stream.scratch.get(ctx);
    let (rows, cols) = causal::project(ctx, &pass.product, scratch);
    let row_values = pass.product.conv.row();
    for row in rows {
        finish(
            pass,
            row * row_values + cols.start..row * row_values + cols.end,
        );
    }
}

/// Applies a conv's output rule to the results in `span` of its target. The target's rows are
/// phase-major, so the channel of a value is its position modulo the conv's outputs.
fn finish(pass: &Pass, span: Range<usize>) {
    let Output {
        add_skip,
        save_skip,
        activation,
    } = pass.output;
    let values = pass.product.target.part(span.clone());
    if let Some(skip) = add_skip {
        for (value, saved) in values.iter_mut().zip(pass.skips[skip].part(span.clone())) {
            *value += *saved;
        }
    }
    if let Some(skip) = save_skip {
        pass.skips[skip].part(span.clone()).copy_from_slice(values);
    }
    match activation {
        Activation::Raw => {}
        Activation::Snake(index) => {
            let outputs = pass.product.conv.outputs;
            for (at, value) in span.zip(values.iter_mut()) {
                *value = pass.snakes[index].apply(at % outputs, *value);
            }
        }
        Activation::Clamp => values
            .iter_mut()
            .for_each(|value| *value = value.clamp(-1.0, 1.0)),
    }
}

/// A stage of the vocoder before it is loaded.
struct Layout {
    name: String,
    conv: CausalConv,
    convnext: Option<convnext::Shape>,
    output: Output,
}

/// Every stage in order, and the name and channels of every snake.
struct Layouts {
    stages: Vec<Layout>,
    snakes: Vec<(String, usize)>,
}

impl Layouts {
    fn snake(&mut self, name: String, channels: usize) -> Activation {
        self.snakes.push((name, channels));
        Activation::Snake(self.snakes.len() - 1)
    }

    fn conv(&mut self, name: String, conv: CausalConv, output: Output) {
        self.stages.push(Layout {
            name,
            conv,
            convnext: None,
            output,
        });
    }
}

fn plain(window: usize, inputs: usize, outputs: usize) -> CausalConv {
    CausalConv {
        window,
        phases: 1,
        inputs,
        outputs,
        dilation: 1,
    }
}

impl Shape {
    /// Every stage in order, with its tensor name under `prefix` and where its results go.
    fn layouts(&self, prefix: &str) -> Layouts {
        let mut layouts = Layouts {
            stages: Vec::new(),
            snakes: Vec::new(),
        };
        for (index, &ratio) in self.upsampling.iter().enumerate() {
            self.upsample(prefix, (index, ratio), &mut layouts);
        }
        let first = layouts.snake(format!("{prefix}.blocks.0.snake"), self.decoder);
        let input = plain(self.kernel, self.latent, self.decoder);
        layouts.conv(format!("{prefix}.input"), input, Output::to(first));
        for (block, &rate) in self.rates.iter().enumerate() {
            self.block(prefix, (block, rate), &mut layouts);
        }
        let last_channels = self.decoder >> self.rates.len();
        let output = plain(self.kernel, last_channels, self.channels);
        let name = format!("{prefix}.output");
        layouts.conv(name, output, Output::to(Activation::Clamp));
        layouts
    }

    fn convnext(&self) -> convnext::Shape {
        convnext::Shape {
            channels: self.latent,
            kernel: self.kernel,
            expansion: self.expansion,
            norm_eps: self.norm_eps,
        }
    }

    /// A transposed conv by `ratio`, then a ConvNeXt block.
    fn upsample(&self, prefix: &str, (index, ratio): (usize, usize), layouts: &mut Layouts) {
        let name = format!("{prefix}.upsample.{index}");
        let transpose = CausalConv {
            phases: ratio,
            ..plain(1, self.latent, self.latent)
        };
        layouts.conv(
            format!("{name}.transpose"),
            transpose,
            Output::to(Activation::Raw),
        );
        layouts.stages.push(Layout {
            name: format!("{name}.convnext"),
            conv: plain(self.kernel, self.latent, self.latent),
            convnext: Some(self.convnext()),
            output: Output::to(Activation::Raw),
        });
    }

    /// One block: the transposed conv by `rate` and its residual units, each unit's last conv
    /// ending in the snake that opens the next block (or the vocoder's last conv).
    fn block(&self, prefix: &str, (block, rate): (usize, usize), layouts: &mut Layouts) {
        let (wide, narrow) = (self.decoder >> block, self.decoder >> (block + 1));
        let name = format!("{prefix}.blocks.{block}");
        let unit_name = |unit: usize, part: &str| format!("{name}.units.{unit}.{part}");
        let first = layouts.snake(unit_name(0, "snake1"), narrow);
        let upsample = CausalConv {
            phases: rate,
            ..plain(2, wide, narrow)
        };
        let opening = Output {
            save_skip: Some(block),
            ..Output::to(first)
        };
        layouts.conv(format!("{name}.upsample"), upsample, opening);
        let units = self.dilations.len();
        for (unit, &dilation) in self.dilations.iter().enumerate() {
            let middle = layouts.snake(unit_name(unit, "snake2"), narrow);
            let conv1 = CausalConv {
                dilation,
                ..plain(self.kernel, narrow, narrow)
            };
            layouts.conv(unit_name(unit, "conv1"), conv1, Output::to(middle));
            let next = if unit + 1 < units {
                unit_name(unit + 1, "snake1")
            } else if block + 1 < self.rates.len() {
                format!("{prefix}.blocks.{}.snake", block + 1)
            } else {
                format!("{prefix}.snake")
            };
            let closing = Output {
                add_skip: Some(block),
                save_skip: (unit + 1 < units).then_some(block),
                ..Output::to(layouts.snake(next, narrow))
            };
            layouts.conv(unit_name(unit, "conv2"), plain(1, narrow, narrow), closing);
        }
    }
}

/// Each conv a linear layer of `[phases · out, window · in]`.
impl Load for Code2wav {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let layouts = shape.layouts(prefix);
        let stages = layouts
            .stages
            .into_iter()
            .map(|layout| {
                let Layout {
                    name,
                    conv,
                    convnext,
                    output,
                } = layout;
                let kind = match convnext {
                    Some(convnext) => {
                        Kind::ConvNext(Box::new(ConvNext::load(store, &name, convnext)?))
                    }
                    None => Kind::Conv {
                        linear: Linear::load(
                            store,
                            &name,
                            [conv.row(), conv.window * conv.inputs],
                        )?,
                        output,
                    },
                };
                Ok(Stage { conv, kind })
            })
            .collect::<Result<_, bundle::Error>>()?;
        let snakes = layouts
            .snakes
            .iter()
            .map(|(name, channels)| Snake::load(store, name, *channels))
            .collect::<Result<_, _>>()?;
        Ok(Code2wav {
            shape,
            stages,
            snakes,
        })
    }
}
