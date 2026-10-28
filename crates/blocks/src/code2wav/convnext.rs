//! Code2Wav's ConvNeXt block: a depthwise causal conv, a LayerNorm over the channels, a
//! feed-forward of `expansion` times the channels with an exact GELU, and the block's input
//! added back. The compiler folds the block's `gamma` into the feed-forward's last layer.
//!
//! Tensors under a prefix `p`: `p.depthwise` (`[channels, kernel]`, each channel's taps, oldest
//! first), `p.depthwise_bias`, the LayerNorm `p.norm`, and the linears `p.up` and `p.down`.

use bundle::{Load, Scope, Store};
use kernels::conv::{Patches, project_rows};
use kernels::ops::gelu_erf;
use kernels::{Linear, Table, Vector};
use team::{Ctx, PerThread, SharedBuf};

use crate::norm::LayerNorm;

#[derive(Debug, Clone, Copy)]
pub(super) struct Shape {
    pub(super) channels: usize,
    pub(super) kernel: usize,
    /// The feed-forward's width over the channels.
    pub(super) expansion: usize,
    pub(super) norm_eps: f32,
}

pub(super) struct ConvNext {
    channels: usize,
    expansion: usize,
    /// `[kernel, channels]`: tap `j` of every channel, contiguous.
    taps: Vec<f32>,
    bias: Vector,
    norm: LayerNorm,
    up: Linear,
    down: Linear,
    norm_eps: f32,
}

impl Load for ConvNext {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let Shape {
            channels,
            kernel,
            expansion,
            norm_eps,
        } = shape;
        let mut scope = Scope::new(store, prefix);
        let depthwise: Table = scope.load("depthwise", [channels, kernel])?;
        let taps = depthwise.dequantize();
        let taps = (0..kernel)
            .flat_map(|tap| (0..channels).map(move |channel| (channel, tap)))
            .map(|(channel, tap)| taps[channel * kernel + tap])
            .collect();
        Ok(Self {
            channels,
            expansion,
            taps,
            bias: scope.load("depthwise_bias", channels)?,
            norm: scope.load("norm", channels)?,
            up: scope.load("up", [expansion * channels, channels])?,
            down: scope.load("down", [channels, expansion * channels])?,
            norm_eps,
        })
    }
}

/// The activations of a call of up to `rows` rows, which every block of a stream shares.
pub(super) struct Buffers {
    depthwise: SharedBuf,
    normed: SharedBuf,
    hidden: SharedBuf,
    patches: PerThread<Patches>,
}

impl Buffers {
    pub(super) fn new(shape: &Shape, rows: usize, threads: usize) -> Self {
        let Shape {
            channels,
            expansion,
            ..
        } = *shape;
        Self {
            depthwise: SharedBuf::new(rows * channels),
            normed: SharedBuf::new(rows * channels),
            hidden: SharedBuf::new(rows * expansion * channels),
            patches: PerThread::new(threads, Patches::default),
        }
    }
}

/// One block's call: `rows` rows of `input` after the `history` of its rows, the result into
/// `target`.
pub(super) struct Call<'call> {
    pub(super) history: &'call SharedBuf,
    pub(super) input: &'call SharedBuf,
    pub(super) target: &'call SharedBuf,
    pub(super) rows: usize,
}

impl ConvNext {
    /// Inside a team run, four phases with barriers between them; the caller barriers before
    /// `target` is read.
    pub(super) fn forward(&self, ctx: &Ctx, buffers: &Buffers, call: &Call) {
        self.depthwise(ctx, buffers, call);
        ctx.barrier();
        self.normalize(ctx, buffers, call.rows);
        ctx.barrier();
        self.expand(ctx, buffers, call.rows);
        ctx.barrier();
        self.contract(ctx, buffers, call);
    }

    /// This thread's channels of the depthwise conv of every row.
    fn depthwise(&self, ctx: &Ctx, buffers: &Buffers, call: &Call) {
        let channels = self.channels;
        let kept = self.taps.len() / channels - 1;
        let (history, input) = (call.history.read(), call.input.read());
        let mine = ctx.split(channels, 1);
        for row in 0..call.rows {
            let out = buffers
                .depthwise
                .part(row * channels + mine.start..row * channels + mine.end);
            out.copy_from_slice(&self.bias.values()[mine.clone()]);
            for (tap, weights) in self.taps.chunks_exact(channels).enumerate() {
                let source = row + tap;
                let values = match source.checked_sub(kept) {
                    Some(fresh) => &input[fresh * channels..][..channels],
                    None => &history[source * channels..][..channels],
                };
                let channel_values = values[mine.clone()].iter().zip(&weights[mine.clone()]);
                for (sum, (value, weight)) in out.iter_mut().zip(channel_values) {
                    *sum += weight * value;
                }
            }
        }
    }

    fn normalize(&self, ctx: &Ctx, buffers: &Buffers, rows: usize) {
        let channels = self.channels;
        let depthwise = buffers.depthwise.read();
        for row in ctx.split(rows, 1) {
            let span = row * channels..(row + 1) * channels;
            let out = buffers.normed.part(span.clone());
            self.norm.apply(&depthwise[span], self.norm_eps, out);
        }
    }

    fn expand(&self, ctx: &Ctx, buffers: &Buffers, rows: usize) {
        let width = self.expansion * self.channels;
        let patches = buffers.patches.get(ctx);
        let input = (buffers.normed.read(), rows);
        project_rows(ctx, &self.up, input, patches, |row, col, value| {
            buffers.hidden.set(row * width + col, gelu_erf(value));
        });
    }

    fn contract(&self, ctx: &Ctx, buffers: &Buffers, call: &Call) {
        let channels = self.channels;
        let patches = buffers.patches.get(ctx);
        let input = call.input.read();
        let hidden = (buffers.hidden.read(), call.rows);
        project_rows(ctx, &self.down, hidden, patches, |row, col, value| {
            let at = row * channels + col;
            call.target.set(at, input[at] + value);
        });
    }
}
