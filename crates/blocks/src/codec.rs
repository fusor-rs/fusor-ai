//! The decoder of a neural audio codec of finite scalar quantization that synthesizes through an
//! inverse STFT (Liquid's audio codec). A frame holds one code per codebook, each a mixed-radix
//! number whose digits, lowest first, have the shape's `levels`; digit `d` of level `l` is the
//! latent `(d − ⌊l/2⌋) / ⌊l/2⌋`. A frame's latents, codebook by codebook, go through
//! `project_out` and `from_quantizer` (linears with biases); each frame is repeated `upsample`
//! times (nearest upsampling) through a causal `interleaved` stack and its final RMSNorm;
//! `spectrum` maps each row to `n_fft / 2 + 1` log-magnitudes then as many phases, and `istft`
//! turns them into windowed frames overlap-added every `hop` samples.
//!
//! Tensors under a prefix `p`: `p.project_out`, `p.from_quantizer`, the stack's
//! (`p.layers.{i}…`), `p.norm`, `p.spectrum` and `p.window` (`n_fft` samples).
//!
//! `Stream::decode` reads consecutive frames of one stream from its first: the stack keeps its
//! window's keys and its convolutions' inputs, the overlap-add the samples later frames still
//! add to. So a stream decoded in calls gives the samples one call over all its frames gives.

use bundle::{Load, Scope, Store};
use kernels::ops::rms_norm;
use kernels::{Linear, QuantizedRows, SharedRows, Vector, project};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::hybrid::Rows;
use crate::interleaved::{self, Interleaved};
use crate::istft::{FftScratch, Istft, OverlapAdd};
use crate::layer;
use crate::rotary::Position;
use crate::transformer::Mask;

/// The codec takes every magnitude as its log-magnitude gives it.
const UNBOUNDED_MAGNITUDE: f32 = f32::INFINITY;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub codebooks: usize,
    /// The levels of a code's digits, lowest first.
    pub levels: &'static [usize],
    /// Width of `project_out`'s output.
    pub latent: usize,
    pub stack: interleaved::Shape,
    /// Rows of the stack per frame.
    pub upsample: usize,
    pub n_fft: usize,
    pub hop: usize,
}

impl Shape {
    /// Codes of a codebook.
    pub fn vocabulary(&self) -> usize {
        self.levels.iter().product()
    }

    /// Latents of a frame.
    fn latents(&self) -> usize {
        self.codebooks * self.levels.len()
    }

    fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

pub struct Decoder {
    shape: Shape,
    project_out: Linear,
    from_quantizer: Linear,
    stack: Interleaved,
    norm: Vector,
    spectrum: Linear,
    istft: Istft,
}

/// One stream's state, and the activations of its calls.
pub struct Stream {
    /// Frames per call.
    frames: usize,
    /// Rows of the stack read so far.
    rows: usize,
    state: interleaved::State,
    stack: layer::Buffers,
    /// `frames × latents`.
    latents: SharedBuf,
    /// `frames × latent`.
    projected: SharedBuf,
    /// The stack's output final-normed, `rows × hidden`.
    normed: SharedBuf,
    /// `rows × 2 · bins`.
    spectra: SharedBuf,
    /// The windowed time frames, `rows × n_fft`.
    windows: SharedBuf,
    fft: PerThread<FftScratch>,
    quant: PerThread<QuantizedRows>,
    overlap: OverlapAdd,
}

impl Decoder {
    /// A stream of up to `max_frames` frames, decoded `frames` at a time.
    pub fn stream(&self, frames: usize, max_frames: usize, threads: usize) -> Stream {
        let shape = &self.shape;
        let rows = frames * shape.upsample;
        let hidden = shape.stack.hidden;
        Stream {
            frames,
            rows: 0,
            state: interleaved::State::new(&shape.stack, max_frames * shape.upsample, rows),
            stack: shape.stack.buffers(rows, threads),
            latents: SharedBuf::new(frames * shape.latents()),
            projected: SharedBuf::new(frames * shape.latent),
            normed: SharedBuf::new(rows * hidden),
            spectra: SharedBuf::new(rows * 2 * shape.bins()),
            windows: SharedBuf::new(rows * shape.n_fft),
            fft: PerThread::new(threads, || self.istft.scratch()),
            quant: PerThread::new(threads, QuantizedRows::default),
            overlap: self.istft.overlap_add(shape.hop),
        }
    }

    /// The windowed time frames of `codes` (`frames × codebooks`) into `stream.windows`, the
    /// stack's rows from `rows.first`.
    fn forward(&self, ctx: &Ctx, stream: &Stream, codes: &[u32], rows: &Rows) {
        self.dequantize(ctx, stream, codes);
        ctx.barrier();
        self.embed(ctx, stream, codes.len() / self.shape.codebooks);
        self.stack.forward(ctx, &stream.state, &stream.stack, rows);
        self.synthesize(ctx, stream, rows.positions.len());
    }

    /// This thread's frames of `stream.latents`.
    fn dequantize(&self, ctx: &Ctx, stream: &Stream, codes: &[u32]) {
        let levels = self.shape.levels;
        let latents = self.shape.latents();
        for frame in ctx.split(codes.len() / self.shape.codebooks, 1) {
            let out = stream.latents.part(frame * latents..(frame + 1) * latents);
            let frame_codes = &codes[frame * self.shape.codebooks..];
            for (digits, &code) in out.chunks_exact_mut(levels.len()).zip(frame_codes) {
                let mut rest = code as usize;
                for (latent, &level) in digits.iter_mut().zip(levels) {
                    let half = (level / 2) as f32;
                    *latent = ((rest % level) as f32 - half) / half;
                    rest /= level;
                }
            }
        }
    }

    /// The stack's input: each frame's latents through `project_out` and `from_quantizer`, in
    /// each of its `upsample` rows.
    fn embed(&self, ctx: &Ctx, stream: &Stream, frames: usize) {
        let (hidden, upsample) = (self.shape.stack.hidden, self.shape.upsample);
        let quant = stream.quant.get(ctx);
        let rows = |values| SharedRows {
            values,
            rows: frames,
        };
        project(
            ctx,
            &self.project_out,
            rows(&stream.latents),
            quant,
            |at, value| stream.projected.set(at, value),
        );
        ctx.barrier();
        project(
            ctx,
            &self.from_quantizer,
            rows(&stream.projected),
            quant,
            |at, value| {
                let (frame, column) = (at / hidden, at % hidden);
                for row in frame * upsample..(frame + 1) * upsample {
                    stream.stack.hidden.set(row * hidden + column, value);
                }
            },
        );
        ctx.barrier();
    }

    /// `rows` rows of the stack's output final-normed, through `spectrum`, into windowed time
    /// frames.
    fn synthesize(&self, ctx: &Ctx, stream: &Stream, rows: usize) {
        let hidden = self.shape.stack.hidden;
        let output = stream.stack.hidden.read();
        for row in ctx.split(rows, 1) {
            let range = row * hidden..(row + 1) * hidden;
            let normed = stream.normed.part(range.clone());
            rms_norm(
                &output[range],
                self.norm.values(),
                self.shape.stack.norm_eps,
                normed,
            );
        }
        ctx.barrier();
        let width = 2 * self.shape.bins();
        let normed = &stream.normed.read()[..rows * hidden];
        let input = stream.quant.get(ctx).input(&self.spectrum.matrix(), normed);
        // `project` splits by `COLUMN_ALIGN`, of which `2 · bins` columns need not be a multiple.
        let cols = ctx.split(width, 1);
        self.spectrum.project(&input, cols, |row, col, value| {
            stream.spectra.set(row * width + col, value);
        });
        ctx.barrier();
        let (bins, n_fft) = (self.shape.bins(), self.shape.n_fft);
        let spectra = stream.spectra.read();
        let scratch = stream.fft.get(ctx);
        for row in ctx.split(rows, 1) {
            let (log_magnitude, phase) = spectra[row * 2 * bins..][..2 * bins].split_at(bins);
            let out = stream.windows.part(row * n_fft..(row + 1) * n_fft);
            let polar = (log_magnitude, phase);
            self.istft.frame(polar, UNBOUNDED_MAGNITUDE, scratch, out);
        }
    }
}

impl Stream {
    /// The next `codes` (`frames × codebooks`) of the stream; appends the samples no later frame
    /// changes to `out`.
    ///
    /// # Panics
    /// When the codes are more frames than a call or the stream holds, or a code is past its
    /// codebook.
    pub fn decode(&mut self, team: &Team, decoder: &Decoder, codes: &[u32], out: &mut Vec<f32>) {
        let shape = &decoder.shape;
        let frames = codes.len() / shape.codebooks;
        assert!(frames <= self.frames, "a call holds the frames");
        assert!(
            codes
                .iter()
                .all(|&code| (code as usize) < shape.vocabulary()),
            "every code is in its codebook"
        );
        let first = self.rows;
        let count = frames * shape.upsample;
        let positions: Vec<Position> = (first..first + count).map(Position::uniform).collect();
        let rows = Rows {
            first,
            positions: &positions,
            mask: Mask::Causal,
        };
        let stream = &*self;
        team.run(|ctx| decoder.forward(ctx, stream, codes, &rows));
        let windows = self.windows.read();
        for window in windows.chunks_exact(shape.n_fft).take(count) {
            self.overlap.add(window);
        }
        self.overlap.emit_ready(out);
        self.rows += count;
    }

    /// Appends the stream's last samples to `out`.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        self.overlap.finish(out);
    }
}

impl Load for Decoder {
    type Shape = Shape;

    fn load(store: &mut Store, prefix: &str, shape: Shape) -> Result<Self, bundle::Error> {
        let hidden = shape.stack.hidden;
        let stack = Interleaved::load(store, prefix, shape.stack)?;
        let mut scope = Scope::new(store, prefix);
        let window: Vector = scope.load("window", shape.n_fft)?;
        Ok(Self {
            shape,
            project_out: scope.load("project_out", [shape.latent, shape.latents()])?,
            from_quantizer: scope.load("from_quantizer", [hidden, shape.latent])?,
            stack,
            norm: scope.load("norm", hidden)?,
            spectrum: scope.load("spectrum", [2 * shape.bins(), hidden])?,
            istft: Istft::new(window.values().to_vec()),
        })
    }
}
