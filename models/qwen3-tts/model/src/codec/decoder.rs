//! The speech tokenizer's decoder: a frame's 16 codes summed through the quantizer's codebooks
//! (the output projections folded in) into a 512-channel row, a causal conv over three rows to
//! 1,024 channels, a Qwen2-style transformer over a sliding window of 72 frames, and
//! `blocks::code2wav` into 1,920 samples. A stream keeps the conv's last two rows, the
//! transformer's cache and the vocoder's histories, so frames decoded in any number of calls give
//! the samples of one call.
//!
//! A reference clip's frames come first in the stream, so that the speech after them is decoded
//! with their context: the transformer reads all of them, the vocoder only the last
//! `history_frames` (its receptive field), whose samples are dropped.
//!
//! Tensors: `codec.decoder.codebooks` (`[16 · 2048, 512]`, codebook-major, the quantizer's output
//! projections folded in), `codec.decoder.pre_conv` (`[1,024, 3 · 512]`, taps oldest first),
//! `codec.decoder.input`, `blocks::transformer`'s under `codec.decoder.transformer`,
//! `codec.decoder.norm`, `codec.decoder.output`, and `blocks::code2wav`'s under `codec.vocoder`.

use blocks::code2wav::{self, Code2wav};
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture, Buffers, KvCache, Mask, Span, Transformer};
use bundle::{Load, Store};
use kernels::conv::{Patches, project_rows};
use kernels::ops::rms_norm;
use kernels::{Linear, Table, Vector};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::Error;
use crate::model::{CODE_GROUPS, CODES, FRAME_SAMPLES, Frame};

/// Channels of the quantizer's output and of the conv's input.
const QUANTIZED: usize = 512;
const LATENT: usize = 1024;
const PRE_CONV_TAPS: usize = 3;
/// The transformer attends to this many frames, itself included.
const WINDOW: usize = 72;
/// Frames per call of a stream.
pub const CALL_FRAMES: usize = 10;

const TRANSFORMER_EPS: f32 = 1e-5;
pub const TRANSFORMER: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen2,
    hidden: QUANTIZED,
    layers: 8,
    heads: 16,
    kv_heads: 16,
    head_dim: 64,
    feed_forward: 1024,
    rope_base: 1e4,
    rotation: Rotation::Sequence,
    norm_eps: TRANSFORMER_EPS,
};

pub const CODE2WAV: code2wav::Shape = code2wav::Shape {
    latent: LATENT,
    upsampling: &[2, 2],
    decoder: 1536,
    rates: &[8, 5, 4, 3],
    dilations: &[1, 3, 9],
    kernel: 7,
    expansion: 4,
    channels: 1,
    norm_eps: 1e-6,
};

pub(crate) struct Decoder {
    codebooks: Table,
    pre_conv: Linear,
    input: Linear,
    transformer: Transformer,
    norm: Vector,
    output: Linear,
    vocoder: Code2wav,
    /// Frames the vocoder reads before the first it keeps.
    history: usize,
}

/// One stream's state and activations.
pub(crate) struct Stream {
    cache: KvCache,
    buffers: Buffers,
    vocoder: code2wav::Stream,
    /// The conv's input rows: the last two of the call before, then this call's.
    rows: Vec<f32>,
    windows: SharedBuf,
    conv: SharedBuf,
    normed: SharedBuf,
    latents: SharedBuf,
    patches: PerThread<Patches>,
    position: usize,
}

impl Stream {
    pub(crate) fn new(decoder: &Decoder, threads: usize) -> Self {
        Self {
            cache: KvCache::new(TRANSFORMER, WINDOW + CALL_FRAMES),
            buffers: TRANSFORMER.buffers(CALL_FRAMES, threads),
            vocoder: decoder.vocoder.stream(CALL_FRAMES, threads),
            rows: vec![0.0; (PRE_CONV_TAPS - 1 + CALL_FRAMES) * QUANTIZED],
            windows: SharedBuf::new(CALL_FRAMES * PRE_CONV_TAPS * QUANTIZED),
            conv: SharedBuf::new(CALL_FRAMES * LATENT),
            normed: SharedBuf::new(CALL_FRAMES * QUANTIZED),
            latents: SharedBuf::new(CALL_FRAMES * LATENT),
            patches: PerThread::new(threads, Patches::default),
            position: 0,
        }
    }

    /// Starts a new stream (the KV ring needs no clearing: positions restart at 0 and every
    /// visible slot is written before it is read).
    pub(crate) fn reset(&mut self) {
        self.vocoder.reset();
        self.rows.fill(0.0);
        self.position = 0;
    }
}

impl Decoder {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            codebooks: Table::load(
                store,
                "codec.decoder.codebooks",
                [CODE_GROUPS * CODES, QUANTIZED],
            )?,
            pre_conv: Linear::load(
                store,
                "codec.decoder.pre_conv",
                [LATENT, PRE_CONV_TAPS * QUANTIZED],
            )?,
            input: Linear::load(store, "codec.decoder.input", [QUANTIZED, LATENT])?,
            transformer: Transformer::load(store, "codec.decoder.transformer", TRANSFORMER)?,
            norm: Vector::load(store, "codec.decoder.norm", QUANTIZED)?,
            output: Linear::load(store, "codec.decoder.output", [LATENT, QUANTIZED])?,
            vocoder: Code2wav::load(store, "codec.vocoder", CODE2WAV)?,
            history: CODE2WAV.history_frames(),
        })
    }

    /// Decodes `frames` (at most `CALL_FRAMES`), appending their samples to `out`.
    pub(crate) fn decode(
        &self,
        team: &Team,
        stream: &mut Stream,
        frames: &[Frame],
        out: &mut Vec<f32>,
    ) {
        self.transform(team, stream, frames);
        self.vocode(team, stream, frames.len());
        let samples = frames.len() * FRAME_SAMPLES;
        out.extend_from_slice(&stream.vocoder.output.read()[..samples]);
    }

    /// Starts `stream` on the reference clip `frames`: the transformer and the conv read them
    /// all, the vocoder the last `history_frames` of them, and no samples come out.
    pub(crate) fn warm(&self, team: &Team, stream: &mut Stream, frames: &[Frame]) {
        let (silent, heard) = frames.split_at(frames.len().saturating_sub(self.history));
        for call in silent.chunks(CALL_FRAMES) {
            self.transform(team, stream, call);
        }
        for call in heard.chunks(CALL_FRAMES) {
            self.transform(team, stream, call);
            self.vocode(team, stream, call.len());
        }
    }

    /// The vocoder's input rows for `frames`: their codes summed, the conv, the transformer, the
    /// final norm and the output projection.
    fn transform(&self, team: &Team, stream: &mut Stream, frames: &[Frame]) {
        let count = frames.len();
        assert!(
            count <= CALL_FRAMES,
            "a call of at most {CALL_FRAMES} frames"
        );
        self.gather(stream, frames);
        let first = stream.position;
        let shared = &*stream;
        team.run(|ctx| self.project(ctx, shared, (count, first)));
        stream.position += count;
    }

    fn vocode(&self, team: &Team, stream: &Stream, count: usize) {
        let vocoder = &stream.vocoder;
        team.run(|ctx| self.vocoder.forward(ctx, vocoder, &stream.latents, count));
    }

    /// The conv's windows over the rows of `frames`' summed codebook entries.
    fn gather(&self, stream: &mut Stream, frames: &[Frame]) {
        let history = (PRE_CONV_TAPS - 1) * QUANTIZED;
        let mut entry = vec![0.0; QUANTIZED];
        let matrix = self.codebooks.matrix();
        for (frame, row) in frames.iter().zip(
            stream.rows[history..]
                .as_chunks_mut::<QUANTIZED>()
                .0
                .iter_mut(),
        ) {
            row.fill(0.0);
            for (book, &code) in frame.iter().enumerate() {
                matrix.dequantize_row(book * CODES + code as usize, &mut entry);
                for (sum, value) in row.iter_mut().zip(&entry) {
                    *sum += value;
                }
            }
        }
        let width = PRE_CONV_TAPS * QUANTIZED;
        for index in 0..frames.len() {
            let window = stream.windows.part(index * width..(index + 1) * width);
            window.copy_from_slice(&stream.rows[index * QUANTIZED..index * QUANTIZED + width]);
        }
        stream.rows.copy_within(frames.len() * QUANTIZED.., 0);
    }

    /// Inside a team run: the windows through the conv, the input projection, the transformer
    /// from position `first`, the norm and the output projection into `stream.latents`.
    fn project(&self, ctx: &Ctx, stream: &Stream, (count, first): (usize, usize)) {
        let patches = stream.patches.get(ctx);
        let windows = (stream.windows.read(), count);
        project_rows(ctx, &self.pre_conv, windows, patches, |row, col, value| {
            stream.conv.set(row * LATENT + col, value);
        });
        ctx.barrier();
        let hidden = &stream.buffers.hidden;
        let conv = (stream.conv.read(), count);
        project_rows(ctx, &self.input, conv, patches, |row, col, value| {
            hidden.set(row * QUANTIZED + col, value);
        });
        ctx.barrier();
        let span = Span {
            rows: count,
            first,
            mask: Mask::Window(WINDOW),
        };
        self.transformer
            .forward(ctx, &stream.cache, &stream.buffers, span);
        for row in ctx.split(count, 1) {
            let span = row * QUANTIZED..(row + 1) * QUANTIZED;
            let out = stream.normed.part(span.clone());
            rms_norm(
                &hidden.read()[span],
                self.norm.values(),
                TRANSFORMER_EPS,
                out,
            );
        }
        ctx.barrier();
        let normed = (stream.normed.read(), count);
        project_rows(ctx, &self.output, normed, patches, |row, col, value| {
            stream.latents.set(row * LATENT + col, value);
        });
        ctx.barrier();
    }
}
