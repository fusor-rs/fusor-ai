//! The speech tokenizer's encoder (Mimi's, in transformers): SEANet's encoder (`blocks::seanet`)
//! turns whole frames of 24 kHz samples into 25 Hz rows of 512 channels, 8 causal Kyutai layers
//! (`blocks::transformer`) read them, a conv halves their rate to the frames' 12.5 Hz, and a
//! split residual vector quantizer names each frame: one code of the semantic codebook, then 15 of
//! the acoustic residual's.
//!
//! The quantizer's search is over exact centroids in f32 (`codec.encoder.codebooks`).
//!
//! Tensors: `blocks::seanet`'s under `codec.encoder.seanet`, `blocks::transformer`'s under
//! `codec.encoder.transformer`; `codec.encoder.downsample` (`[512, 4 · 512]` taps, oldest first),
//! `codec.encoder.semantic` and `codec.encoder.acoustic` (`[256, 512]`), and
//! `codec.encoder.codebooks`, `[16 · 2048, 256]`, the semantic codebook then the acoustic ones.

use blocks::rotary::Rotation;
use blocks::seanet::{self, Direction, Seanet};
use blocks::transformer::{self, Architecture, Buffers, KvCache, Mask, Span, Transformer};
use bundle::{Load, Store};
use kernels::conv::{Patches, project_rows};
use kernels::{Linear, Table, dot_f32};
use team::{Ctx, PerThread, SharedBuf, Team};

use crate::Error;
use crate::model::{CODE_GROUPS, CODES, FRAME_SAMPLES, Frame, MAX_REFERENCE_SECONDS, SAMPLE_RATE};

const LATENT: usize = 512;
/// Channels of the quantizer's codebooks.
const QUANTIZED: usize = 256;

pub const SEANET: seanet::Shape = seanet::Shape {
    direction: Direction::Encode,
    dimension: LATENT,
    filters: 64,
    ratios: &[4, 5, 6, 8],
    kernel: 7,
    last_kernel: 3,
    residual_kernel: 3,
    compress: 2,
    channels: 1,
};

pub const TRANSFORMER: transformer::Shape = transformer::Shape {
    architecture: Architecture::Kyutai,
    hidden: LATENT,
    layers: 8,
    heads: 8,
    kv_heads: 8,
    head_dim: 64,
    feed_forward: 2048,
    rope_base: 1e4,
    rotation: Rotation::Sequence,
    norm_eps: 1e-5,
};

/// SEANet's rows per frame.
const ROWS_PER_FRAME: usize = 2;
/// Samples SEANet reads per call.
const ENCODE_CHUNK: usize = 8 * FRAME_SAMPLES;
/// The downsampling conv's taps, the rows each frame steps by, and the rows replicated before
/// the first.
const DOWNSAMPLE_TAPS: usize = 4;
const DOWNSAMPLE_PAD: usize = 2;
const MAX_FRAMES: usize = MAX_REFERENCE_SECONDS * SAMPLE_RATE as usize / FRAME_SAMPLES;

/// A clip's frames.
pub struct Encoded {
    /// The quantizer's input: `frames × 512` rows at 12.5 Hz.
    pub latent: Vec<f32>,
    pub codes: Vec<Frame>,
}

pub(crate) struct Encoder {
    seanet: Seanet,
    state: seanet::State,
    input: SharedBuf,
    transformer: Transformer,
    cache: KvCache,
    buffers: Buffers,
    downsample: Linear,
    semantic: Linear,
    acoustic: Linear,
    /// Every codebook's centroids, `CODES × QUANTIZED` each.
    codebooks: Vec<f32>,
    /// The squared length of every centroid.
    norms: Vec<f32>,
    /// Each frame's three-tap windows of the transformer's rows.
    windows: SharedBuf,
    latent: SharedBuf,
    /// Each frame's semantic and acoustic projections.
    projected: SharedBuf,
    codes: SharedBuf<u32>,
    patches: PerThread<Patches>,
}

impl Encoder {
    pub(crate) fn load(store: &mut Store, threads: usize) -> Result<Self, Error> {
        let seanet = Seanet::load(store, "codec.encoder.seanet", SEANET)?;
        let codebooks: Table = Table::load(
            store,
            "codec.encoder.codebooks",
            [CODE_GROUPS * CODES, QUANTIZED],
        )?;
        let codebooks = codebooks.dequantize();
        let norms = codebooks
            .as_chunks::<QUANTIZED>()
            .0
            .iter()
            .map(|centroid| dot_f32(centroid, centroid))
            .collect();
        Ok(Self {
            state: seanet::State::new(&seanet, ENCODE_CHUNK, threads),
            seanet,
            input: SharedBuf::new(ENCODE_CHUNK),
            transformer: Transformer::load(store, "codec.encoder.transformer", TRANSFORMER)?,
            cache: KvCache::new(TRANSFORMER, ROWS_PER_FRAME * MAX_FRAMES),
            buffers: TRANSFORMER.buffers(ROWS_PER_FRAME * MAX_FRAMES, threads),
            downsample: Linear::load(
                store,
                "codec.encoder.downsample",
                [LATENT, DOWNSAMPLE_TAPS * LATENT],
            )?,
            semantic: Linear::load(store, "codec.encoder.semantic", [QUANTIZED, LATENT])?,
            acoustic: Linear::load(store, "codec.encoder.acoustic", [QUANTIZED, LATENT])?,
            codebooks,
            norms,
            windows: SharedBuf::new(MAX_FRAMES * DOWNSAMPLE_TAPS * LATENT),
            latent: SharedBuf::new(MAX_FRAMES * LATENT),
            projected: SharedBuf::new(MAX_FRAMES * 2 * QUANTIZED),
            codes: SharedBuf::new(MAX_FRAMES * CODE_GROUPS),
            patches: PerThread::new(threads, Patches::default),
        })
    }

    /// The frames of `samples`, whole frames of at most `MAX_REFERENCE_SECONDS`.
    ///
    /// # Panics
    /// When `samples` are not whole frames, or more than the encoder holds.
    pub(crate) fn encode(&mut self, team: &Team, samples: &[f32]) -> Encoded {
        let frames = samples.len() / FRAME_SAMPLES;
        assert!(
            samples.len() == frames * FRAME_SAMPLES && frames <= MAX_FRAMES,
            "a clip of whole frames, at most {MAX_REFERENCE_SECONDS} s"
        );
        self.state.reset();
        let mut rows = 0;
        for chunk in samples.chunks(ENCODE_CHUNK) {
            self.input.part(0..chunk.len()).copy_from_slice(chunk);
            team.run(|ctx| {
                self.seanet
                    .forward(ctx, &self.state, &self.input, chunk.len())
            });
            let latents =
                &self.state.output.read()[..self.seanet.output_rows(chunk.len()) * LATENT];
            let end = rows + latents.len();
            self.buffers.hidden.part(rows..end).copy_from_slice(latents);
            rows = end;
        }
        team.run(|ctx| self.quantize(ctx, frames));
        let codes = self.codes.read()[..frames * CODE_GROUPS]
            .as_chunks()
            .0
            .to_vec();
        Encoded {
            latent: self.latent.read()[..frames * LATENT].to_vec(),
            codes,
        }
    }

    /// Inside a team run: the transformer over SEANet's rows in `buffers.hidden`, the
    /// downsampling conv, the quantizer's projections and the search.
    fn quantize(&self, ctx: &Ctx, frames: usize) {
        let span = Span {
            rows: ROWS_PER_FRAME * frames,
            first: 0,
            mask: Mask::Causal,
        };
        self.transformer
            .forward(ctx, &self.cache, &self.buffers, span);
        self.downsample(ctx, frames);
        self.project(ctx, frames);
        self.search(ctx, frames);
    }

    /// Each frame's four taps of the transformer's rows, the first frame's missing ones replicating
    /// row 0, through the conv into `latent`.
    fn downsample(&self, ctx: &Ctx, frames: usize) {
        let hidden = self.buffers.hidden.read();
        for frame in ctx.split(frames, 1) {
            for tap in 0..DOWNSAMPLE_TAPS {
                let row = (ROWS_PER_FRAME * frame + tap).saturating_sub(DOWNSAMPLE_PAD);
                let at = (frame * DOWNSAMPLE_TAPS + tap) * LATENT;
                let window = self.windows.part(at..at + LATENT);
                window.copy_from_slice(&hidden[row * LATENT..(row + 1) * LATENT]);
            }
        }
        ctx.barrier();
        let patches = self.patches.get(ctx);
        let windows = (self.windows.read(), frames);
        project_rows(
            ctx,
            &self.downsample,
            windows,
            patches,
            |row, col, value| {
                self.latent.set(row * LATENT + col, value);
            },
        );
        ctx.barrier();
    }

    /// The semantic and acoustic projections of `latent` into `projected`.
    fn project(&self, ctx: &Ctx, frames: usize) {
        let patches = self.patches.get(ctx);
        let latent = (self.latent.read(), frames);
        for (part, linear) in [&self.semantic, &self.acoustic].into_iter().enumerate() {
            project_rows(ctx, linear, latent, patches, |row, col, value| {
                self.projected
                    .set((row * 2 + part) * QUANTIZED + col, value);
            });
        }
        ctx.barrier();
    }

    /// This thread's frames' codes: the semantic codebook's nearest centroid to the semantic
    /// projection, then each acoustic codebook's to the acoustic projection less the centroids
    /// before.
    fn search(&self, ctx: &Ctx, frames: usize) {
        let projected = self.projected.read();
        for frame in ctx.split(frames, 1) {
            let (semantic, acoustic) =
                projected[frame * 2 * QUANTIZED..][..2 * QUANTIZED].split_at(QUANTIZED);
            let codes = self
                .codes
                .part(frame * CODE_GROUPS..(frame + 1) * CODE_GROUPS);
            codes[0] = self.nearest(0, semantic);
            let mut residual = [0.0; QUANTIZED];
            residual.copy_from_slice(acoustic);
            for (book, code) in codes.iter_mut().enumerate().skip(1) {
                *code = self.nearest(book, &residual);
                let centroid = self.centroid(book, *code);
                for (value, taken) in residual.iter_mut().zip(centroid) {
                    *value -= taken;
                }
            }
        }
        ctx.barrier();
    }

    fn centroid(&self, book: usize, code: u32) -> &[f32] {
        let start = (book * CODES + code as usize) * QUANTIZED;
        &self.codebooks[start..start + QUANTIZED]
    }

    /// The code of `book` whose centroid is nearest `vector`, the first of equals: the least
    /// `|centroid|² − 2 · centroid · vector`.
    fn nearest(&self, book: usize, vector: &[f32]) -> u32 {
        let norms = &self.norms[book * CODES..(book + 1) * CODES];
        let distance =
            |code: usize| norms[code] - 2.0 * dot_f32(self.centroid(book, code as u32), vector);
        let mut best = 0;
        let mut least = distance(0);
        for code in 1..CODES {
            let candidate = distance(code);
            if candidate < least {
                best = code;
                least = candidate;
            }
        }
        best as u32
    }
}
