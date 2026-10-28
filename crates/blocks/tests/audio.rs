//! The audio blocks loaded from a bundle and run on the team.

#[expect(dead_code, reason = "the stack tests use the rest of the support")]
mod support;

use blocks::codec::{self, Decoder};
use blocks::depth::{self, Depth};
use blocks::hybrid::{AttentionShape, Gate, HeadNorm};
use blocks::interleaved::{self, Activation, Mixer, Norms, Window};
use blocks::rotary::Rotation;
use bundle::Load;
use kernels::DType;
use team::{SharedBuf, Team};

use support::{Bias, Weights, relative_error, widen};

const HIDDEN: usize = 16;
const HEAD_DIM: usize = 8;
const FEED_FORWARD: usize = 32;
const CONV_KERNEL: usize = 3;
const FRAMES: usize = 5;
/// Frames per call of the stream decoded in calls: 2, 2 and 1.
const CALL: usize = 2;

/// LFM2's layers in small: a gated short convolution, then ungated attention over a window.
const STACK: interleaved::Shape = interleaved::Shape {
    hidden: HIDDEN,
    mixers: &[Mixer::Conv, Mixer::Window],
    attention: AttentionShape {
        heads: 2,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rotary_dim: HEAD_DIM,
        rope_base: 10_000.0,
        rotation: Rotation::Sequence,
        head_norm: HeadNorm::Rms,
        gate: Gate::None,
    },
    window: Window {
        rows: 3,
        rotary_dim: HEAD_DIM,
        rope_base: 10_000.0,
        rotation: Rotation::Sequence,
    },
    conv_kernel: CONV_KERNEL,
    feed_forward: FEED_FORWARD,
    activation: Activation::Silu,
    norms: Norms::Pre,
    norm_eps: 1e-5,
};

const DEPTH: depth::Shape = depth::Shape {
    codebooks: 3,
    vocabulary: 8,
    input: 24,
    stack: interleaved::Shape {
        mixers: &[Mixer::Full],
        ..STACK
    },
};

const CODEC: codec::Shape = codec::Shape {
    codebooks: 2,
    levels: &[3, 4],
    latent: 16,
    stack: STACK,
    upsample: 2,
    n_fft: 8,
    hop: 2,
};

#[test]
fn codec_stream_decodes_in_calls_as_in_one() {
    let decoder = Decoder::load(&mut codec_weights().store(DType::F32), "codec", CODEC)
        .expect("the bundle holds every tensor the block loads");
    let codes: Vec<u32> = (0..FRAMES * CODEC.codebooks)
        .map(|index| (index * 7 % CODEC.vocabulary()) as u32)
        .collect();
    let whole = decode(&decoder, &codes, FRAMES);
    let in_calls = decode(&decoder, &codes, CALL);
    assert_eq!(whole.len(), in_calls.len());
    let error = relative_error(&in_calls, &widen(&whole));
    assert!(error < 1e-5, "error {error:.2e}");
}

/// Every sample of `codes`, decoded `call` frames at a time.
fn decode(decoder: &Decoder, codes: &[u32], call: usize) -> Vec<f32> {
    let team = Team::global();
    let mut stream = decoder.stream(call, FRAMES, team.threads());
    let mut samples = Vec::new();
    for frames in codes.chunks(call * CODEC.codebooks) {
        stream.decode(&team, decoder, frames, &mut samples);
    }
    stream.finish(&mut samples);
    samples
}

fn codec_weights() -> Weights {
    let mut weights = Weights::new("codec", 5);
    let latents = CODEC.codebooks * CODEC.levels.len();
    weights.linear("codec.project_out", [CODEC.latent, latents], Bias::With);
    weights.linear("codec.from_quantizer", [HIDDEN, CODEC.latent], Bias::With);
    stack_weights(&mut weights, "codec", STACK.mixers);
    weights.tensor("codec.norm", &[HIDDEN], 1.0, 0.1);
    let bins = CODEC.n_fft / 2 + 1;
    weights.linear("codec.spectrum", [2 * bins, HIDDEN], Bias::With);
    weights.tensor("codec.window", &[CODEC.n_fft], 0.75, 0.25);
    weights
}

/// The layers of a stack of `STACK`'s dimensions mixing by `mixers`, under `prefix`.
fn stack_weights(weights: &mut Weights, prefix: &str, mixers: &[Mixer]) {
    for (layer, mixer) in mixers.iter().enumerate() {
        let name = |part: &str| format!("{prefix}.layers.{layer}.{part}");
        let mixing = match mixer {
            Mixer::Conv => [("input", 3 * HIDDEN), ("out", HIDDEN)],
            Mixer::Full | Mixer::Window => [("qkv", 4 * HEAD_DIM), ("o", HIDDEN)],
        };
        for (part, outputs) in mixing {
            weights.linear(&name(part), [outputs, HIDDEN], Bias::Without);
        }
        match mixer {
            Mixer::Conv => weights.tensor(&name("conv"), &[HIDDEN, CONV_KERNEL], 0.0, 0.5),
            Mixer::Full | Mixer::Window => {
                weights.tensor(&name("q_norm"), &[HEAD_DIM], 1.0, 0.1);
                weights.tensor(&name("k_norm"), &[HEAD_DIM], 1.0, 0.1)
            }
        };
        for norm in ["input_norm", "post_norm"] {
            weights.tensor(&name(norm), &[HIDDEN], 1.0, 0.1);
        }
        for part in ["gate", "up"] {
            weights.linear(&name(part), [FEED_FORWARD, HIDDEN], Bias::Without);
        }
        weights.linear(&name("down"), [HIDDEN, FEED_FORWARD], Bias::Without);
    }
}

/// Every slot sees every other, so revealing the last codebook's code changes the first's
/// logits; and a pass keeps nothing of the passes before it.
#[test]
fn depth_decoder_scores_each_pass_over_every_slot() {
    let mut weights = Weights::new("depth", 9);
    let codebooks = DEPTH.codebooks;
    weights.linear("depth.slots", [codebooks * HIDDEN, DEPTH.input], Bias::With);
    weights.tensor("depth.mask", &[HIDDEN], 0.0, 1.0);
    stack_weights(&mut weights, "depth", DEPTH.stack.mixers);
    weights.tensor("depth.norm", &[HIDDEN], 1.0, 0.1);
    for codebook in 0..codebooks {
        let name = |part: &str| format!("depth.codebooks.{codebook}.{part}");
        weights.linear(
            &name("embedding"),
            [DEPTH.vocabulary, HIDDEN],
            Bias::Without,
        );
        weights.tensor(&name("norm"), &[HIDDEN], 1.0, 0.1);
    }
    let hidden = SharedBuf::new(DEPTH.input);
    hidden
        .part(0..DEPTH.input)
        .copy_from_slice(&weights.floats(DEPTH.input, 0.0, 1.0));
    let depth = Depth::load(&mut weights.store(DType::F32), "depth", DEPTH)
        .expect("the bundle holds every tensor the block loads");
    let team = Team::global();
    let buffers = depth::Buffers::new(&DEPTH, team.threads());
    let first_logits = |codes: &[Option<u32>]| {
        team.run(|ctx| {
            depth.slots(ctx, &buffers, &hidden);
            depth.score(ctx, &buffers, codes, &[0]);
        });
        buffers.predictions[0].logits.read().to_vec()
    };
    let masked = first_logits(&[None, None, None]);
    let revealed = first_logits(&[None, None, Some(5)]);
    assert!(relative_error(&revealed, &widen(&masked)) > 1e-3);
    assert_eq!(first_logits(&[None, None, None]), masked);
}
