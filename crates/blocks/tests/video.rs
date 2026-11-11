//! The video blocks loaded from a bundle and run on the team.

#[expect(dead_code, reason = "the stack tests use the rest of the support")]
mod support;

use blocks::wan_vae::{Capacity, Decoder, Shape};
use bundle::Load;
use kernels::DType;
use kernels::conv::Grid;
use team::Team;

use support::Weights;

/// Wan2.1's decoder in small: three levels, the first doubling the frames.
const SHAPE: Shape = Shape {
    latent: 4,
    channels: &[32, 32, 16],
    blocks: 2,
    temporal: &[true, false],
    outputs: 3,
    kernel: 3,
};
/// Taps of a causal convolution: frames, then the window's rows and columns.
const TAPS: usize = 27;
const LATENT_FRAMES: usize = 3;
/// Tall enough that a strip's windows cut the frame at every level, with both cuts inside it.
const GRID: Grid = Grid {
    images: 1,
    height: 26,
    width: 5,
};

/// Linear `name`, `[outputs, inputs]`, with a bias.
fn linear(weights: &mut Weights, name: &str, [outputs, inputs]: [usize; 2]) {
    let scale = 1.0 / (inputs as f64).sqrt();
    weights.tensor(&format!("{name}.weight"), &[outputs, inputs], 0.0, scale);
    weights.tensor(&format!("{name}.bias"), &[outputs], 0.0, 0.1);
}

fn resnet(weights: &mut Weights, prefix: &str, inputs: usize, outputs: usize) {
    weights.tensor(&format!("{prefix}.norm1"), &[inputs], 1.0, 0.2);
    linear(
        weights,
        &format!("{prefix}.conv1"),
        [outputs, TAPS * inputs],
    );
    weights.tensor(&format!("{prefix}.norm2"), &[outputs], 1.0, 0.2);
    linear(
        weights,
        &format!("{prefix}.conv2"),
        [outputs, TAPS * outputs],
    );
    if inputs != outputs {
        linear(weights, &format!("{prefix}.shortcut"), [outputs, inputs]);
    }
}

/// Every tensor of `SHAPE`'s decoder under `vae`.
fn decoder_weights() -> Weights {
    let Shape {
        latent,
        channels,
        outputs,
        ..
    } = SHAPE;
    let mut weights = Weights::new("wan-vae", 3);
    linear(&mut weights, "vae.post_quant", [latent, latent]);
    linear(&mut weights, "vae.conv_in", [channels[0], TAPS * latent]);
    resnet(&mut weights, "vae.middle.first", channels[0], channels[0]);
    weights.tensor("vae.middle.attention.norm", &[channels[0]], 1.0, 0.2);
    linear(
        &mut weights,
        "vae.middle.attention.qkv",
        [3 * channels[0], channels[0]],
    );
    linear(
        &mut weights,
        "vae.middle.attention.out",
        [channels[0], channels[0]],
    );
    resnet(&mut weights, "vae.middle.second", channels[0], channels[0]);
    let mut inputs = channels[0];
    for (level, &width) in channels.iter().enumerate() {
        for block in 0..SHAPE.blocks {
            let block_inputs = if block == 0 { inputs } else { width };
            resnet(
                &mut weights,
                &format!("vae.up.{level}.resnets.{block}"),
                block_inputs,
                width,
            );
        }
        if SHAPE.temporal.get(level) == Some(&true) {
            linear(
                &mut weights,
                &format!("vae.up.{level}.time"),
                [2 * width, SHAPE.kernel * width],
            );
        }
        if level + 1 < channels.len() {
            let resample = [width / 2, SHAPE.kernel * SHAPE.kernel * width];
            linear(&mut weights, &format!("vae.up.{level}.resample"), resample);
        }
        inputs = width / 2;
    }
    let last = channels[channels.len() - 1];
    weights.tensor("vae.head.norm", &[last], 1.0, 0.2);
    linear(&mut weights, "vae.head.conv", [outputs, TAPS * last]);
    weights
}

/// The video of `latents` decoded `strip` latent rows at a time: frame after frame, pixel-major.
fn decode(decoder: &Decoder, latents: &[f32], strip: usize) -> Vec<f32> {
    let team = Team::global();
    let capacity = Capacity {
        pixels: GRID.area(),
        width: GRID.width,
        video: LATENT_FRAMES * GRID.area(),
        strip,
    };
    let mut buffers = SHAPE.decoder_buffers(capacity, team.threads());
    let values = GRID.area() * SHAPE.latent;
    for (index, frame) in latents.chunks_exact(values).enumerate() {
        buffers.latent.get_mut()[..values].copy_from_slice(frame);
        team.run(|ctx| decoder.decode_latents(ctx, &buffers, (GRID, index)));
    }
    let scale = SHAPE.scale();
    let row = GRID.width * scale * SHAPE.outputs;
    let frame = GRID.height * scale * row;
    let frames: usize = (0..LATENT_FRAMES).map(|index| SHAPE.frames(index)).sum();
    let mut video = vec![0.0; frames * frame];
    for start in (0..GRID.height).step_by(strip) {
        let rows = start..(start + strip).min(GRID.height);
        let mut first = 0;
        for index in 0..LATENT_FRAMES {
            team.run(|ctx| decoder.decode_strip(ctx, &buffers, (GRID, index), rows.clone()));
            let band = rows.len() * scale * row;
            let decoded = buffers.frames.read().chunks_exact(band);
            for (offset, values) in (first..first + SHAPE.frames(index)).zip(decoded) {
                let at = offset * frame + rows.start * scale * row;
                video[at..at + band].copy_from_slice(values);
            }
            first += SHAPE.frames(index);
        }
    }
    video
}

#[test]
fn strips_decode_the_frames_bit_for_bit() {
    let mut store = decoder_weights().store(DType::Q7b32);
    let decoder =
        Decoder::load(&mut store, "vae", SHAPE).expect("the bundle holds every tensor it loads");
    let latents = Weights::new("wan-vae-latents", 5).floats(
        LATENT_FRAMES * GRID.area() * SHAPE.latent,
        0.0,
        1.0,
    );
    let whole = decode(&decoder, &latents, GRID.height);
    for strip in [4, 13] {
        let strips = decode(&decoder, &latents, strip);
        let differing = (strips.iter().zip(&whole))
            .filter(|(strip, frame)| strip.to_bits() != frame.to_bits())
            .count();
        assert_eq!(differing, 0, "strips of {strip} latent rows");
    }
}
