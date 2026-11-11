//! The autoencoder (`vae/diffusion_pytorch_model.safetensors`, diffusers' names) under
//! `motif_video::model::VAE`: its decoder, and its encoder cut to a video's first frame.

use compile::{Checkpoint, Error, Tensors};
use motif_video::model::{LATENT_MEAN, LATENT_STD, VAE, VAE_SHAPE};

use crate::Profile;

/// Where a residual block's parts are in the checkpoint, under the block.
const NORM1: &str = "norm1.gamma";
const CONV1: &str = "conv1";
const NORM2: &str = "norm2.gamma";
const CONV2: &str = "conv2";
const SHORTCUT: &str = "conv_shortcut";

/// Which frames of a causal convolution's kernel are written.
#[derive(Clone, Copy)]
enum Frames {
    All,
    /// The kernel's last frame: the only one a video's first frame meets, after two zero frames.
    Last,
}

/// The checkpoint's tensors written under the runtime's names.
struct Tables<'call, 'bundle> {
    checkpoint: &'call Checkpoint,
    tensors: &'call mut Tensors<'bundle, Profile>,
}

impl Tables<'_, '_> {
    /// The convolution `source`, of any rank, as the matrix `name`.
    fn conv(&mut self, name: &str, source: &str) -> Result<(), Error> {
        self.tensors
            .conv(self.checkpoint, &format!("{VAE}.{name}"), source)
    }

    /// The 3-D convolution `source` as `name`, with `frames` of its kernel.
    fn conv3d(&mut self, name: &str, source: &str, frames: Frames) -> Result<(), Error> {
        match frames {
            Frames::All => self.conv(name, source),
            Frames::Last => self.last_frame(name, source),
        }
    }

    /// The 3-D convolution `source` `[out, in, t, h, w]` with only its last frame tap, as the
    /// matrix `name` `[out, h · w · in]`: columns tap by tap, row-major, then channel.
    fn last_frame(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let weight_name = format!("{source}.weight");
        let &[outputs, inputs, frames, height, width] = self.checkpoint.shape(&weight_name)? else {
            return Err(Error::Format(format!(
                "{weight_name}: [out, in, t, h, w] expected"
            )));
        };
        let weight = self.checkpoint.f32(&weight_name)?;
        let bias = self.checkpoint.f32(&format!("{source}.bias"))?;
        let taps = height * width;
        let mut kept = Vec::with_capacity(outputs * taps * inputs);
        for output in 0..outputs {
            for tap in 0..taps {
                let last = |input: usize| {
                    weight[((output * inputs + input) * frames + frames - 1) * taps + tap]
                };
                kept.extend((0..inputs).map(last));
            }
        }
        self.tensors
            .linear(&format!("{VAE}.{name}"), outputs, &kept, Some(&bias))
    }

    /// The encoder's stride-2 resample `source` `[c, c, 3, 3]` as `name` `[c, 25 · c]`: its 3 × 3
    /// window at offset 2 of a window of `Shape::encoder_resample` rows and columns, zeros
    /// elsewhere.
    fn resample(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let weight_name = format!("{source}.weight");
        let &[outputs, inputs, height, width] = self.checkpoint.shape(&weight_name)? else {
            return Err(Error::Format(format!(
                "{weight_name}: [out, in, h, w] expected"
            )));
        };
        let weight = self.checkpoint.f32(&weight_name)?;
        let bias = self.checkpoint.f32(&format!("{source}.bias"))?;
        let side = VAE_SHAPE.encoder_resample();
        let offset = side / 2;
        let row = side * side * inputs;
        let mut window = vec![0.0; outputs * row];
        for output in 0..outputs {
            for tap in 0..height * width {
                let at = ((tap / width + offset) * side + tap % width + offset) * inputs;
                for input in 0..inputs {
                    window[output * row + at + input] =
                        weight[(output * inputs + input) * height * width + tap];
                }
            }
        }
        self.tensors
            .linear(&format!("{VAE}.{name}"), outputs, &window, Some(&bias))
    }

    /// The norm's `gamma` `source` as the vector `name`.
    fn gamma(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let values = self.checkpoint.f32(source)?;
        self.tensors.vector(&format!("{VAE}.{name}"), &values)
    }

    /// A residual block `source` as `name`.
    fn resnet(&mut self, name: &str, source: &str, frames: Frames) -> Result<(), Error> {
        let part = |part: &str| format!("{source}.{part}");
        self.gamma(&format!("{name}.norm1"), &part(NORM1))?;
        self.conv3d(&format!("{name}.conv1"), &part(CONV1), frames)?;
        self.gamma(&format!("{name}.norm2"), &part(NORM2))?;
        self.conv3d(&format!("{name}.conv2"), &part(CONV2), frames)?;
        let shortcut = part(SHORTCUT);
        if self.checkpoint.contains(&format!("{shortcut}.weight")) {
            self.conv3d(&format!("{name}.shortcut"), &shortcut, frames)?;
        }
        Ok(())
    }

    /// A middle block `source` as `name`: two residual blocks around an attention.
    fn middle(&mut self, name: &str, source: &str, frames: Frames) -> Result<(), Error> {
        self.resnet(
            &format!("{name}.first"),
            &format!("{source}.resnets.0"),
            frames,
        )?;
        let attention = format!("{source}.attentions.0");
        self.gamma(
            &format!("{name}.attention.norm"),
            &format!("{attention}.norm.gamma"),
        )?;
        self.conv(
            &format!("{name}.attention.qkv"),
            &format!("{attention}.to_qkv"),
        )?;
        self.conv(
            &format!("{name}.attention.out"),
            &format!("{attention}.proj"),
        )?;
        self.resnet(
            &format!("{name}.second"),
            &format!("{source}.resnets.1"),
            frames,
        )
    }

    /// `post_quant_conv`, the 1×1 convolution after the latents, with `z · std + mean` folded in.
    fn post_quant(&mut self) -> Result<(), Error> {
        let weight = self.checkpoint.f32("post_quant_conv.weight")?;
        let bias = self.checkpoint.f32("post_quant_conv.bias")?;
        self.tensors.linear_of_affine(
            &format!("{VAE}.post_quant"),
            (&weight, bias),
            (&LATENT_STD, &LATENT_MEAN),
        )
    }

    fn decoder(&mut self) -> Result<(), Error> {
        self.post_quant()?;
        self.conv("conv_in", "decoder.conv_in")?;
        self.middle("middle", "decoder.mid_block", Frames::All)?;
        let levels = VAE_SHAPE.channels.len();
        for level in 0..levels {
            let source = format!("decoder.up_blocks.{level}");
            for block in 0..VAE_SHAPE.blocks {
                let name = format!("up.{level}.resnets.{block}");
                self.resnet(&name, &format!("{source}.resnets.{block}"), Frames::All)?;
            }
            let upsampler = format!("{source}.upsamplers.0");
            if VAE_SHAPE.has_time(level) {
                self.conv(
                    &format!("up.{level}.time"),
                    &format!("{upsampler}.time_conv"),
                )?;
            }
            if level + 1 < levels {
                self.conv(
                    &format!("up.{level}.resample"),
                    &format!("{upsampler}.resample.1"),
                )?;
            }
        }
        self.gamma("head.norm", "decoder.norm_out.gamma")?;
        self.conv("head.conv", "decoder.conv_out")
    }

    /// The first-frame encoder: each level's residual blocks then a resample, but the last
    /// level's; the checkpoint numbers a level's blocks consecutively.
    fn encoder(&mut self) -> Result<(), Error> {
        self.conv3d("encoder.conv_in", "encoder.conv_in", Frames::Last)?;
        let levels = VAE_SHAPE.channels.len();
        let resnets = VAE_SHAPE.blocks - 1;
        for level in 0..levels {
            let first = level * VAE_SHAPE.blocks;
            for block in 0..resnets {
                let name = format!("encoder.down.{level}.resnets.{block}");
                let source = format!("encoder.down_blocks.{}", first + block);
                self.resnet(&name, &source, Frames::Last)?;
            }
            if level + 1 < levels {
                let name = format!("encoder.down.{level}.resample");
                let source = format!("encoder.down_blocks.{}.resample.1", first + resnets);
                self.resample(&name, &source)?;
            }
        }
        self.middle("encoder.middle", "encoder.mid_block", Frames::Last)?;
        self.gamma("encoder.head.norm", "encoder.norm_out.gamma")?;
        self.conv3d("encoder.head.conv", "encoder.conv_out", Frames::Last)?;
        self.conv("encoder.quant", "quant_conv")
    }
}

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut tables = Tables {
        checkpoint,
        tensors,
    };
    tables.decoder()?;
    tables.encoder()
}
