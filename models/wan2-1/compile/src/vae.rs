//! The decoder of Wan2.1's autoencoder (`Wan2.1_VAE.pth`) under `wan2_1::model::VAE`; its
//! encoder and `conv1` are not read.

use compile::{Checkpoint, Error, Tensors};
use wan2_1::model::{LATENT_MEAN, LATENT_STD, VAE, VAE_SHAPE};

use crate::Profile;

/// Where a residual block's parts are in the checkpoint, under the block.
const NORM1: &str = "residual.0.gamma";
const CONV1: &str = "residual.2";
const NORM2: &str = "residual.3.gamma";
const CONV2: &str = "residual.6";
const SHORTCUT: &str = "shortcut";

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

    /// The norm's `gamma` `source` as the vector `name`.
    fn gamma(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let values = self.checkpoint.f32(source)?;
        self.tensors.vector(&format!("{VAE}.{name}"), &values)
    }

    /// A residual block: `decoder.…` `source` as `name`.
    fn resnet(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let part = |part: &str| format!("{source}.{part}");
        self.gamma(&format!("{name}.norm1"), &part(NORM1))?;
        self.conv(&format!("{name}.conv1"), &part(CONV1))?;
        self.gamma(&format!("{name}.norm2"), &part(NORM2))?;
        self.conv(&format!("{name}.conv2"), &part(CONV2))?;
        let shortcut = part(SHORTCUT);
        if self.checkpoint.contains(&format!("{shortcut}.weight")) {
            self.conv(&format!("{name}.shortcut"), &shortcut)?;
        }
        Ok(())
    }

    /// `conv2`, the 1×1 convolution after the latents, with `z · std + mean` folded in:
    /// `W·(std ⊙ z + mean) + b`.
    fn post_quant(&mut self) -> Result<(), Error> {
        let weight = self.checkpoint.f32("conv2.weight")?;
        let bias = self.checkpoint.f32("conv2.bias")?;
        let name = format!("{VAE}.post_quant");
        self.tensors
            .linear_of_affine(&name, (&weight, bias), (&LATENT_STD, &LATENT_MEAN))
    }
}

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut tables = Tables {
        checkpoint,
        tensors,
    };
    tables.post_quant()?;
    tables.conv("conv_in", "decoder.conv1")?;
    tables.resnet("middle.first", "decoder.middle.0")?;
    tables.gamma("middle.attention.norm", "decoder.middle.1.norm.gamma")?;
    tables.conv("middle.attention.qkv", "decoder.middle.1.to_qkv")?;
    tables.conv("middle.attention.out", "decoder.middle.1.proj")?;
    tables.resnet("middle.second", "decoder.middle.2")?;
    let levels = VAE_SHAPE.channels.len();
    let level_stride = VAE_SHAPE.blocks + 1;
    for level in 0..levels {
        let upsample = |at: usize| format!("decoder.upsamples.{}", level * level_stride + at);
        for block in 0..VAE_SHAPE.blocks {
            let name = format!("up.{level}.resnets.{block}");
            tables.resnet(&name, &upsample(block))?;
        }
        let resample = upsample(VAE_SHAPE.blocks);
        if VAE_SHAPE.has_time(level) {
            tables.conv(
                &format!("up.{level}.time"),
                &format!("{resample}.time_conv"),
            )?;
        }
        if level + 1 < levels {
            tables.conv(
                &format!("up.{level}.resample"),
                &format!("{resample}.resample.1"),
            )?;
        }
    }
    tables.gamma("head.norm", "decoder.head.0.gamma")?;
    tables.conv("head.conv", "decoder.head.2")
}
