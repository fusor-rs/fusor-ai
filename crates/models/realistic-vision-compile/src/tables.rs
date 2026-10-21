//! The runtime tensors (`realistic_vision::model`'s names) computed from the checkpoints', with
//! the folds the crate docs list.

use compile::{Checkpoint, Error, Tensors};
use realistic_vision::model::{LATENT_SCALING, TEXT_SHAPE, UNET_SHAPE, VAE_SHAPE};

use crate::Profile;

const TEXT: &str = "cond_stage_model.transformer.text_model";
const UNET: &str = "model.diffusion_model";
const DECODER: &str = "decoder";

/// The checkpoint's tensors under a prefix, written under the runtime's names.
struct Tables<'call, 'bundle> {
    checkpoint: &'call Checkpoint,
    tensors: &'call mut Tensors<'bundle, Profile>,
}

impl Tables<'_, '_> {
    /// The checkpoint's norm or linear layer (or pointwise convolution) `source` as `name`.
    fn copy(&mut self, name: &str, source: &str) -> Result<(), Error> {
        self.tensors.copy_linear(self.checkpoint, name, source)
    }

    fn norm(&mut self, name: &str, source: &str) -> Result<(), Error> {
        self.tensors.copy_norm(self.checkpoint, name, source)
    }

    /// The linear layers `sources` (biases where they have them), their rows stacked, as `name`.
    fn stacked(&mut self, name: &str, sources: &[String]) -> Result<(), Error> {
        let (mut weight, mut bias) = (Vec::new(), Vec::new());
        for source in sources {
            weight.extend(self.checkpoint.f32(&format!("{source}.weight"))?);
            let bias_name = format!("{source}.bias");
            if self.checkpoint.contains(&bias_name) {
                bias.extend(self.checkpoint.f32(&bias_name)?);
            }
        }
        let rows = sources.iter().try_fold(0, |rows, source| {
            Ok::<_, Error>(rows + self.checkpoint.shape(&format!("{source}.weight"))?[0])
        })?;
        let bias = (!bias.is_empty()).then_some(&bias[..]);
        self.tensors.linear(name, rows, &weight, bias)
    }

    /// The convolution `source` (`[outputs, inputs, k, k]`) as the matrix `kernels::conv`
    /// reads: `[outputs, k² · inputs]`, tap-major.
    fn conv(&mut self, name: &str, source: &str) -> Result<(), Error> {
        let weight_name = format!("{source}.weight");
        let &[outputs, inputs, height, width] = self.checkpoint.shape(&weight_name)? else {
            return Err(Error::Format(format!(
                "{weight_name}: a 4-d convolution expected"
            )));
        };
        let weight = self.checkpoint.f32(&weight_name)?;
        let taps = height * width;
        let matrix: Vec<f32> = (0..weight.len())
            .map(|at| {
                let (output, column) = (at / (taps * inputs), at % (taps * inputs));
                let (tap, input) = (column / inputs, column % inputs);
                weight[(output * inputs + input) * taps + tap]
            })
            .collect();
        let bias = self.checkpoint.f32(&format!("{source}.bias"))?;
        self.tensors.linear(name, outputs, &matrix, Some(&bias))
    }

    /// An LDM residual block (`ResBlock` or the autoencoder's `ResnetBlock`) as `resnet`'s names.
    fn resnet(&mut self, name: &str, source: &str, parts: &ResnetParts) -> Result<(), Error> {
        let path = |part: &str| format!("{source}.{part}");
        self.norm(&format!("{name}.norm1"), &path(parts.norm1))?;
        self.conv(&format!("{name}.conv1"), &path(parts.conv1))?;
        if let Some(time) = parts.time {
            self.copy(&format!("{name}.time"), &path(time))?;
        }
        self.norm(&format!("{name}.norm2"), &path(parts.norm2))?;
        self.conv(&format!("{name}.conv2"), &path(parts.conv2))?;
        let shortcut = path(parts.shortcut);
        if self.checkpoint.contains(&format!("{shortcut}.weight")) {
            self.copy(&format!("{name}.shortcut"), &shortcut)?;
        }
        Ok(())
    }
}

/// Where a residual block's parts are in a checkpoint.
struct ResnetParts {
    norm1: &'static str,
    conv1: &'static str,
    time: Option<&'static str>,
    norm2: &'static str,
    conv2: &'static str,
    shortcut: &'static str,
}

/// The UNet's `ResBlock`.
const UNET_RESNET: ResnetParts = ResnetParts {
    norm1: "in_layers.0",
    conv1: "in_layers.2",
    time: Some("emb_layers.1"),
    norm2: "out_layers.0",
    conv2: "out_layers.3",
    shortcut: "skip_connection",
};

/// The autoencoder's `ResnetBlock`.
const VAE_RESNET: ResnetParts = ResnetParts {
    norm1: "norm1",
    conv1: "conv1",
    time: None,
    norm2: "norm2",
    conv2: "conv2",
    shortcut: "nin_shortcut",
};

/// CLIP's text encoder: each layer's queries, keys and values stacked.
pub(crate) fn text(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut tables = Tables {
        checkpoint,
        tensors,
    };
    let embeddings = format!("{TEXT}.embeddings");
    let embedding = checkpoint.f32(&format!("{embeddings}.token_embedding.weight"))?;
    let hidden = TEXT_SHAPE.hidden;
    let shape = vec![embedding.len() / hidden, hidden];
    tables.tensors.add("text.embedding", shape, &embedding)?;
    let positions = checkpoint.f32(&format!("{embeddings}.position_embedding.weight"))?;
    let shape = vec![positions.len() / hidden, hidden];
    tables.tensors.add("text.positions", shape, &positions)?;
    for layer in 0..TEXT_SHAPE.layers {
        let (name, source) = (
            format!("text.layers.{layer}"),
            format!("{TEXT}.encoder.layers.{layer}"),
        );
        let attention = |part: &str| format!("{source}.self_attn.{part}_proj");
        tables.norm(&format!("{name}.norm1"), &format!("{source}.layer_norm1"))?;
        tables.stacked(&format!("{name}.qkv"), &["q", "k", "v"].map(attention))?;
        tables.copy(&format!("{name}.out"), &attention("out"))?;
        tables.norm(&format!("{name}.norm2"), &format!("{source}.layer_norm2"))?;
        tables.copy(&format!("{name}.ff1"), &format!("{source}.mlp.fc1"))?;
        tables.copy(&format!("{name}.ff2"), &format!("{source}.mlp.fc2"))?;
    }
    tables.norm("text.norm", &format!("{TEXT}.final_layer_norm"))
}

/// The UNet, from LDM's `input_blocks`, `middle_block` and `output_blocks`.
pub(crate) fn unet(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut tables = Tables {
        checkpoint,
        tensors,
    };
    tables.copy("unet.time_in", &format!("{UNET}.time_embed.0"))?;
    tables.copy("unet.time_out", &format!("{UNET}.time_embed.2"))?;
    tables.conv("unet.conv_in", &format!("{UNET}.input_blocks.0.0"))?;
    unet_down(&mut tables)?;
    let middle = format!("{UNET}.middle_block");
    tables.resnet("unet.middle.first", &format!("{middle}.0"), &UNET_RESNET)?;
    let depth = UNET_SHAPE.middle_depth;
    spatial(
        &mut tables,
        ("unet.middle.transformer", depth),
        &format!("{middle}.1"),
    )?;
    tables.resnet("unet.middle.second", &format!("{middle}.2"), &UNET_RESNET)?;
    unet_up(&mut tables)?;
    tables.norm("unet.norm_out", &format!("{UNET}.out.0"))?;
    tables.conv("unet.conv_out", &format!("{UNET}.out.2"))
}

/// `input_blocks` after `conv_in`: per level, `blocks` (residual block, transformer) pairs, then
/// the downsampling convolution.
fn unet_down(tables: &mut Tables) -> Result<(), Error> {
    let (levels, blocks) = (UNET_SHAPE.channels.len(), UNET_SHAPE.blocks);
    for level in 0..levels {
        let depth = UNET_SHAPE.depths[level];
        let name = format!("unet.down.{level}");
        for block in 0..blocks {
            let source = format!("{UNET}.input_blocks.{}", 1 + level * (blocks + 1) + block);
            tables.resnet(
                &format!("{name}.resnets.{block}"),
                &format!("{source}.0"),
                &UNET_RESNET,
            )?;
            if depth > 0 {
                let transformer = format!("{name}.transformers.{block}");
                spatial(tables, (&transformer, depth), &format!("{source}.1"))?;
            }
        }
        if level + 1 < levels {
            let source = format!("{UNET}.input_blocks.{}.0.op", (level + 1) * (blocks + 1));
            tables.conv(&format!("{name}.resample"), &source)?;
        }
    }
    Ok(())
}

/// `output_blocks`, the deepest level first: per level, `blocks + 1` (residual block,
/// transformer) pairs, the last followed by the upsampling convolution.
fn unet_up(tables: &mut Tables) -> Result<(), Error> {
    let (levels, blocks) = (UNET_SHAPE.channels.len(), UNET_SHAPE.blocks);
    for (order, level) in (0..levels).rev().enumerate() {
        let depth = UNET_SHAPE.depths[level];
        let name = format!("unet.up.{level}");
        for block in 0..=blocks {
            let source = format!("{UNET}.output_blocks.{}", order * (blocks + 1) + block);
            tables.resnet(
                &format!("{name}.resnets.{block}"),
                &format!("{source}.0"),
                &UNET_RESNET,
            )?;
            if depth > 0 {
                let transformer = format!("{name}.transformers.{block}");
                spatial(tables, (&transformer, depth), &format!("{source}.1"))?;
            }
        }
        if level > 0 {
            let last = format!("{UNET}.output_blocks.{}", order * (blocks + 1) + blocks);
            // The upsampler follows the block's residual block and its transformer, if any.
            let part = 1 + usize::from(depth > 0);
            tables.conv(&format!("{name}.resample"), &format!("{last}.{part}.conv"))?;
        }
    }
    Ok(())
}

/// A `SpatialTransformer` as `unet::transformer`'s names: its attentions' projections stacked,
/// its GEGLU projection split into the hidden and gate halves.
fn spatial(tables: &mut Tables, (name, depth): (&str, usize), source: &str) -> Result<(), Error> {
    tables.norm(&format!("{name}.norm"), &format!("{source}.norm"))?;
    tables.copy(&format!("{name}.proj_in"), &format!("{source}.proj_in"))?;
    tables.copy(&format!("{name}.proj_out"), &format!("{source}.proj_out"))?;
    for block in 0..depth {
        let (name, source) = (
            format!("{name}.blocks.{block}"),
            format!("{source}.transformer_blocks.{block}"),
        );
        let part = |part: &str| format!("{source}.{part}");
        for norm in ["norm1", "norm2", "norm3"] {
            tables.norm(&format!("{name}.{norm}"), &part(norm))?;
        }
        tables.stacked(
            &format!("{name}.qkv"),
            &["to_q", "to_k", "to_v"].map(|to| part(&format!("attn1.{to}"))),
        )?;
        tables.copy(&format!("{name}.out1"), &part("attn1.to_out.0"))?;
        tables.copy(&format!("{name}.query"), &part("attn2.to_q"))?;
        tables.stacked(
            &format!("{name}.context"),
            &["to_k", "to_v"].map(|to| part(&format!("attn2.{to}"))),
        )?;
        tables.copy(&format!("{name}.out2"), &part("attn2.to_out.0"))?;
        geglu(tables, &name, &part("ff.net.0.proj"))?;
        tables.copy(&format!("{name}.down"), &part("ff.net.2"))?;
    }
    Ok(())
}

/// The GEGLU's projection `[2 · inner, channels]`: its first `inner` rows as `name.hidden`, the
/// rest as `name.gate`.
fn geglu(tables: &mut Tables, name: &str, source: &str) -> Result<(), Error> {
    let weight = tables.checkpoint.f32(&format!("{source}.weight"))?;
    let bias = tables.checkpoint.f32(&format!("{source}.bias"))?;
    let inner = bias.len() / 2;
    let (hidden, gate) = weight.split_at(weight.len() / 2);
    let (hidden_bias, gate_bias) = bias.split_at(inner);
    tables
        .tensors
        .linear(&format!("{name}.hidden"), inner, hidden, Some(hidden_bias))?;
    tables
        .tensors
        .linear(&format!("{name}.gate"), inner, gate, Some(gate_bias))
}

/// The autoencoder's decoder, its latents' scaling folded into `post_quant`.
pub(crate) fn vae(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut tables = Tables {
        checkpoint,
        tensors,
    };
    let weight = checkpoint.f32("post_quant_conv.weight")?;
    let bias = checkpoint.f32("post_quant_conv.bias")?;
    let scaled: Vec<f32> = weight.iter().map(|value| value / LATENT_SCALING).collect();
    tables
        .tensors
        .linear("vae.post_quant", bias.len(), &scaled, Some(&bias))?;
    tables.conv("vae.conv_in", &format!("{DECODER}.conv_in"))?;
    let middle = format!("{DECODER}.mid");
    tables.resnet(
        "vae.middle.first",
        &format!("{middle}.block_1"),
        &VAE_RESNET,
    )?;
    let attention = format!("{middle}.attn_1");
    tables.norm("vae.middle.attention.norm", &format!("{attention}.norm"))?;
    let parts = ["q", "k", "v"].map(|part| format!("{attention}.{part}"));
    tables.stacked("vae.middle.attention.qkv", &parts)?;
    tables.copy("vae.middle.attention.out", &format!("{attention}.proj_out"))?;
    tables.resnet(
        "vae.middle.second",
        &format!("{middle}.block_2"),
        &VAE_RESNET,
    )?;
    for level in 0..VAE_SHAPE.channels.len() {
        let (name, source) = (format!("vae.up.{level}"), format!("{DECODER}.up.{level}"));
        for block in 0..=VAE_SHAPE.blocks {
            let resnet = format!("{name}.resnets.{block}");
            tables.resnet(&resnet, &format!("{source}.block.{block}"), &VAE_RESNET)?;
        }
        if level > 0 {
            tables.conv(
                &format!("{name}.resample"),
                &format!("{source}.upsample.conv"),
            )?;
        }
    }
    tables.norm("vae.norm_out", &format!("{DECODER}.norm_out"))?;
    tables.conv("vae.conv_out", &format!("{DECODER}.conv_out"))
}
