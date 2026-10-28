//! `d1 compile`: a d1-2B or d1-Omni-2B checkpoint (one safetensors file, bf16, its
//! `config.json`, which names the variant, its `calibration.json` and its tokenizer) → a browser
//! bundle.
//!
//! Renames to the runtime's vocabulary (`d1::model`), fuses each attention layer's query, key,
//! value and gate projections (one gate row per query head, last) into one, and each vision and
//! audio attention layer's query, key and value projections; lays d1-Omni-2B's audio
//! downsampler out as `blocks::audio`'s subsampler reads it (its 3 × 3 kernels tap-major over
//! time then frequency, its last linear's inputs frequency-major); quantizes per profile.
//! fusor-ai's `BundleWriter` packs and hashes the parts and writes `manifest.json`. The
//! checkpoint's speech output (its codebooks and depth decoder) is d1's no part.

use std::path::Path;

use blocks::interleaved::Mixer;
use blocks::vit::interleaved::Mixer as VisionMixer;
use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use d1::model::{AUDIO, CALIBRATION, CONFIG, Config, LANGUAGE, MELS, TOWER, Variant};
use kernels::DType;

const WEIGHTS: &str = "model.safetensors";
const LANGUAGE_PREFIX: &str = "model.language_model";
const AUDIO_PREFIX: &str = "model.audio_encoder";
/// Taps of the audio downsampler's square kernels.
const TAPS: usize = AUDIO.subsampling.kernel * AUDIO.subsampling.kernel;

/// Where each variant comes from: its repository and revision.
fn source(variant: Variant) -> (&'static str, &'static str) {
    match variant {
        Variant::Vision => ("LiquidAI/d1-2B", "97c57047d5852d77e02dfac211f8768aff77a8b2"),
        Variant::Omni => (
            "LiquidAI/d1-Omni-2B",
            "4cccbfd5ba3405ee0f973134bd03e0d9d60ef1d8",
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The large matrices 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    /// The patch embedding and the audio downsampler's linears.
    const ALWAYS_F32: &'static [&'static str] = &["vision.patch.", "audio.subsampling."];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q7 => "q7",
        }
    }

    fn matrix_dtype(self, _name: &str) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Q7 => DType::Q7b32,
        }
    }
}

/// Each layer's tensors taken as they are: the runtime's name and the checkpoint's, under the
/// layer; `post_operator_norm` and `post_ffn_norm` only in LFM3's sandwiched layers.
const LAYER_PARTS: [(&str, &str); 5] = [
    ("input_norm", "operator_norm.weight"),
    ("post_norm", "ffn_norm.weight"),
    ("gate.weight", "feed_forward.w1.weight"),
    ("up.weight", "feed_forward.w3.weight"),
    ("down.weight", "feed_forward.w2.weight"),
];
const SANDWICH_PARTS: [(&str, &str); 2] = [
    ("mixer_out_norm", "post_operator_norm.weight"),
    ("feed_forward_out_norm", "post_ffn_norm.weight"),
];

/// A conv layer's tensors, as `LAYER_PARTS`, the convolution's under `conv_name`.
fn conv_parts(conv_name: &'static str) -> [(&'static str, String); 3] {
    [
        ("input.weight", "conv.in_proj.weight".into()),
        ("conv", format!("conv.{conv_name}.weight")),
        ("out.weight", "conv.out_proj.weight".into()),
    ]
}

/// An attention layer's norms and output, as `LAYER_PARTS`: LFM3's language model's names, and
/// its vision tower's and LFM2's.
const ATTENTION_PARTS: [(&str, &str); 3] = [
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
    ("o.weight", "self_attn.o_proj.weight"),
];
const LAYERNORM_ATTENTION_PARTS: [(&str, &str); 3] = [
    ("q_norm", "self_attn.q_layernorm.weight"),
    ("k_norm", "self_attn.k_layernorm.weight"),
    ("o.weight", "self_attn.out_proj.weight"),
];

/// Layer `layer`'s tensors from the checkpoint's `from`: `extra` beside `LAYER_PARTS`, and the
/// projections `fused` into its `qkv` when it attends.
fn layer(
    table: &mut Table,
    (layer, from): (&str, &str),
    extra: &[(&str, String)],
    fused: Option<&[&str]>,
) {
    let parts = LAYER_PARTS
        .iter()
        .map(|&(name, source)| (name, source.to_owned()));
    for (name, source) in parts.chain(extra.iter().cloned()) {
        table.one(format!("{layer}.{name}"), format!("{from}.{source}"));
    }
    if let Some(fused) = fused {
        let sources = fused
            .iter()
            .map(|part| format!("{from}.self_attn.{part}.weight"));
        table.push(
            format!("{layer}.qkv.weight"),
            sources.collect(),
            Layout::AsIs,
        );
    }
}

fn owned(parts: &[(&'static str, &str)]) -> Vec<(&'static str, String)> {
    parts
        .iter()
        .map(|&(name, source)| (name, source.to_owned()))
        .collect()
}

/// The language model's tensors as the runtime names them: the head (the input embeddings,
/// tied), the final norm, and each layer's.
fn language_table() -> Table {
    let mut table = Table::default();
    table.one(
        "llm.head.weight".into(),
        format!("{LANGUAGE_PREFIX}.embed_tokens.weight"),
    );
    table.one(
        "llm.norm".into(),
        format!("{LANGUAGE_PREFIX}.embedding_norm.weight"),
    );
    for (i, mixer) in LANGUAGE.mixers.iter().enumerate() {
        let names = (
            &*format!("llm.layers.{i}"),
            &*format!("{LANGUAGE_PREFIX}.layers.{i}"),
        );
        let mut extra = owned(&SANDWICH_PARTS);
        match mixer {
            Mixer::Conv => {
                extra.extend(conv_parts("conv"));
                layer(&mut table, names, &extra, None);
            }
            Mixer::Full | Mixer::Window => {
                extra.extend(owned(&ATTENTION_PARTS));
                let fused = ["q_proj", "k_proj", "v_proj", "gate_proj"];
                layer(&mut table, names, &extra, Some(&fused));
            }
        }
    }
    table
}

/// The vision tower and projector as the runtime names them.
fn vision_table() -> Table {
    const TOWER_PREFIX: &str = "model.vision_tower";
    let mut table = Table::default();
    table.one(
        "vision.patch.weight".into(),
        format!("{TOWER_PREFIX}.patchifier.weight"),
    );
    table.one(
        "vision.patch_norm".into(),
        format!("{TOWER_PREFIX}.embedding_norm.weight"),
    );
    for (i, mixer) in TOWER.mixers.iter().enumerate() {
        let names = (
            &*format!("vision.layers.{i}"),
            &*format!("{TOWER_PREFIX}.layers.{i}"),
        );
        match mixer {
            VisionMixer::Conv => layer(&mut table, names, &conv_parts("spatial_conv"), None),
            VisionMixer::Attention => {
                let extra = owned(&LAYERNORM_ATTENTION_PARTS);
                layer(
                    &mut table,
                    names,
                    &extra,
                    Some(&["q_proj", "k_proj", "v_proj"]),
                );
            }
        }
    }
    projector(&mut table, "projector", "model.multi_modal_projector");
    table
}

/// A LayerNorm, a linear, a GELU and a linear, `p.norm`, `p.up` and `p.down`.
fn projector(table: &mut Table, name: &str, from: &str) {
    let parts = [
        ("norm", "layer_norm"),
        ("up", "linear_1"),
        ("down", "linear_2"),
    ];
    for (part, source) in parts {
        table.linear(
            &format!("{name}.{part}"),
            &[format!("{from}.{source}")],
            &Layout::AsIs,
        );
    }
}

/// d1-Omni-2B's audio encoder's LFM2 stack, final norm and projector.
fn audio_table() -> Table {
    let from = format!("{AUDIO_PREFIX}.lfm");
    let mut table = Table::default();
    for (i, mixer) in AUDIO.stack.mixers.iter().enumerate() {
        let names = (
            &*format!("audio.layers.{i}"),
            &*format!("{from}.layers.{i}"),
        );
        match mixer {
            Mixer::Conv => layer(&mut table, names, &conv_parts("conv"), None),
            Mixer::Full | Mixer::Window => {
                let extra = owned(&LAYERNORM_ATTENTION_PARTS);
                layer(
                    &mut table,
                    names,
                    &extra,
                    Some(&["q_proj", "k_proj", "v_proj"]),
                );
            }
        }
    }
    table.one("audio.norm".into(), format!("{from}.embedding_norm.weight"));
    projector(&mut table, "audio.projector", "model.audio_projector");
    table
}

/// The audio downsampler as `blocks::audio`'s subsampler reads it: the reference convolves
/// (frequency, time) planes, the subsampler (time, frequency) ones, so each square kernel goes
/// tap-major over time then frequency, and the last linear's inputs, the reference's
/// channel-major, go frequency-major.
fn downsampler(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    const FROM: &str = "model.audio_encoder.downsample";
    const TO: &str = "audio.subsampling";
    let kernels = [
        ("conv", "conv_in"),
        ("depthwise1", "conv_mid"),
        ("depthwise2", "conv_end"),
    ];
    for (name, source) in kernels {
        let weight = checkpoint.f32(&format!("{FROM}.{source}.weight"))?;
        let channels = weight.len() / TAPS;
        let side = AUDIO.subsampling.kernel;
        let tap_major: Vec<f32> = (0..TAPS * channels)
            .map(|at| {
                let (tap, channel) = (at / channels, at % channels);
                let (time, frequency) = (tap / side, tap % side);
                weight[channel * TAPS + frequency * side + time]
            })
            .collect();
        tensors.vector(&format!("{TO}.{name}.weight"), &tap_major)?;
        tensors.vector(
            &format!("{TO}.{name}.bias"),
            &checkpoint.f32(&format!("{FROM}.{source}.bias"))?,
        )?;
    }
    for (name, source) in [("pointwise1", "lin_mid"), ("pointwise2", "lin_end")] {
        tensors.copy_linear(
            checkpoint,
            &format!("{TO}.{name}"),
            &format!("{FROM}.{source}"),
        )?;
    }
    let weight = checkpoint.f32(&format!("{FROM}.lin_out.weight"))?;
    let channels = AUDIO.subsampling.channels;
    let frequencies = MELS / AUDIO.subsampling.factor();
    let cols = channels * frequencies;
    let reordered: Vec<f32> = (0..weight.len())
        .map(|at| {
            let (row, input) = (at / cols, at % cols);
            let (frequency, channel) = (input / channels, input % channels);
            weight[row * cols + channel * frequencies + frequency]
        })
        .collect();
    let bias = checkpoint.f32(&format!("{FROM}.lin_out.bias"))?;
    tensors.linear(
        &format!("{TO}.out"),
        reordered.len() / cols,
        &reordered,
        Some(&bias),
    )
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let config_path = checkpoint_dir.join(CONFIG);
    let variant = compile::read_json::<Config>(&config_path)?.model_type;
    let (repo, revision) = source(variant);
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    language_table().write(&checkpoint, &mut tensors)?;
    vision_table().write(&checkpoint, &mut tensors)?;
    if variant == Variant::Omni {
        downsampler(&checkpoint, &mut tensors)?;
        audio_table().write(&checkpoint, &mut tensors)?;
    }
    for blob in [CONFIG, CALIBRATION, TOKENIZER_PART] {
        writer.add_blob_file(blob, &checkpoint_dir.join(blob))?;
    }
    writer.finish(Source {
        repo: repo.into(),
        revision: revision.into(),
        profile: profile.name().into(),
    })
}
