//! `lfm3 compile`: an LFM3-1.7B, LFM3-VL-2B, LFM3-Audio-2B or LFM3-Omni-2B checkpoint (one
//! safetensors file, bf16, its `config.json`, which names the variant, its tokenizer, and for the
//! audio variants the codec's `audio_tokenizer/model.safetensors`) → a browser bundle.
//!
//! Renames to the runtime's vocabulary (`lfm3::model`, `lfm3::audio`), fuses each attention
//! layer's query, key, value and, where it has one, gate projections (one gate row per query
//! head, last) into one, keeps the patch embedding's two frames (an image fills both), lays the
//! audio subsampler's 3 × 3 kernels out time-major and its last linear's inputs
//! frequency-major, and quantizes per profile; fusor-ai's `BundleWriter` packs and hashes the
//! parts and writes `manifest.json`. LFM3-Audio-2B's checkpoint names its tensors as its release
//! did, LFM3-Omni-2B's as transformers does (`AudioNames`).

use std::path::Path;

use blocks::interleaved::Mixer;
use blocks::vit::interleaved::Mixer as VisionMixer;
use bundle::{Manifest, Source, TOKENIZER_PART};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use lfm3::audio::{CODEBOOKS, CODEC, DEPTH, ENCODER, SUBSAMPLING};
use lfm3::model::{CONFIG, Config, LANGUAGE, TOWER, Variant};

const WEIGHTS: &str = "model.safetensors";
/// The codec's checkpoint, beside an audio variant's.
const CODEC_WEIGHTS: &str = "audio_tokenizer/model.safetensors";

/// Where a variant comes from: its repository and revision, the prefix of its language model's
/// tensors, and its audio's names.
struct Origin {
    repo: &'static str,
    revision: &'static str,
    language: &'static str,
    audio: Option<AudioNames>,
}

/// How an audio variant's checkpoint names its audio tensors.
struct AudioNames {
    downsample: &'static str,
    encoder: &'static str,
    /// The projector's LayerNorm, then its two linears.
    projector: [&'static str; 3],
    depth_linear: &'static str,
    mask: &'static str,
    depthformer: &'static str,
    depth_embeddings: &'static str,
    codebooks: &'static str,
}

fn origin(variant: Variant) -> Origin {
    match variant {
        Variant::Text => Origin {
            repo: "LiquidAI/LFM3-1.7B",
            revision: "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
            language: "model",
            audio: None,
        },
        Variant::Vision => Origin {
            repo: "LiquidAI/LFM3-VL-2B",
            revision: "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
            language: "model.language_model",
            audio: None,
        },
        Variant::Audio => Origin {
            repo: "LiquidAI/LFM3-Audio-2B",
            revision: "9d49701c372d6011d900f6abb40c6b296fb706dd",
            language: "lfm.model",
            audio: Some(AudioNames {
                downsample: "encoder.downsample",
                encoder: "encoder.lfm.lfm",
                projector: ["mlp.model.0", "mlp.model.1", "mlp.model.3"],
                depth_linear: "depth_linear",
                mask: "mask_embedding",
                depthformer: "depthformer.model",
                depth_embeddings: "depth_embeddings",
                codebooks: "codebook_embedding.weight",
            }),
        },
        Variant::Omni => Origin {
            repo: "LiquidAI/LFM3-Omni-2B",
            revision: "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
            language: "model.language_model",
            audio: Some(AudioNames {
                downsample: "model.audio_encoder.downsample",
                encoder: "model.audio_encoder.lfm",
                projector: [
                    "model.audio_projector.layer_norm",
                    "model.audio_projector.linear_1",
                    "model.audio_projector.linear_2",
                ],
                depth_linear: "model.depth_decoder.depth_linear",
                mask: "model.depth_decoder.mask_embedding",
                depthformer: "model.depth_decoder.model",
                depth_embeddings: "model.depth_decoder.depth_embeddings",
                codebooks: "model.codebook_embedding.weight",
            }),
        },
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

/// The runtime's names of a layer's tensors and the checkpoint's, under the layer.
type Parts = &'static [(&'static str, &'static str)];

/// How a stack's checkpoint names its layers' tensors.
struct Naming {
    /// Every layer's norms and feed-forward.
    layer: Parts,
    /// The norms of the sublayers' outputs, in a stack that has them.
    sandwich: Parts,
    conv: Parts,
    attention: Parts,
    /// The projections fused into an attention layer's `qkv`, in its order.
    fused: &'static [&'static str],
}

const LAYER_PARTS: Parts = &[
    ("input_norm", "operator_norm.weight"),
    ("post_norm", "ffn_norm.weight"),
    ("gate.weight", "feed_forward.w1.weight"),
    ("up.weight", "feed_forward.w3.weight"),
    ("down.weight", "feed_forward.w2.weight"),
];
const SANDWICH_PARTS: Parts = &[
    ("mixer_out_norm", "post_operator_norm.weight"),
    ("feed_forward_out_norm", "post_ffn_norm.weight"),
];
const CONV_PARTS: Parts = &[
    ("input.weight", "conv.in_proj.weight"),
    ("conv", "conv.conv.weight"),
    ("out.weight", "conv.out_proj.weight"),
];
const GATED_ATTENTION_PARTS: Parts = &[
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
    ("o.weight", "self_attn.o_proj.weight"),
];
const ATTENTION_PARTS: Parts = &[
    ("q_norm", "self_attn.q_layernorm.weight"),
    ("k_norm", "self_attn.k_layernorm.weight"),
    ("o.weight", "self_attn.out_proj.weight"),
];

/// LFM3's layers: the language model's and the codec's.
const LFM3: Naming = Naming {
    layer: LAYER_PARTS,
    sandwich: SANDWICH_PARTS,
    conv: CONV_PARTS,
    attention: GATED_ATTENTION_PARTS,
    fused: &["q_proj", "k_proj", "v_proj", "gate_proj"],
};
/// The depth decoder's: LFM3's, without sandwich norms.
const LFM3_PRE: Naming = Naming {
    sandwich: &[],
    ..LFM3
};
/// LFM2's: the audio encoder's.
const LFM2: Naming = Naming {
    layer: LAYER_PARTS,
    sandwich: &[],
    conv: CONV_PARTS,
    attention: ATTENTION_PARTS,
    fused: &["q_proj", "k_proj", "v_proj"],
};
/// The vision tower's: LFM2's, its convolutions spatial.
const VISION: Naming = Naming {
    conv: &[
        ("input.weight", "conv.in_proj.weight"),
        ("conv", "conv.spatial_conv.weight"),
        ("out.weight", "conv.out_proj.weight"),
    ],
    ..LFM2
};

/// How a layer mixes its rows, as far as its names go.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Conv,
    Attention,
}

impl Kind {
    fn of(mixer: &Mixer) -> Self {
        match mixer {
            Mixer::Conv => Self::Conv,
            Mixer::Full | Mixer::Window => Self::Attention,
        }
    }
}

/// The layers `to.layers.{i}` of a stack from the checkpoint's `from.layers.{i}`.
fn stack(table: &mut Table, naming: &Naming, (to, from): (&str, &str), kinds: &[Kind]) {
    for (i, &kind) in kinds.iter().enumerate() {
        let (layer, source) = (format!("{to}.layers.{i}"), format!("{from}.layers.{i}"));
        let mixer = match kind {
            Kind::Conv => naming.conv,
            Kind::Attention => naming.attention,
        };
        let parts = naming.layer.iter().chain(naming.sandwich).chain(mixer);
        for (name, part) in parts {
            table.one(format!("{layer}.{name}"), format!("{source}.{part}"));
        }
        if kind == Kind::Attention {
            let fused = naming.fused.iter();
            let fused = fused.map(|part| format!("{source}.self_attn.{part}.weight"));
            table.push(format!("{layer}.qkv.weight"), fused.collect(), Layout::AsIs);
        }
    }
}

fn kinds(mixers: &[Mixer]) -> Vec<Kind> {
    mixers.iter().map(Kind::of).collect()
}

/// The language model's tensors as the runtime names them, from those under `prefix`: the head
/// (the input embeddings, tied), the final norm, and each layer's.
fn language_table(prefix: &str) -> Table {
    let mut table = Table::default();
    table.one(
        "llm.head.weight".into(),
        format!("{prefix}.embed_tokens.weight"),
    );
    table.one("llm.norm".into(), format!("{prefix}.embedding_norm.weight"));
    stack(&mut table, &LFM3, ("llm", prefix), &kinds(LANGUAGE.mixers));
    table
}

/// The vision tower and projector of LFM3-VL-2B and LFM3-Omni-2B as the runtime names them.
fn vision_table() -> Table {
    const TOWER_PREFIX: &str = "model.vision_tower";
    const PROJECTOR_PREFIX: &str = "model.multi_modal_projector";
    let mut table = Table::default();
    table.one(
        "vision.patch.weight".into(),
        format!("{TOWER_PREFIX}.patchifier.weight"),
    );
    table.one(
        "vision.patch_norm".into(),
        format!("{TOWER_PREFIX}.embedding_norm.weight"),
    );
    let tower_kinds: Vec<Kind> = TOWER
        .mixers
        .iter()
        .map(|mixer| match mixer {
            VisionMixer::Conv => Kind::Conv,
            VisionMixer::Attention => Kind::Attention,
        })
        .collect();
    stack(&mut table, &VISION, ("vision", TOWER_PREFIX), &tower_kinds);
    let projector = [
        ("norm", "layer_norm"),
        ("up", "linear_1"),
        ("down", "linear_2"),
    ];
    for (name, source) in projector {
        let source = format!("{PROJECTOR_PREFIX}.{source}");
        table.linear(&format!("projector.{name}"), &[source], &Layout::AsIs);
    }
    table
}

/// The audio encoder's stack and projector, and the depth decoder, as the runtime names them.
fn audio_table(names: &AudioNames) -> Table {
    let mut table = Table::default();
    stack(
        &mut table,
        &LFM2,
        ("audio", names.encoder),
        &kinds(ENCODER.stack.mixers),
    );
    table.one(
        "audio.norm".into(),
        format!("{}.embedding_norm.weight", names.encoder),
    );
    let projector = ["norm", "up", "down"].iter().zip(names.projector);
    for (name, source) in projector {
        let to = format!("audio.projector.{name}");
        table.linear(&to, &[source.to_owned()], &Layout::AsIs);
    }
    table.linear(
        "depth.slots",
        &[names.depth_linear.to_owned()],
        &Layout::AsIs,
    );
    table.one("depth.mask".into(), names.mask.into());
    let depth_kinds = kinds(DEPTH.stack.mixers);
    stack(
        &mut table,
        &LFM3_PRE,
        ("depth", names.depthformer),
        &depth_kinds,
    );
    table.one(
        "depth.norm".into(),
        format!("{}.embedding_norm.weight", names.depthformer),
    );
    for codebook in 0..CODEBOOKS {
        let (to, from) = (
            format!("depth.codebooks.{codebook}"),
            format!("{}.{codebook}", names.depth_embeddings),
        );
        table.one(
            format!("{to}.embedding.weight"),
            format!("{from}.embedding.weight"),
        );
        table.one(
            format!("{to}.norm"),
            format!("{from}.embedding_norm.weight"),
        );
    }
    table.one("codebooks".into(), names.codebooks.into());
    table
}

/// The codec's decoder as the runtime names it.
fn codec_table() -> Table {
    const DECODER: &str = "decoder.lfm";
    let mut table = Table::default();
    let linears = [
        ("project_out", "quantizer.project_out"),
        ("from_quantizer", "from_quantizer"),
        ("spectrum", "decoder.lin"),
    ];
    for (name, source) in linears {
        table.linear(&format!("codec.{name}"), &[source.into()], &Layout::AsIs);
    }
    stack(
        &mut table,
        &LFM3,
        ("codec", DECODER),
        &kinds(CODEC.stack.mixers),
    );
    table.one(
        "codec.norm".into(),
        format!("{DECODER}.embedding_norm.weight"),
    );
    table.one("codec.window".into(), "decoder.istft.window".into());
    table
}

/// The audio subsampler: each 3 × 3 kernel's taps time-major (the checkpoint's are mel-major),
/// the 1 × 1 convolutions as linears, and the last linear's inputs frequency-major (the
/// checkpoint's are channel-major).
fn subsampler<P: compile::Profile>(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<P>,
    prefix: &str,
) -> Result<(), Error> {
    const TO: &str = "audio.subsampling";
    let side = SUBSAMPLING.kernel;
    let kernels = [
        ("conv", "conv_in"),
        ("depthwise1", "conv_mid"),
        ("depthwise2", "conv_end"),
    ];
    for (name, source) in kernels {
        let weight = checkpoint.f32(&format!("{prefix}.{source}.weight"))?;
        let taps = side * side;
        let time_major = (0..taps * SUBSAMPLING.channels).map(|at| {
            let (tap, channel) = (at / SUBSAMPLING.channels, at % SUBSAMPLING.channels);
            let (time, mel) = (tap / side, tap % side);
            weight[channel * taps + mel * side + time]
        });
        tensors.vector(
            &format!("{TO}.{name}.weight"),
            &time_major.collect::<Vec<_>>(),
        )?;
        let bias = checkpoint.f32(&format!("{prefix}.{source}.bias"))?;
        tensors.vector(&format!("{TO}.{name}.bias"), &bias)?;
    }
    for (name, source) in [("pointwise1", "lin_mid"), ("pointwise2", "lin_end")] {
        tensors.copy_linear(
            checkpoint,
            &format!("{TO}.{name}"),
            &format!("{prefix}.{source}"),
        )?;
    }
    let weight = checkpoint.f32(&format!("{prefix}.lin_out.weight"))?;
    let channels = SUBSAMPLING.channels;
    let inputs = weight.len() / SUBSAMPLING.hidden;
    let frequencies = inputs / channels;
    let frequency_major = (0..weight.len()).map(|at| {
        let (output, input) = (at / inputs, at % inputs);
        let (frequency, channel) = (input / channels, input % channels);
        weight[output * inputs + channel * frequencies + frequency]
    });
    let weight: Vec<f32> = frequency_major.collect();
    let bias = checkpoint.f32(&format!("{prefix}.lin_out.bias"))?;
    tensors.linear(
        &format!("{TO}.out"),
        SUBSAMPLING.hidden,
        &weight,
        Some(&bias),
    )
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let config_path = checkpoint_dir.join(CONFIG);
    let variant = compile::read_json::<Config>(&config_path)?.model_type;
    let origin = origin(variant);
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    let checkpoint = Checkpoint::read(&checkpoint_dir.join(WEIGHTS))?;
    language_table(origin.language).write(&checkpoint, &mut tensors)?;
    if variant.sees() {
        vision_table().write(&checkpoint, &mut tensors)?;
    }
    if let Some(names) = &origin.audio {
        audio_table(names).write(&checkpoint, &mut tensors)?;
        subsampler(&checkpoint, &mut tensors, names.downsample)?;
        let codec = Checkpoint::read(&checkpoint_dir.join(CODEC_WEIGHTS))?;
        codec_table().write(&codec, &mut tensors)?;
    }
    writer.add_blob_file(CONFIG, &config_path)?;
    writer.add_blob_file(TOKENIZER_PART, &checkpoint_dir.join(TOKENIZER_PART))?;
    writer.finish(Source {
        repo: origin.repo.into(),
        revision: origin.revision.into(),
        profile: profile.name().into(),
    })
}
