//! `kronos compile`: the checkpoints (Kronos-base's and Kronos-Tokenizer-base's
//! `model.safetensors`, f32, in `model/` and `tokenizer/`) → a browser bundle: the tensors the
//! model reads, renamed to its vocabulary (`kronos::model`), each attention layer's query, key
//! and value projections fused into one, and quantized per profile; fusor-ai's `BundleWriter`
//! packs and hashes the parts and writes `manifest.json`.

use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Checkpoint, Error, Layout, Profile as _, Table, Tensors};
use kernels::DType;
use kronos::model::{Field, STACK};
use kronos::tokenizer;

const REPO: &str = "NeoQuasar/Kronos-base";
const REVISION: &str = "2b554741eca47781b64468546e77fef3e85130e6";
const WEIGHTS: &str = "model.safetensors";
/// The checkpoints' directories under the checkpoint directory.
const PREDICTOR_DIR: &str = "model";
const TOKENIZER_DIR: &str = "tokenizer";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32 (parity).
    F32,
    /// The predictor's stack, dependency layer and temporal embeddings 7-bit.
    Q7,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32, Self::Q7];
    /// The tokenizer, whose codes' signs are its tokens, and the predictor's embeddings and
    /// heads: 7-bit, they move the logits enough to change a greedy forecast's tokens.
    const ALWAYS_F32: &'static [&'static str] = &["tokenizer.", "embed.", "head."];

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

/// A stack of `layers` reference `TransformerBlock`s under `source`: `blocks::transformer`'s
/// Qwen2 names under `prefix`.
fn stack(table: &mut Table, prefix: &str, source: &str, layers: usize) {
    for i in 0..layers {
        let (layer, from) = (format!("{prefix}.layers.{i}"), format!("{source}.{i}"));
        let qkv = ["q", "k", "v"].map(|head| format!("{from}.self_attn.{head}_proj"));
        table.linear(&format!("{layer}.qkv"), &qkv, &Layout::AsIs);
        table.linear(
            &format!("{layer}.o"),
            &[format!("{from}.self_attn.out_proj")],
            &Layout::AsIs,
        );
        table.one(
            format!("{layer}.input_norm"),
            format!("{from}.norm1.weight"),
        );
        table.one(format!("{layer}.post_norm"), format!("{from}.norm2.weight"));
        for (projection, part) in [("gate", "w1"), ("up", "w3"), ("down", "w2")] {
            table.one(
                format!("{layer}.{projection}.weight"),
                format!("{from}.ffn.{part}.weight"),
            );
        }
    }
}

/// Linear layers taken as they are, `(name, source)`.
fn linears(table: &mut Table, pairs: &[(&str, &str)]) {
    for (name, source) in pairs {
        table.linear(name, &[(*source).to_string()], &Layout::AsIs);
    }
}

/// The predictor's tensors, as `kronos::model` names them.
fn predictor_table() -> Table {
    let mut table = Table::default();
    stack(&mut table, "llm", "transformer", STACK.layers);
    table.one("llm.norm".into(), "norm.weight".into());
    table.one("embed.s1".into(), "embedding.emb_s1.weight".into());
    table.one("embed.s2".into(), "embedding.emb_s2.weight".into());
    for field in Field::ALL {
        let name = field.name();
        table.one(
            format!("time.{name}"),
            format!("time_emb.{name}_embed.weight"),
        );
    }
    linears(
        &mut table,
        &[
            ("embed.fusion", "embedding.fusion_proj"),
            ("head.s1", "head.proj_s1"),
            ("head.s2", "head.proj_s2"),
            ("dependency.query", "dep_layer.cross_attn.q_proj"),
            ("dependency.key", "dep_layer.cross_attn.k_proj"),
            ("dependency.value", "dep_layer.cross_attn.v_proj"),
            ("dependency.out", "dep_layer.cross_attn.out_proj"),
        ],
    );
    table.one("dependency.norm".into(), "dep_layer.norm.weight".into());
    table
}

/// The tokenizer's tensors, as `kronos::tokenizer` names them under `tokenizer`.
fn tokenizer_table() -> Table {
    let mut table = Table::default();
    let layers = tokenizer::SHAPE.layers;
    stack(&mut table, "tokenizer.encoder", "encoder", layers);
    stack(&mut table, "tokenizer.decoder", "decoder", layers);
    linears(
        &mut table,
        &[
            ("tokenizer.embed", "embed"),
            ("tokenizer.quantize", "quant_embed"),
            ("tokenizer.dequantize", "post_quant_embed"),
            ("tokenizer.head", "head"),
        ],
    );
    table
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let predictor = Checkpoint::read(&checkpoint_dir.join(PREDICTOR_DIR).join(WEIGHTS))?;
    let tokenizer = Checkpoint::read(&checkpoint_dir.join(TOKENIZER_DIR).join(WEIGHTS))?;
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    predictor_table().write(&predictor, &mut tensors)?;
    tokenizer_table().write(&tokenizer, &mut tensors)?;
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
