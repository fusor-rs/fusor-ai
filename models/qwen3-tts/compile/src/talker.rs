//! The talker's and the code predictor's tensors: their stacks, final norms, heads and codec
//! embeddings, renamed.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use qwen3_tts::model::{PREDICTOR, TALKER};

use crate::Profile;

const PREDICTOR_SOURCE: &str = "talker.code_predictor";

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut table = Table::default();
    layers(&mut table, ("talker", "talker.model"), TALKER.stack.layers);
    layers(
        &mut table,
        ("predictor", &format!("{PREDICTOR_SOURCE}.model")),
        PREDICTOR.layers,
    );
    table.one("talker.norm".into(), "talker.model.norm.weight".into());
    table.one(
        "talker.head.weight".into(),
        "talker.codec_head.weight".into(),
    );
    table.one(
        "talker.embed".into(),
        "talker.model.codec_embedding.weight".into(),
    );
    table.one(
        "predictor.norm".into(),
        format!("{PREDICTOR_SOURCE}.model.norm.weight"),
    );
    for group in 0..PREDICTOR_GROUPS {
        table.one(
            format!("predictor.heads.{group}.weight"),
            format!("{PREDICTOR_SOURCE}.lm_head.{group}.weight"),
        );
        table.one(
            format!("predictor.embed.{group}"),
            format!("{PREDICTOR_SOURCE}.model.codec_embedding.{group}.weight"),
        );
    }
    table.write(checkpoint, tensors)
}

/// The codebooks the code predictor draws: every one but the talker's.
const PREDICTOR_GROUPS: usize = qwen3_tts::model::CODE_GROUPS - 1;

/// `layers` Qwen3 layers of the checkpoint's `source` as `target`'s.
fn layers(table: &mut Table, (target, source): (&str, &str), layers: usize) {
    for layer in 0..layers {
        let to = format!("{target}.layers.{layer}");
        let from = format!("{source}.layers.{layer}");
        let attention = |part: &str| format!("{from}.self_attn.{part}");
        let mlp = |part: &str| format!("{from}.mlp.{part}.weight");
        table.one(
            format!("{to}.input_norm"),
            format!("{from}.input_layernorm.weight"),
        );
        table.one(
            format!("{to}.post_norm"),
            format!("{from}.post_attention_layernorm.weight"),
        );
        let projections =
            ["q_proj", "k_proj", "v_proj"].map(|part| attention(&format!("{part}.weight")));
        table.push(
            format!("{to}.qkv.weight"),
            projections.to_vec(),
            Layout::AsIs,
        );
        table.one(format!("{to}.q_norm"), attention("q_norm.weight"));
        table.one(format!("{to}.k_norm"), attention("k_norm.weight"));
        table.one(format!("{to}.o.weight"), attention("o_proj.weight"));
        for (name, part) in [
            ("gate", "gate_proj"),
            ("up", "up_proj"),
            ("down", "down_proj"),
        ] {
            table.one(format!("{to}.{name}.weight"), mlp(part));
        }
    }
}
