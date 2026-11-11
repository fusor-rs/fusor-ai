//! Wan's diffusion transformer (`diffusion_pytorch_model.safetensors`) under
//! `wan2_1::model::DIT`.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use wan2_1::model::{DIT, DIT_SHAPE};

use crate::Profile;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    tensors.conv(checkpoint, &format!("{DIT}.patch"), "patch_embedding")?;
    let mut table = Table::default();
    let name = |part: &str| format!("{DIT}.{part}");
    table.linear(&name("text.0"), &["text_embedding.0".into()], &Layout::AsIs);
    table.linear(&name("text.1"), &["text_embedding.2".into()], &Layout::AsIs);
    table.linear(&name("time.0"), &["time_embedding.0".into()], &Layout::AsIs);
    table.linear(&name("time.1"), &["time_embedding.2".into()], &Layout::AsIs);
    table.linear(
        &name("modulation"),
        &["time_projection.1".into()],
        &Layout::AsIs,
    );
    table.one(name("head.modulation"), "head.modulation".into());
    table.linear(&name("head.linear"), &["head.head".into()], &Layout::AsIs);
    for block in 0..DIT_SHAPE.depth {
        let name = |part: &str| format!("{DIT}.blocks.{block}.{part}");
        let source = |part: &str| format!("blocks.{block}.{part}");
        let attention = |part: &str| source(&format!("self_attn.{part}"));
        let cross = |part: &str| source(&format!("cross_attn.{part}"));
        table.one(name("modulation"), source("modulation"));
        let qkv = ["q", "k", "v"].map(attention);
        table.linear(&name("qkv"), &qkv, &Layout::AsIs);
        table.one(name("q_norm"), attention("norm_q.weight"));
        table.one(name("k_norm"), attention("norm_k.weight"));
        table.linear(&name("out"), &[attention("o")], &Layout::AsIs);
        table.one(name("norm3.weight"), source("norm3.weight"));
        table.one(name("norm3.bias"), source("norm3.bias"));
        table.linear(&name("cross_query"), &[cross("q")], &Layout::AsIs);
        table.one(name("cross_q_norm"), cross("norm_q.weight"));
        table.one(name("cross_k_norm"), cross("norm_k.weight"));
        let context = ["k", "v"].map(cross);
        table.linear(&name("cross_context"), &context, &Layout::AsIs);
        table.linear(&name("cross_out"), &[cross("o")], &Layout::AsIs);
        table.linear(&name("ff1"), &[source("ffn.0")], &Layout::AsIs);
        table.linear(&name("ff2"), &[source("ffn.2")], &Layout::AsIs);
    }
    table.write(checkpoint, tensors)
}
