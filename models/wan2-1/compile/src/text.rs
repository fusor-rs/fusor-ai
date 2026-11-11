//! umT5-XXL's encoder (`models_t5_umt5-xxl-enc-bf16.pth`) under `wan2_1::model::TEXT`.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use wan2_1::model::{TEXT, TEXT_SHAPE};

use crate::Profile;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut table = Table::default();
    table.one(format!("{TEXT}.embedding"), "token_embedding.weight".into());
    for layer in 0..TEXT_SHAPE.layers {
        let name = |part: &str| format!("{TEXT}.layers.{layer}.{part}");
        let source = |part: &str| format!("blocks.{layer}.{part}.weight");
        let qkv = ["q", "k", "v"].map(|part| source(&format!("attn.{part}")));
        table.one(name("norm1"), source("norm1"));
        table.push(name("qkv.weight"), qkv.to_vec(), Layout::AsIs);
        table.one(name("o.weight"), source("attn.o"));
        table.one(name("norm2"), source("norm2"));
        table.one(name("gate.weight"), source("ffn.gate.0"));
        table.one(name("up.weight"), source("ffn.fc1"));
        table.one(name("down.weight"), source("ffn.fc2"));
    }
    table.one(format!("{TEXT}.norm"), "norm.weight".into());
    table.write(checkpoint, tensors)?;
    for layer in 0..TEXT_SHAPE.layers {
        let source = format!("blocks.{layer}.pos_embedding.embedding.weight");
        let values = checkpoint.f32(&source)?;
        tensors.vector(&format!("{TEXT}.layers.{layer}.bias"), &values)?;
    }
    Ok(())
}
