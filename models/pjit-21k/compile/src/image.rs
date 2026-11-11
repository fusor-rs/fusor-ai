//! The image bundle's tensors: P-JiT 21K's DiT (`model.safetensors`) under
//! `pjit_21k::model::IMAGE`.

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use pjit_21k::model::{DIT, IMAGE};

use crate::Profile;

/// The checkpoint's groups of blocks, which keep their names in the bundle, and how many each has.
const BLOCK_GROUPS: [(&str, usize); 3] = [
    ("image_refiner", DIT.refiners),
    ("text_refiner", DIT.refiners),
    ("blocks", DIT.depth),
];
/// A block's vectors (the same names in the bundle) and their checkpoint names.
const BLOCK_VECTORS: [(&str, &str); 6] = [
    ("norm1", "norm1.weight"),
    ("norm2", "norm2.weight"),
    ("post_norm1", "post_norm1.weight"),
    ("post_norm2", "post_norm2.weight"),
    ("q_norm", "attn.norm_q.weight"),
    ("k_norm", "attn.norm_k.weight"),
];

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let name = |part: &str| format!("{IMAGE}.{part}");
    tensors.conv(checkpoint, &name("patch.0"), "patch_embed.proj1")?;
    tensors.conv(checkpoint, &name("patch.1"), "patch_embed.proj2")?;
    table().write(checkpoint, tensors)?;
    tensors.copy_norm(checkpoint, &name("head.norm"), "head.norm")
}

/// Every tensor but the patch embedding and the head's LayerNorm, renamed.
fn table() -> Table {
    let mut table = Table::default();
    let name = |part: &str| format!("{IMAGE}.{part}");
    table.linear(&name("time.0"), &["time_embed.0".into()], &Layout::AsIs);
    table.linear(&name("time.1"), &["time_embed.2".into()], &Layout::AsIs);
    table.linear(
        &name("roots.image"),
        &["image_adaln.1".into()],
        &Layout::AsIs,
    );
    table.linear(&name("roots.text"), &["text_adaln.1".into()], &Layout::AsIs);
    table.one(name("text.0.weight"), "text_proj.0.weight".into());
    table.one(name("text.1.weight"), "text_proj.2.weight".into());
    for (group, count) in BLOCK_GROUPS {
        for index in 0..count {
            block(&mut table, &format!("{group}.{index}"));
        }
    }
    table.one(name("head.modulation"), "head.modulation".into());
    table.linear(&name("head.linear"), &["head.proj".into()], &Layout::AsIs);
    table
}

/// The block `block`'s tensors, under the same name in the bundle.
fn block(table: &mut Table, block: &str) {
    let name = |part: &str| format!("{IMAGE}.{block}.{part}");
    let source = |part: &str| format!("{block}.{part}");
    for (part, from) in BLOCK_VECTORS {
        table.one(name(part), source(from));
    }
    let qkv_gate = ["q", "k", "v", "gate_proj"].map(|part| source(&format!("attn.{part}")));
    table.linear(&name("qkv_gate"), &qkv_gate, &Layout::AsIs);
    table.linear(&name("out"), &[source("attn.o")], &Layout::AsIs);
    table.one(name("ff1.weight"), source("ffn.w13.weight"));
    table.one(name("ff2.weight"), source("ffn.w2.weight"));
    table.linear(&name("modulation"), &[source("adaln_up")], &Layout::AsIs);
}
