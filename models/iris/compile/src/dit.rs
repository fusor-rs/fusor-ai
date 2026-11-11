//! Iris's transformer (`model.safetensors`) under `iris::model::DIT`.

use std::path::Path;

use compile::{Checkpoint, Error, Layout, Table, Tensors};
use iris::model::{
    ADAPTER_SHAPE, DIT, DUAL_BLOCKS, EMPTY_PROMPT, FEED_FORWARD, HIDDEN, IMAGE_CHANNELS,
    PATCH_PIXELS, PIXEL_BLOCKS, PIXEL_HIDDEN, REFINER_SHAPE, SINGLE_BLOCKS, TEXT_LAYERS, TEXT_ROWS,
    Variant,
};

use crate::Profile;

const WEIGHTS: &str = "model.safetensors";
const EMPTY_PROMPT_FILE: &str = "empty_prompt.safetensors";
/// The prefix of every tensor of the depth checkpoint's transformer.
const DEPTH_PREFIX: &str = "pixel.";

pub(crate) fn write(
    dir: &Path,
    tensors: &mut Tensors<Profile>,
    variant: Variant,
) -> Result<(), Error> {
    let checkpoint = Checkpoint::read(&dir.join(WEIGHTS))?;
    let prefix = if variant == Variant::Depth {
        DEPTH_PREFIX
    } else {
        ""
    };
    let source = |part: &str| format!("{prefix}{part}");
    patch_embedding(&checkpoint, tensors, &source)?;
    let positions = checkpoint.f32(&source("y_pos_embedding"))?;
    tensors.add(
        &format!("{DIT}.positions"),
        vec![TEXT_ROWS, HIDDEN],
        &positions,
    )?;
    for part in ["weight", "bias"] {
        let pool = checkpoint.f32(&source(&format!("y_embedder.layer_pool.{part}")))?;
        tensors.vector(&format!("{DIT}.adapter.pool.{part}"), &pool)?;
    }
    let mut table = Table::default();
    time_and_cores(&mut table, &source);
    adapter(&mut table, &source);
    for block in 0..DUAL_BLOCKS + SINGLE_BLOCKS {
        trunk_block(&mut table, block, &source);
    }
    pixel_head(&mut table, &source);
    table.write(&checkpoint, tensors)?;
    pixel_embedding(&checkpoint, tensors, &source)?;
    final_layer(&checkpoint, tensors, &source, variant)?;
    if variant.has_empty_prompt() {
        empty_prompt(dir, tensors)?;
    }
    Ok(())
}

/// The patch embedding, whose columns are the unfolded patch's channel-major values
/// (`c · 256 + dy · 16 + dx`), in the order `Patching::gather` lays a patch out: pixel-major
/// (`(dy · 16 + dx) · 3 + c`). The depth model's extra input channel is dropped.
fn patch_embedding(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    source: &dyn Fn(&str) -> String,
) -> Result<(), Error> {
    let weight = checkpoint.f32(&source("s_embedder.proj.weight"))?;
    let columns = weight.len() / HIDDEN;
    let pixel_major = |row: &[f32]| -> Vec<f32> {
        let channels = (0..PATCH_PIXELS).flat_map(|pixel| {
            (0..IMAGE_CHANNELS).map(move |channel| channel * PATCH_PIXELS + pixel)
        });
        channels.map(|column| row[column]).collect()
    };
    let permuted: Vec<f32> = weight.chunks_exact(columns).flat_map(pixel_major).collect();
    let bias = checkpoint.f32(&source("s_embedder.proj.bias"))?;
    tensors.linear(&format!("{DIT}.patch"), HIDDEN, &permuted, Some(&bias))
}

/// The pixel embedding, its rows' first `IMAGE_CHANNELS` values (the depth model's has a fourth,
/// for zeros).
fn pixel_embedding(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    source: &dyn Fn(&str) -> String,
) -> Result<(), Error> {
    let weight = checkpoint.f32(&source("pixel_embedder.proj.weight"))?;
    let columns = weight.len() / PIXEL_HIDDEN;
    let kept: Vec<f32> = weight
        .chunks_exact(columns)
        .flat_map(|row| &row[..IMAGE_CHANNELS])
        .copied()
        .collect();
    let bias = checkpoint.f32(&source("pixel_embedder.proj.bias"))?;
    let name = format!("{DIT}.pixel.embed");
    tensors.linear(&name, PIXEL_HIDDEN, &kept, Some(&bias))
}

/// The final layer, whose linear the depth model's 1×1 reducer follows: `W'[k] = Σ_c r[c]·W[c][k]`
/// and `b' = Σ_c r[c]·b[c] + r_b`, summed in f64 in channel order.
fn final_layer(
    checkpoint: &Checkpoint,
    tensors: &mut Tensors<Profile>,
    source: &dyn Fn(&str) -> String,
    variant: Variant,
) -> Result<(), Error> {
    let name = format!("{DIT}.pixel.final");
    let weight = checkpoint.f32(&source("final_layer.linear.weight"))?;
    let bias = checkpoint.f32(&source("final_layer.linear.bias"))?;
    if variant != Variant::Depth {
        return tensors.linear(&name, bias.len(), &weight, Some(&bias));
    }
    let reducer = checkpoint.f32("depth_reducer.weight")?;
    let reducer_bias = checkpoint.f32("depth_reducer.bias")?[0];
    let columns = weight.len() / reducer.len();
    let folded: Vec<f32> = (0..columns)
        .map(|column| {
            let sum: f64 = reducer
                .iter()
                .enumerate()
                .map(|(channel, &factor)| {
                    f64::from(factor) * f64::from(weight[channel * columns + column])
                })
                .sum();
            sum as f32
        })
        .collect();
    let offset: f64 = reducer
        .iter()
        .zip(&bias)
        .map(|(&factor, &channel)| f64::from(factor) * f64::from(channel))
        .sum();
    let folded_bias = [(offset + f64::from(reducer_bias)) as f32];
    tensors.linear(&name, 1, &folded, Some(&folded_bias))
}

/// The empty prompt's states (`embeddings`: `[1, 300, 12, 2560]`, `mask`: `[1, 300]`), their
/// real rows only: the mask is those rows' trues, and the states after them are zeros.
fn empty_prompt(dir: &Path, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let checkpoint = Checkpoint::read(&dir.join(EMPTY_PROMPT_FILE))?;
    let mask = checkpoint.bool("mask")?;
    let embeddings = checkpoint.f32("embeddings")?;
    let rows = mask.iter().filter(|&&real| real).count();
    let row_values = TEXT_LAYERS.len() * HIDDEN;
    let (real, padding) = embeddings.split_at(rows * row_values);
    let rows_first = mask[..rows].iter().all(|&real| real);
    if !rows_first || mask.len() != TEXT_ROWS || padding.iter().any(|&value| value != 0.0) {
        return Err(Error::Format(format!(
            "{EMPTY_PROMPT_FILE}: the mask is not {rows} real rows then padding of zeros"
        )));
    }
    tensors.vector(EMPTY_PROMPT, real)
}

/// The timestep MLP and the two modulation cores.
fn time_and_cores(table: &mut Table, source: &dyn Fn(&str) -> String) {
    let links = [
        ("time.0", "t_embedder.mlp.0"),
        ("time.1", "t_embedder.mlp.2"),
        ("modulation.image", "modulation_cores.adaln_img"),
        ("modulation.text", "modulation_cores.adaln_txt"),
    ];
    for (name, from) in links {
        table.linear(&format!("{DIT}.{name}"), &[source(from)], &Layout::AsIs);
    }
}

/// The text adapter: two layer-wise blocks, the pool is written beside, the refiner's linear and
/// two layers, and its final norm.
fn adapter(table: &mut Table, source: &dyn Fn(&str) -> String) {
    let prefix = format!("{DIT}.adapter");
    for block in 0..ADAPTER_SHAPE.depth {
        let name = |part: &str| format!("{prefix}.layers.blocks.{block}.{part}");
        let from = |part: &str| source(&format!("y_embedder.layer_blocks.{block}.{part}"));
        table.one(name("norm1"), from("norm1.weight"));
        table.one(name("norm2"), from("norm2.weight"));
        table.one(name("qkv.weight"), from("attn.qkv.weight"));
        table.linear(&name("out"), &[from("attn.proj")], &Layout::AsIs);
        table.linear(&name("ff1"), &[from("mlp.0")], &Layout::AsIs);
        table.linear(&name("ff2"), &[from("mlp.2")], &Layout::AsIs);
    }
    let refiner = |part: &str| source(&format!("y_embedder.refiner.{part}"));
    table.linear(
        &format!("{prefix}.input"),
        &[refiner("proj")],
        &Layout::AsIs,
    );
    table.one(format!("{prefix}.norm"), refiner("norm.weight"));
    for block in 0..REFINER_SHAPE.layers {
        let name = |part: &str| format!("{prefix}.refiner.layers.{block}.{part}");
        let from = |part: &str| refiner(&format!("blocks.{block}.{part}"));
        table.one(name("input_norm"), from("norm1.weight"));
        table.one(name("post_norm"), from("norm2.weight"));
        table.one(name("qkv.weight"), from("attn.qkv.weight"));
        table.linear(&name("o"), &[from("attn.proj")], &Layout::AsIs);
        table.one(name("q_norm"), from("attn.q_norm.weight"));
        table.one(name("k_norm"), from("attn.k_norm.weight"));
        let gated = ["gate", "up", "down"].map(|part| name(&format!("{part}.weight")));
        let sources = ["w1", "w3", "w2"].map(|part| from(&format!("mlp.{part}.weight")));
        add_feed_forward(table, gated, sources);
    }
}

/// A SwiGLU's `w1`, `w3` and `w2` (`sources`) as the bundle tensors `names`, padded to
/// `FEED_FORWARD` hidden units: zero rows for the gate and the up projection, zero columns for
/// the down projection.
fn add_feed_forward(table: &mut Table, names: [String; 3], sources: [String; 3]) {
    let layouts = [
        Layout::PadRows(FEED_FORWARD),
        Layout::PadRows(FEED_FORWARD),
        Layout::PadColumns(FEED_FORWARD),
    ];
    for ((name, source), layout) in names.into_iter().zip(sources).zip(layouts) {
        table.push(name, vec![source], layout);
    }
}

/// A trunk block stream's names in the checkpoint.
struct Stream {
    norm1: String,
    norm2: String,
    qkv: [String; 3],
    q_norm: String,
    k_norm: String,
    gate: String,
    out: String,
    post_attention: String,
    post_ff: String,
    /// The SwiGLU's `w1`, `w3` and `w2`.
    feed_forward: [String; 3],
    bias: String,
}

impl Stream {
    /// The stream of a dual block for `side`: `x` is the image, `y` the text.
    fn dual(block: usize, side: char, modulation: &str, source: &dyn Fn(&str) -> String) -> Self {
        let from = |part: String| source(&format!("blocks.{block}.{part}"));
        Self {
            norm1: from(format!("norm_{side}1.weight")),
            norm2: from(format!("norm_{side}2.weight")),
            qkv: ["q", "k", "v"].map(|part| from(format!("attn.{part}_proj_{side}.weight"))),
            q_norm: from(format!("attn.q_norm_{side}.weight")),
            k_norm: from(format!("attn.k_norm_{side}.weight")),
            gate: from(format!("attn_gate_{side}.weight")),
            out: from(format!("attn.proj_{side}")),
            post_attention: from(format!("attn_post_norm_{side}.weight")),
            post_ff: from(format!("mlp_post_norm_{side}.weight")),
            feed_forward: ["w1", "w3", "w2"].map(|part| from(format!("mlp_{side}.{part}.weight"))),
            bias: from(format!("adaln_{modulation}.bias")),
        }
    }

    /// The one stream of a single block.
    fn single(block: usize, source: &dyn Fn(&str) -> String) -> Self {
        let from = |part: String| source(&format!("blocks.{block}.{part}"));
        Self {
            norm1: from("norm1.weight".into()),
            norm2: from("norm2.weight".into()),
            qkv: ["q", "k", "v"].map(|part| from(format!("{part}_proj.weight"))),
            q_norm: from("q_norm.weight".into()),
            k_norm: from("k_norm.weight".into()),
            gate: from("attn_gate.weight".into()),
            out: from("attn_proj".into()),
            post_attention: from("attn_post_norm.weight".into()),
            post_ff: from("mlp_post_norm.weight".into()),
            feed_forward: ["w1", "w3", "w2"].map(|part| from(format!("mlp.{part}.weight"))),
            bias: from("adaln.bias".into()),
        }
    }

    /// This stream as the bundle's `prefix.*`.
    fn add(self, table: &mut Table, prefix: &str) {
        let name = |part: &str| format!("{prefix}.{part}");
        table.one(name("norm1"), self.norm1);
        table.one(name("norm2"), self.norm2);
        table.push(name("qkv.weight"), self.qkv.to_vec(), Layout::AsIs);
        table.one(name("q_norm"), self.q_norm);
        table.one(name("k_norm"), self.k_norm);
        table.one(name("gate.weight"), self.gate);
        table.linear(&name("out"), &[self.out], &Layout::AsIs);
        table.one(name("post_attention"), self.post_attention);
        table.one(name("post_ff"), self.post_ff);
        let names = ["ff_gate", "ff_up", "ff_down"].map(|part| name(&format!("{part}.weight")));
        add_feed_forward(table, names, self.feed_forward);
        table.one(name("bias"), self.bias);
    }
}

/// Block `block` of the trunk: the dual blocks' text and image streams, the single blocks' one.
fn trunk_block(table: &mut Table, block: usize, source: &dyn Fn(&str) -> String) {
    let prefix = format!("{DIT}.blocks.{block}");
    if block < DUAL_BLOCKS {
        let text = Stream::dual(block, 'y', "txt", source);
        text.add(table, &format!("{prefix}.text"));
        let image = Stream::dual(block, 'x', "img", source);
        image.add(table, &format!("{prefix}.image"));
    } else {
        Stream::single(block, source).add(table, &format!("{prefix}.shared"));
    }
}

/// The pixel head's blocks and final norm.
fn pixel_head(table: &mut Table, source: &dyn Fn(&str) -> String) {
    for block in 0..PIXEL_BLOCKS {
        let name = |part: &str| format!("{DIT}.pixel.blocks.{block}.{part}");
        let from = |part: &str| source(&format!("pixel_blocks.{block}.{part}"));
        table.one(name("norm1"), from("norm1.weight"));
        table.one(name("norm2"), from("norm2.weight"));
        table.one(name("qkv.weight"), from("attn.qkv.weight"));
        table.one(name("q_norm"), from("attn.q_norm.weight"));
        table.one(name("k_norm"), from("attn.k_norm.weight"));
        let linears = [
            ("compress", "compress"),
            ("expand", "expand"),
            ("modulation", "adaln"),
            ("out", "attn.proj"),
            ("ff1", "mlp.fc1"),
            ("ff2", "mlp.fc2"),
        ];
        for (bundle_part, checkpoint_part) in linears {
            table.linear(&name(bundle_part), &[from(checkpoint_part)], &Layout::AsIs);
        }
    }
    let norm = source("final_layer.norm.weight");
    table.one(format!("{DIT}.pixel.final_norm"), norm);
}
