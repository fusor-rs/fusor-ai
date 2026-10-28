//! The text embedding through the text projection, `silu(fc1(x))` into `fc2`, for every token:
//! the projection depends on the token alone, so the runtime reads one table. It also halves the
//! largest table (2,048 channels to 1,024).

use compile::{Checkpoint, Error, Tensors};
use kernels::conv::{Patches, project_rows};
use kernels::ops::silu;
use kernels::{DType, Linear, Table, Tensor, Vector};
use qwen3_tts::model::{HIDDEN, TEXT_VOCABULARY};
use team::{PerThread, SharedBuf, Team};

use crate::Profile;

const EMBEDDING: &str = "talker.model.text_embedding.weight";
const PROJECTION: &str = "talker.text_projection";
/// Tokens projected at a time.
const CHUNK_ROWS: usize = 4096;

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let embedding = checkpoint.f32(EMBEDDING)?;
    let width = embedding.len() / TEXT_VOCABULARY;
    let first = linear(checkpoint, "linear_fc1")?;
    let second = linear(checkpoint, "linear_fc2")?;
    let team = Team::global();
    let patches = PerThread::new(team.threads(), Patches::default);
    let hidden = SharedBuf::new(CHUNK_ROWS * width);
    let out = SharedBuf::new(CHUNK_ROWS * HIDDEN);
    let mut projected = Vec::with_capacity(TEXT_VOCABULARY * HIDDEN);
    for chunk in embedding.chunks(CHUNK_ROWS * width) {
        let rows = chunk.len() / width;
        team.run(|ctx| {
            let patches = patches.get(ctx);
            project_rows(ctx, &first, (chunk, rows), patches, |row, col, value| {
                hidden.set(row * width + col, silu(value));
            });
            ctx.barrier();
            let activated = (hidden.read(), rows);
            project_rows(ctx, &second, activated, patches, |row, col, value| {
                out.set(row * HIDDEN + col, value);
            });
        });
        projected.extend_from_slice(&out.read()[..rows * HIDDEN]);
    }
    tensors.add("talker.text", vec![TEXT_VOCABULARY, HIDDEN], &projected)
}

/// The projection's layer `name` as an f32 `Linear` with its bias.
fn linear(checkpoint: &Checkpoint, name: &str) -> Result<Linear, Error> {
    let source = format!("{PROJECTION}.{name}");
    let weight_name = format!("{source}.weight");
    let shape = checkpoint.shape(&weight_name)?.to_vec();
    let weight = checkpoint.f32(&weight_name)?;
    let bias = checkpoint.f32(&format!("{source}.bias"))?;
    let format_error = |error: kernels::Error| Error::Format(format!("{source}: {error}"));
    let weight = Tensor::encode(DType::F32, shape, &weight).map_err(format_error)?;
    let bias = Tensor::encode(DType::F32, vec![bias.len()], &bias).map_err(format_error)?;
    let table = Table::new(weight).map_err(format_error)?;
    let bias = Vector::new(bias).map_err(format_error)?;
    Linear::new(table, Some(bias)).map_err(format_error)
}
