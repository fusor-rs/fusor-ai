//! The math decoder: PyTorch's pre-norm decoder layers (self-attention over the tokens so far,
//! attention over the memory's valid steps, an exact-GELU MLP), a final LayerNorm and a head
//! over the tokens. Each step reads one token per line at one position; the keys and values of
//! the tokens before it and of the memory stay cached.
//!
//! Tensors under `math_dec.c.dec.`: `emb.weight`, `pos.weight`, `layers.layers.{i}.{norm1,
//! self_attn.in_proj, self_attn.out_proj, norm2, multihead_attn.in_proj_weight,
//! multihead_attn.in_proj_bias, multihead_attn.out_proj, norm3, linear1, linear2}`, `norm` and
//! `out`.

use blocks::norm::LayerNorm;
use bundle::{Load, Store, load_each};
use kernels::conv::project_rows;
use kernels::ops::{attend, gelu_erf};
use kernels::{DType, Linear, Table, Tensor, Vector};
use team::{SharedBuf, Team};

use crate::Error;
use crate::math::encoder::{Memory, WIDTH, normalize};
use crate::plane::{Layers, Scratch};

/// The longest reading in tokens, its first (the start token) included.
pub const MAX_TOKENS: usize = 320;
/// The tokens: four special ones, then the text's.
pub const TOKENS: usize = 652;
const LAYERS: usize = 3;
const HEADS: usize = 8;
const HEAD_DIM: usize = WIDTH / HEADS;
const FEED_FORWARD: usize = 768;

pub struct Decoder {
    embedding: Table,
    positions: Table,
    layers: Vec<Layer>,
    norm: LayerNorm,
    head: Linear,
}

struct Layer {
    norm1: LayerNorm,
    self_qkv: Linear,
    self_out: Linear,
    norm2: LayerNorm,
    cross_query: Linear,
    cross_kv: Linear,
    cross_out: Linear,
    norm3: LayerNorm,
    up: Linear,
    down: Linear,
}

/// A batch's decoding: the memory's keys and values, each line's keys and values of the tokens
/// read so far, and one step's activations.
pub struct State {
    /// The position the next step reads at.
    position: usize,
    lengths: Vec<usize>,
    cross: Cache,
    tokens: Cache,
    x: SharedBuf,
    normed: SharedBuf,
    projected: SharedBuf,
    attended: SharedBuf,
    inner: SharedBuf,
    logits: SharedBuf,
}

/// Keys and values: `lines × LAYERS × HEADS × capacity` heads of `HEAD_DIM`.
struct Cache {
    capacity: usize,
    keys: SharedBuf,
    values: SharedBuf,
}

impl Cache {
    fn new(lines: usize, capacity: usize) -> Self {
        let len = lines * LAYERS * HEADS * capacity * HEAD_DIM;
        Self {
            capacity,
            keys: SharedBuf::new(len),
            values: SharedBuf::new(len),
        }
    }

    fn offset(&self, line: usize, layer: usize, head: usize, position: usize) -> usize {
        (((line * LAYERS + layer) * HEADS + head) * self.capacity + position) * HEAD_DIM
    }

    /// Keeps `keys_values` (a row's keys, then its values) at `position`.
    fn keep(&self, (line, layer, position): (usize, usize, usize), keys_values: &[f32]) {
        for head in 0..HEADS {
            let at = self.offset(line, layer, head, position);
            let key = &keys_values[head * HEAD_DIM..][..HEAD_DIM];
            let value = &keys_values[WIDTH + head * HEAD_DIM..][..HEAD_DIM];
            self.keys.part(at..at + HEAD_DIM).copy_from_slice(key);
            self.values.part(at..at + HEAD_DIM).copy_from_slice(value);
        }
    }

    /// `query`'s attention over positions `0..count` of a line's head into `out`.
    fn attend(
        &self,
        (line, layer, head): (usize, usize, usize),
        (query, count): (&[f32], usize),
        scores: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        let (keys, values) = (self.keys.read(), self.values.read());
        let first = self.offset(line, layer, head, 0);
        attend(
            query,
            count,
            |position| {
                let at = first + position * HEAD_DIM;
                (&keys[at..at + HEAD_DIM], &values[at..at + HEAD_DIM])
            },
            scores,
            out,
        );
    }
}

/// A line still being read, and the token it reads at this step.
#[derive(Debug, Clone, Copy)]
pub struct Reading {
    pub line: usize,
    pub token: u32,
}

impl Decoder {
    /// The decoding state of a batch, its memory's keys and values computed.
    pub fn start(&self, team: &Team, scratch: &Scratch, memory: &Memory) -> State {
        let lines = memory.lengths.len();
        let rows = lines * memory.steps;
        let state = State {
            position: 0,
            lengths: memory.lengths.clone(),
            cross: Cache::new(lines, memory.steps),
            tokens: Cache::new(lines, MAX_TOKENS),
            x: SharedBuf::new(lines * WIDTH),
            normed: SharedBuf::new(lines * WIDTH),
            projected: SharedBuf::new(rows.max(lines) * 3 * WIDTH),
            attended: SharedBuf::new(lines * WIDTH),
            inner: SharedBuf::new(lines * FEED_FORWARD),
            logits: SharedBuf::new(lines * TOKENS),
        };
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            let patches = layers.patches.get(ctx);
            for (index, layer) in self.layers.iter().enumerate() {
                project_rows(
                    ctx,
                    &layer.cross_kv,
                    (&memory.values, rows),
                    patches,
                    |row, column, value| {
                        state.projected.set(row * 2 * WIDTH + column, value);
                    },
                );
                ctx.barrier();
                for row in ctx.split(rows, 1) {
                    let keys_values =
                        &state.projected.read()[row * 2 * WIDTH..(row + 1) * 2 * WIDTH];
                    let at = (row / memory.steps, index, row % memory.steps);
                    state.cross.keep(at, keys_values);
                }
                ctx.barrier();
            }
        });
        state
    }

    /// The logits of each of `readings`' next token (`readings.len() × TOKENS`), every line
    /// reading at the state's position, which then moves on.
    pub fn step(
        &self,
        team: &Team,
        scratch: &Scratch,
        state: &mut State,
        readings: &[Reading],
    ) -> Vec<f32> {
        let rows = readings.len();
        let position = state.position;
        let state_ref = &*state;
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            self.embed(&layers, state_ref, readings);
            for (layer, weights) in self.layers.iter().enumerate() {
                let pass = Pass {
                    layers: &layers,
                    state: state_ref,
                    readings,
                    layer,
                };
                weights.forward(&pass);
            }
            normalize(&layers, &self.norm, &state_ref.x, &state_ref.normed, rows);
            let normed = (state_ref.normed.read(), rows);
            project_rows(
                ctx,
                &self.head,
                normed,
                layers.patches.get(ctx),
                |row, token, value| {
                    state_ref.logits.set(row * TOKENS + token, value);
                },
            );
        });
        state.position = position + 1;
        state.logits.read()[..rows * TOKENS].to_vec()
    }

    /// Each row's token embedding plus its position's.
    fn embed(&self, layers: &Layers, state: &State, readings: &[Reading]) {
        let mut place = [0.0; WIDTH];
        self.positions
            .matrix()
            .dequantize_row(state.position, &mut place);
        for row in layers.ctx.split(readings.len(), 1) {
            let x = state.x.part(row * WIDTH..(row + 1) * WIDTH);
            self.embedding
                .matrix()
                .dequantize_row(readings[row].token as usize, x);
            x.iter_mut()
                .zip(place)
                .for_each(|(value, place)| *value += place);
        }
        layers.ctx.barrier();
    }
}

/// One layer's step: the thread's layers, the state, the lines read and the layer's index.
struct Pass<'step> {
    layers: &'step Layers<'step>,
    state: &'step State,
    readings: &'step [Reading],
    layer: usize,
}

impl Layer {
    fn forward(&self, pass: &Pass) {
        let (layers, state) = (pass.layers, pass.state);
        let rows = pass.readings.len();
        normalize(layers, &self.norm1, &state.x, &state.normed, rows);
        project(
            layers,
            &self.self_qkv,
            (&state.normed, rows),
            &state.projected,
        );
        for row in layers.ctx.split(rows, 1) {
            let keys_values =
                &state.projected.read()[row * 3 * WIDTH + WIDTH..(row + 1) * 3 * WIDTH];
            let at = (pass.readings[row].line, pass.layer, state.position);
            state.tokens.keep(at, keys_values);
        }
        layers.ctx.barrier();
        attend_heads(pass, &state.tokens, 3 * WIDTH, |_| state.position + 1);
        add_projection(layers, &self.self_out, &state.attended, &state.x, rows);
        normalize(layers, &self.norm2, &state.x, &state.normed, rows);
        project(
            layers,
            &self.cross_query,
            (&state.normed, rows),
            &state.projected,
        );
        attend_heads(pass, &state.cross, WIDTH, |line| state.lengths[line]);
        add_projection(layers, &self.cross_out, &state.attended, &state.x, rows);
        normalize(layers, &self.norm3, &state.x, &state.normed, rows);
        let normed = (state.normed.read(), rows);
        project_rows(
            layers.ctx,
            &self.up,
            normed,
            layers.patches.get(layers.ctx),
            |row, column, value| {
                state
                    .inner
                    .set(row * FEED_FORWARD + column, gelu_erf(value));
            },
        );
        layers.ctx.barrier();
        add_projection(layers, &self.down, &state.inner, &state.x, rows);
    }
}

/// `into` = `linear` of the first `rows` rows of `input`.
fn project(layers: &Layers, linear: &Linear, (input, rows): (&SharedBuf, usize), into: &SharedBuf) {
    let outputs = linear.matrix().rows;
    let patches = layers.patches.get(layers.ctx);
    project_rows(
        layers.ctx,
        linear,
        (input.read(), rows),
        patches,
        |row, column, value| {
            into.set(row * outputs + column, value);
        },
    );
    layers.ctx.barrier();
}

/// Each row's query heads (leading its row of `state.projected`, `stride` wide) attending over
/// the first `count(line)` positions of its line in `cache`, into `state.attended`.
fn attend_heads(pass: &Pass, cache: &Cache, stride: usize, count: impl Fn(usize) -> usize) {
    let (layers, state) = (pass.layers, pass.state);
    let scores = layers.scratch.get(layers.ctx);
    for unit in layers.ctx.split(pass.readings.len() * HEADS, 1) {
        let (row, head) = (unit / HEADS, unit % HEADS);
        let line = pass.readings[row].line;
        let query = &state.projected.read()[row * stride + head * HEAD_DIM..][..HEAD_DIM];
        let at = row * WIDTH + head * HEAD_DIM;
        let out = state.attended.part(at..at + HEAD_DIM);
        cache.attend((line, pass.layer, head), (query, count(line)), scores, out);
    }
    layers.ctx.barrier();
}

/// `x += linear(input)` over the first `rows` rows.
fn add_projection(layers: &Layers, linear: &Linear, input: &SharedBuf, x: &SharedBuf, rows: usize) {
    let patches = layers.patches.get(layers.ctx);
    project_rows(
        layers.ctx,
        linear,
        (input.read(), rows),
        patches,
        |row, channel, value| {
            let at = row * WIDTH + channel;
            x.set(at, x.get(at) + value);
        },
    );
    layers.ctx.barrier();
}

impl Load for Decoder {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let layers = load_each(store, &name("layers.layers"), LAYERS, |store, layer| {
            Layer::load(store, layer, ())
        })?;
        Ok(Self {
            embedding: Table::load(store, &name("emb.weight"), [TOKENS, WIDTH])?,
            positions: Table::load(store, &name("pos.weight"), [MAX_TOKENS + 1, WIDTH])?,
            layers,
            norm: LayerNorm::load(store, &name("norm"), WIDTH)?,
            head: Linear::load(store, &name("out"), [TOKENS, WIDTH])?,
        })
    }
}

impl Load for Layer {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let weight = Table::load(
            store,
            &name("multihead_attn.in_proj_weight"),
            [3 * WIDTH, WIDTH],
        )?;
        let bias = Vector::load(store, &name("multihead_attn.in_proj_bias"), 3 * WIDTH)?;
        let (weight, bias) = (weight.dequantize(), bias.values());
        Ok(Self {
            norm1: LayerNorm::load(store, &name("norm1"), WIDTH)?,
            self_qkv: Linear::load(store, &name("self_attn.in_proj"), [3 * WIDTH, WIDTH])?,
            self_out: Linear::load(store, &name("self_attn.out_proj"), [WIDTH, WIDTH])?,
            norm2: LayerNorm::load(store, &name("norm2"), WIDTH)?,
            cross_query: rows_of(&weight, bias, 0..WIDTH)?,
            cross_kv: rows_of(&weight, bias, WIDTH..3 * WIDTH)?,
            cross_out: Linear::load(store, &name("multihead_attn.out_proj"), [WIDTH, WIDTH])?,
            norm3: LayerNorm::load(store, &name("norm3"), WIDTH)?,
            up: Linear::load(store, &name("linear1"), [FEED_FORWARD, WIDTH])?,
            down: Linear::load(store, &name("linear2"), [WIDTH, FEED_FORWARD])?,
        })
    }
}

/// The linear layer of rows `rows` of an f32 projection (`weight`, `bias`).
fn rows_of(weight: &[f32], bias: &[f32], rows: std::ops::Range<usize>) -> Result<Linear, Error> {
    let shape = vec![rows.len(), WIDTH];
    let weight = Tensor::encode(
        DType::F32,
        shape,
        &weight[rows.start * WIDTH..rows.end * WIDTH],
    )?;
    let bias = Tensor::encode(DType::F32, vec![rows.len()], &bias[rows])?;
    Ok(Linear::new(Table::new(weight)?, Some(Vector::new(bias)?))?)
}
