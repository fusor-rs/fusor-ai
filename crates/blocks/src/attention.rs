//! Softmax attention without a mask over the rows of shared buffers: every query of an image
//! sees every key of that image (a spatial self-attention, or a cross-attention to a context).
//!
//! Both products run on the int8 kernels. Per (image, head), a thread quantizes the keys as
//! `Q7b32` rows and the values transposed (a row per channel, a column per key), each padded
//! with zeros to whole blocks; a unit's queries are quantized as the activations of `Q·Kᵀ`, and
//! their exponentials as those of `P·Vᵀ`, whose outputs are divided by the exponentials' sums
//! (the softmax's normalization, on `head_dim` values rather than every key's). The keys lose
//! their mean first (SageAttention's smoothing): that shifts all of a query's scores alike,
//! which the softmax ignores, and leaves the quantization a smaller range. `attend` splits
//! (image, head, block of `QUERY_BLOCK` queries) units over the team; a thread quantizes a head
//! once for its consecutive units.

use std::ops::Range;

use kernels::ops::exp;
use kernels::{BLOCK, QuantizedMatrix, QuantizedRows, matmul};
use team::{Ctx, SharedBuf};

/// Queries per unit.
const QUERY_BLOCK: usize = BLOCK;
/// Independent partial maxima and sums over a row of scores, so that their loops vectorize.
const LANES: usize = BLOCK / 4;

/// The rows of a query, key or value operand: head `h` of row `r` is the `head_dim` values at
/// `r · stride + offset + h · head_dim`.
#[derive(Clone, Copy)]
pub(crate) struct Operand<'buf> {
    pub(crate) values: &'buf [f32],
    pub(crate) stride: usize,
    pub(crate) offset: usize,
}

/// One attention: `images` images, each of `queries` query rows over `keys` key and value rows,
/// in `heads` heads of `head_dim`.
pub(crate) struct Call<'buf> {
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) images: usize,
    pub(crate) queries: usize,
    pub(crate) keys: usize,
    pub(crate) query: Operand<'buf>,
    pub(crate) key: Operand<'buf>,
    pub(crate) value: Operand<'buf>,
}

/// One thread's working set.
#[derive(Default)]
pub(crate) struct Scratch {
    /// The (image, head) whose keys and values are quantized.
    head: Option<(usize, usize)>,
    keys: QuantizedMatrix,
    values: QuantizedMatrix,
    /// The keys' mean, and a row on its way into `keys` or `values`.
    mean: Vec<f32>,
    row: Vec<f32>,
    /// A unit's queries, padded; then their scores and their exponentials, padded, and the
    /// exponentials' sum of each query.
    queries: Vec<f32>,
    scores: Vec<f32>,
    sums: Vec<f32>,
    quant: QuantizedRows,
}

impl Operand<'_> {
    fn head(&self, row: usize, head: usize, head_dim: usize) -> &[f32] {
        let start = row * self.stride + self.offset + head * head_dim;
        &self.values[start..start + head_dim]
    }
}

/// `length` rounded up to whole blocks.
fn padded(length: usize) -> usize {
    length.div_ceil(BLOCK) * BLOCK
}

/// This thread's share of `call` into `out`, rows of `heads · head_dim` values, the scores
/// scaled by `1/√head_dim`.
pub(crate) fn attend(ctx: &Ctx, call: &Call, out: &SharedBuf, scratch: &mut Scratch) {
    let blocks = call.queries.div_ceil(QUERY_BLOCK);
    scratch.head = None;
    for unit in ctx.split(call.images * call.heads * blocks, 1) {
        let (pair, block) = (unit / blocks, unit % blocks);
        let head = (pair / call.heads, pair % call.heads);
        if scratch.head != Some(head) {
            scratch.quantize(call, head);
        }
        let first = block * QUERY_BLOCK;
        let queries = first..(first + QUERY_BLOCK).min(call.queries);
        scratch.unit(call, head, queries, out);
    }
}

impl Scratch {
    /// The keys and the transposed values of `(image, head)`, padded.
    fn quantize(&mut self, call: &Call, (image, head): (usize, usize)) {
        let (head_dim, keys) = (call.head_dim, call.keys);
        let first = image * keys;
        self.mean.clear();
        self.mean.resize(head_dim, 0.0);
        for key in first..first + keys {
            let row = call.key.head(key, head, head_dim);
            for (sum, value) in self.mean.iter_mut().zip(row) {
                *sum += value;
            }
        }
        for sum in &mut self.mean {
            *sum /= keys as f32;
        }
        self.keys.reset(padded(head_dim));
        self.row.clear();
        self.row.resize(padded(head_dim), 0.0);
        for key in first..first + keys {
            let row = call.key.head(key, head, head_dim);
            for ((slot, value), mean) in self.row.iter_mut().zip(row).zip(&self.mean) {
                *slot = value - mean;
            }
            self.keys.push(&self.row);
        }
        self.values.reset(padded(keys));
        self.row.clear();
        self.row.resize(padded(keys), 0.0);
        for channel in 0..head_dim {
            for (key, slot) in self.row[..keys].iter_mut().enumerate() {
                *slot = call.value.head(first + key, head, head_dim)[channel];
            }
            self.values.push(&self.row);
        }
        self.head = Some((image, head));
    }

    /// The attention of `queries` of `(image, head)` into `out`.
    fn unit(
        &mut self,
        call: &Call,
        (image, head): (usize, usize),
        queries: Range<usize>,
        out: &SharedBuf,
    ) {
        let Self {
            keys,
            values,
            queries: padded_queries,
            scores,
            sums,
            quant,
            ..
        } = self;
        let (head_dim, width) = (call.head_dim, padded(call.head_dim));
        let first = image * call.queries + queries.start;
        padded_queries.clear();
        padded_queries.resize(queries.len() * width, 0.0);
        for (row, padded_query) in padded_queries.chunks_exact_mut(width).enumerate() {
            padded_query[..head_dim].copy_from_slice(call.query.head(first + row, head, head_dim));
        }
        let (keys_matrix, columns) = (keys.matrix(), padded(call.keys));
        scores.clear();
        scores.resize(queries.len() * columns, 0.0);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let input = quant.input(&keys_matrix, padded_queries);
        matmul(&keys_matrix, &input, 0..call.keys, |row, key, value| {
            scores[row * columns + key] = value * scale;
        });
        sums.clear();
        let rows = scores.chunks_exact_mut(columns);
        sums.extend(rows.map(|row| exponentials(&mut row[..call.keys])));
        let values_matrix = values.matrix();
        let input = quant.input(&values_matrix, scores);
        let stride = call.heads * head_dim;
        matmul(
            &values_matrix,
            &input,
            0..head_dim,
            |row, channel, value| {
                out.set(
                    (first + row) * stride + head * head_dim + channel,
                    value / sums[row],
                );
            },
        );
    }
}

/// `scores` replaced by `exp(score − max)`; returns their sum.
fn exponentials(scores: &mut [f32]) -> f32 {
    let mut maxima = [f32::NEG_INFINITY; LANES];
    let (lanes, rest) = scores.as_chunks::<LANES>();
    for chunk in lanes {
        for (max, score) in maxima.iter_mut().zip(chunk) {
            *max = max.max(*score);
        }
    }
    let max = rest
        .iter()
        .chain(&maxima)
        .fold(f32::NEG_INFINITY, |max, &score| max.max(score));
    let mut sums = [0.0; LANES];
    let (lanes, rest) = scores.as_chunks_mut::<LANES>();
    for chunk in lanes {
        for (sum, score) in sums.iter_mut().zip(chunk) {
            *score = exp(*score - max);
            *sum += *score;
        }
    }
    for score in rest.iter_mut() {
        *score = exp(*score - max);
    }
    rest.iter().chain(&sums).sum()
}
