//! Transformer-XL relative-position attention and the sinusoids of its relative positions.

use kernels::ops::{rope_inv_freq, softmax};
use kernels::{DOTS, dot_f32_many};

use super::{Attention, Pass, Scratch, Shape};
use crate::SINUSOID_BASE;

/// Row `r` holds the sinusoids of the relative position `p = frames − 1 − r`, for `r` from 0 to
/// `2 · frames − 2`: `sin(p · fᵢ)` then `cos(p · fᵢ)` for each `fᵢ = 10000^(−2i / hidden)`.
/// The rows of a shorter sequence are the middle `2 · n − 1` rows.
pub(super) fn position_table(frames: usize, hidden: usize) -> Vec<f32> {
    let frequencies = rope_inv_freq(SINUSOID_BASE, hidden);
    let mut table = Vec::with_capacity((2 * frames - 1) * hidden);
    for row in 0..2 * frames - 1 {
        let position = (frames - 1) as f32 - row as f32;
        for frequency in &frequencies {
            let phase = position * frequency;
            table.extend([phase.sin(), phase.cos()]);
        }
    }
    table
}

/// Attention for this thread's groups of `DOTS` queries of a head into `mixed`: for query `i`
/// and key `j`, `((qᵢ + u) · kⱼ + (qᵢ + v) · rᵢ₋ⱼ) / √d`, softmax over `j`, times `vⱼ`. `rᵢ₋ⱼ` is
/// row `frames − 1 − i + j` of the projected positions. A group reads each key, position and
/// value row once for all its queries; the last group repeats its last query to fill it.
pub(super) fn attend(shape: Shape, pass: &Pass, weights: &Attention, scratch: &mut Scratch) {
    let head_dim = shape.head_dim();
    let frames = pass.frames;
    let groups = frames.div_ceil(DOTS);
    let Scratch {
        scores, queries, ..
    } = scratch;
    queries.resize(2 * DOTS * head_dim, 0.0);
    scores.resize(DOTS * frames, 0.0);
    for unit in pass.ctx.split(shape.heads * groups, 1) {
        let group = Group {
            head: unit / groups,
            first: unit % groups * DOTS,
            count: DOTS.min(frames - unit % groups * DOTS),
        };
        let (content, position) = queries.split_at_mut(DOTS * head_dim);
        group.biased_queries(&shape, pass, weights, (content, position));
        group.scores(&shape, pass, (content, position), scores);
        group.mix(&shape, pass, scores);
    }
}

/// `count` consecutive queries of a head, from `first`.
struct Group {
    head: usize,
    first: usize,
    count: usize,
}

impl Group {
    /// Query `member` of the group: the last one again past `count`.
    fn query(&self, member: usize) -> usize {
        self.first + member.min(self.count - 1)
    }

    /// Row `frame` of the head's queries (`part` 0), keys (1) or values (2).
    fn row<'call>(
        &self,
        shape: &Shape,
        pass: &'call Pass,
        (frame, part): (usize, usize),
    ) -> &'call [f32] {
        let hidden = shape.hidden;
        let head_dim = shape.head_dim();
        let start = frame * shape.qkv() + part * hidden + self.head * head_dim;
        &pass.buffers.qkv.read()[start..start + head_dim]
    }

    /// Each query plus the content bias, and plus the position bias.
    fn biased_queries(
        &self,
        shape: &Shape,
        pass: &Pass,
        weights: &Attention,
        (content, position): (&mut [f32], &mut [f32]),
    ) {
        let head_dim = shape.head_dim();
        let lane = self.head * head_dim..(self.head + 1) * head_dim;
        let bias_u = &weights.bias_u.values()[lane.clone()];
        let bias_v = &weights.bias_v.values()[lane];
        let rows = content
            .chunks_exact_mut(head_dim)
            .zip(position.chunks_exact_mut(head_dim));
        for (member, (content, position)) in rows.enumerate() {
            let query = self.row(shape, pass, (self.query(member), 0));
            for (i, value) in query.iter().enumerate() {
                content[i] = value + bias_u[i];
                position[i] = value + bias_v[i];
            }
        }
    }

    /// The scaled scores of every query of the group against every key, `DOTS × frames`.
    fn scores(
        &self,
        shape: &Shape,
        pass: &Pass,
        (content, position): (&[f32], &[f32]),
        scores: &mut [f32],
    ) {
        let (hidden, head_dim, frames) = (shape.hidden, shape.head_dim(), pass.frames);
        let (content, position) = (rows(content, head_dim), rows(position, head_dim));
        for key in 0..frames {
            let dots = dot_f32_many(self.row(shape, pass, (key, 1)), content);
            for (member, dot) in dots.into_iter().enumerate() {
                scores[member * frames + key] = dot;
            }
        }
        let positions = pass.buffers.positions.read();
        let last = self.first + self.count - 1;
        for relative in frames - 1 - last..2 * frames - 1 - self.first {
            let row = &positions[relative * hidden + self.head * head_dim..][..head_dim];
            let dots = dot_f32_many(row, position);
            for (member, dot) in dots.into_iter().enumerate().take(self.count) {
                let Some(key) = (relative + self.query(member)).checked_sub(frames - 1) else {
                    continue;
                };
                if key < frames {
                    scores[member * frames + key] += dot;
                }
            }
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        for score in scores.iter_mut() {
            *score *= scale;
        }
    }

    /// Each query's softmax over its scores, times the values, into `mixed`.
    fn mix(&self, shape: &Shape, pass: &Pass, scores: &mut [f32]) {
        let (hidden, head_dim, frames) = (shape.hidden, shape.head_dim(), pass.frames);
        for row in scores.chunks_exact_mut(frames).take(self.count) {
            softmax(row);
        }
        let mut outputs: [Option<&mut [f32]>; DOTS] = std::array::from_fn(|member| {
            let at = (self.first + member) * hidden + self.head * head_dim;
            (member < self.count).then(|| pass.buffers.mixed.part(at..at + head_dim))
        });
        for out in outputs.iter_mut().flatten() {
            out.fill(0.0);
        }
        for key in 0..frames {
            let value = self.row(shape, pass, (key, 2));
            for (member, out) in outputs.iter_mut().flatten().enumerate() {
                let weight = scores[member * frames + key];
                for (output, component) in out.iter_mut().zip(value) {
                    *output += weight * component;
                }
            }
        }
    }
}

/// The `DOTS` rows of `head_dim` values in `queries`.
fn rows(queries: &[f32], head_dim: usize) -> [&[f32]; DOTS] {
    std::array::from_fn(|member| &queries[member * head_dim..(member + 1) * head_dim])
}
