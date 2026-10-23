//! The reading-order network: per line its features and Fourier features of its box (the sine
//! and cosine of each coordinate times each frequency), an MLP to the encoder's width, an
//! encoder over every line of the page (`blocks::transformer`, `Architecture::Encoder`) and a
//! final LayerNorm; then pairwise heads: `order[i][j] > 0` when line `i` reads before line `j`
//! (`S − Sᵀ`, `S = q kᵀ / √width`), `same[i][j] > 0` when both lines belong to one layout block
//! (`B + Bᵀ`, `B = u wᵀ / √width`), and each line's layout-class logits.
//!
//! Tensors under `order.`: `frequencies`, `inp.0`, `inp.2`, the encoder's layers, `norm`, `q`,
//! `k`, `bu`, `bw` and `cls`.

use blocks::norm::LayerNorm;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture, KvCache, Mask, Span, Transformer};
use bundle::{Load, Store};
use kernels::ops::gelu_erf;
use kernels::{Linear, QuantizedRows, Vector, dot_f32};
use team::Team;

use crate::Error;

pub(crate) const PREFIX: &str = "order";
/// The tensor of the Fourier frequencies, under `PREFIX`.
pub const FREQUENCIES: &str = "frequencies";
/// A line's features (`reading::features`).
pub const FEATURES: usize = 16;
/// The box coordinates leading the features, and the frequencies each is turned by.
const COORDINATES: usize = 4;
const FREQUENCY_COUNT: usize = 16;
const INPUT: usize = FEATURES + COORDINATES * FREQUENCY_COUNT * 2;
const WIDTH: usize = 128;
const EPS: f32 = 1e-5;
/// The layout classes (`layout::CLASSES`).
pub const CLASSES: usize = 10;
const ENCODER: transformer::Shape = transformer::Shape {
    architecture: Architecture::Encoder,
    hidden: WIDTH,
    layers: 4,
    heads: 4,
    kv_heads: 4,
    head_dim: WIDTH / 4,
    feed_forward: 256,
    rope_base: 0.0,
    rotation: Rotation::Sequence,
    norm_eps: EPS,
};

pub struct Order {
    frequencies: Vector,
    input: [Linear; 2],
    encoder: Transformer,
    norm: LayerNorm,
    query: Linear,
    key: Linear,
    block_query: Linear,
    block_key: Linear,
    classes: Linear,
}

/// The heads of `lines` lines: `order` and `same`, `lines × lines`; `classes`, `lines ×
/// CLASSES`.
pub struct Relations {
    pub lines: usize,
    pub order: Vec<f32>,
    pub same: Vec<f32>,
    pub classes: Vec<f32>,
}

impl Order {
    /// The heads over lines of `features` (`lines × FEATURES`).
    pub fn forward(&self, team: &Team, features: &[f32]) -> Relations {
        let lines = features.len() / FEATURES;
        let mut quant = QuantizedRows::default();
        let input = self.fourier(features);
        let mut inner = vec![0.0; lines * WIDTH];
        self.input[0].apply(&input, &mut quant, &mut inner);
        inner.iter_mut().for_each(|value| *value = gelu_erf(*value));
        let mut buffers = ENCODER.buffers(lines, team.threads());
        self.input[1].apply(
            &inner,
            &mut quant,
            &mut buffers.hidden.get_mut()[..lines * WIDTH],
        );
        let cache = KvCache::new(ENCODER, lines);
        let span = Span {
            rows: lines,
            first: 0,
            mask: Mask::Full,
        };
        team.run(|ctx| self.encoder.forward(ctx, &cache, &buffers, span));
        let mut normed = vec![0.0; lines * WIDTH];
        let rows = buffers.hidden.read().as_chunks::<WIDTH>().0;
        for (row, out) in rows.iter().zip(normed.as_chunks_mut::<WIDTH>().0) {
            self.norm.apply(row, EPS, out);
        }
        let project = |linear: &Linear| {
            let mut out = vec![0.0; lines * linear.matrix().rows];
            linear.apply(&normed, &mut QuantizedRows::default(), &mut out);
            out
        };
        Relations {
            lines,
            order: pairwise(&project(&self.query), &project(&self.key), lines, -1.0),
            same: pairwise(
                &project(&self.block_query),
                &project(&self.block_key),
                lines,
                1.0,
            ),
            classes: project(&self.classes),
        }
    }

    /// Each line's features, then the sine and the cosine of each box coordinate times each
    /// frequency.
    fn fourier(&self, features: &[f32]) -> Vec<f32> {
        let frequencies = self.frequencies.values();
        let mut out = Vec::with_capacity(features.len() / FEATURES * INPUT);
        for line in features.as_chunks::<FEATURES>().0 {
            out.extend_from_slice(line);
            for &coordinate in &line[..COORDINATES] {
                let angles = frequencies.iter().map(|frequency| coordinate * frequency);
                out.extend(angles.clone().map(f32::sin));
                out.extend(angles.map(f32::cos));
            }
        }
        out
    }
}

/// `P + sign · Pᵀ` with `P[i][j] = left_i · right_j / √WIDTH`.
fn pairwise(left: &[f32], right: &[f32], lines: usize, sign: f32) -> Vec<f32> {
    let scale = (WIDTH as f32).sqrt();
    let product = |i: usize, j: usize| {
        dot_f32(&left[i * WIDTH..][..WIDTH], &right[j * WIDTH..][..WIDTH]) / scale
    };
    let mut out = vec![0.0; lines * lines];
    for i in 0..lines {
        for j in 0..lines {
            out[i * lines + j] = product(i, j) + sign * product(j, i);
        }
    }
    out
}

impl Load for Order {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let head = |store: &mut Store, part: &str, rows: usize| {
            Linear::load(store, &name(part), [rows, WIDTH])
        };
        Ok(Self {
            frequencies: Vector::load(store, &name(FREQUENCIES), FREQUENCY_COUNT)?,
            input: [
                Linear::load(store, &name("inp.0"), [WIDTH, INPUT])?,
                Linear::load(store, &name("inp.2"), [WIDTH, WIDTH])?,
            ],
            encoder: Transformer::load(store, prefix, ENCODER)?,
            norm: LayerNorm::load(store, &name("norm"), WIDTH)?,
            query: head(store, "q", WIDTH)?,
            key: head(store, "k", WIDTH)?,
            block_query: head(store, "bu", WIDTH)?,
            block_key: head(store, "bw", WIDTH)?,
            classes: head(store, "cls", CLASSES)?,
        })
    }
}
