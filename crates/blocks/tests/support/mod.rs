//! What the block tests share: seeded weights written as a bundle and streamed into a `Store` the
//! way a runtime loads one, and f64 references of the layers written from the architectures'
//! definitions, independently of the kernels.

use bundle::{Source, Store};
use compile::BundleWriter;
use kernels::{DType, Tensor};

/// Bytes per write into the store, fewer than a part's: parts arrive in pieces.
const CHUNK: usize = 4096;
/// Layers per stack: the second reads what the first wrote.
pub const LAYERS: usize = 2;

/// Named tensors of seeded values, kept as f64 for the references.
pub struct Weights {
    name: &'static str,
    state: u64,
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl Weights {
    /// `name` names the bundle's temporary directory: one per test.
    pub fn new(name: &'static str, seed: u64) -> Self {
        Self {
            name,
            state: seed,
            tensors: Vec::new(),
        }
    }

    /// `count` values uniform in `[−scale, scale]`, from a 64-bit LCG (Knuth's constants).
    fn values(&mut self, count: usize, scale: f64) -> Vec<f64> {
        (0..count)
            .map(|_| {
                self.state = self
                    .state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (self.state >> 11) as f64 / (1u64 << 53) as f64;
                (2.0 * unit - 1.0) * scale
            })
            .collect()
    }

    /// `count` f32 values within `scale` of `center`.
    pub fn floats(&mut self, count: usize, center: f64, scale: f64) -> Vec<f32> {
        self.values(count, scale)
            .into_iter()
            .map(|value| (center + value) as f32)
            .collect()
    }

    /// Tensor `name` of `shape`, seeded values within `scale` of `center`, returned as the f64 of
    /// its f32 values.
    pub fn tensor(&mut self, name: &str, shape: &[usize], center: f64, scale: f64) -> Vec<f64> {
        let values = self.floats(shape.iter().product(), center, scale);
        let exact = widen(&values);
        self.tensors.push((name.into(), shape.to_vec(), values));
        exact
    }

    /// The linear layer `name`, `[outputs, inputs]`.
    pub fn linear(&mut self, name: &str, [outputs, inputs]: [usize; 2], bias: Bias) -> Linear {
        let scale = 1.0 / (inputs as f64).sqrt();
        Linear {
            weight: self.tensor(&format!("{name}.weight"), &[outputs, inputs], 0.0, scale),
            bias: match bias {
                Bias::With => Some(self.tensor(&format!("{name}.bias"), &[outputs], 0.0, 0.1)),
                Bias::Without => None,
            },
            inputs,
        }
    }

    fn norm(&mut self, name: &str, width: usize, kind: NormKind) -> Norm {
        match kind {
            NormKind::Rms => Norm::Rms(self.tensor(name, &[width], 1.0, 0.2)),
            NormKind::Layer => Norm::Layer(
                self.tensor(&format!("{name}.weight"), &[width], 1.0, 0.2),
                self.tensor(&format!("{name}.bias"), &[width], 0.0, 0.1),
            ),
        }
    }

    /// `LAYERS` layers of `kind`, the `i`th under `prefix.i`.
    pub fn stack(&mut self, prefix: &str, dims: &Dims, kind: Kind) -> Stack {
        let layers = (0..LAYERS).map(|i| {
            let prefix = format!("{prefix}.{i}");
            match kind {
                Kind::Llama(llama) => self.llama_layer(&prefix, dims, llama),
                Kind::Plain(plain) => self.plain_layer(&prefix, dims, plain),
            }
        });
        Stack(layers.collect())
    }

    /// A Llama-style layer with Qwen2's names (`input_norm`, `qkv`, `o`, `post_norm`, `gate`,
    /// `up`, `down`, and Qwen3's `q_norm` and `k_norm`): RMSNorms and a SiLU-gated MLP; only
    /// `qkv` may have a bias.
    fn llama_layer(&mut self, prefix: &str, dims: &Dims, llama: Llama) -> Layer {
        let Dims {
            hidden,
            feed_forward,
            head_dim,
            ..
        } = *dims;
        let name = |part: &str| format!("{prefix}.{part}");
        let norm1 = self.norm(&name("input_norm"), hidden, NormKind::Rms);
        let qkv = self.linear(&name("qkv"), [dims.qkv(), hidden], llama.qkv_bias);
        let out = self.linear(&name("o"), [hidden, dims.width()], Bias::Without);
        let norm2 = self.norm(&name("post_norm"), hidden, NormKind::Rms);
        let gate = self.linear(&name("gate"), [feed_forward, hidden], Bias::Without);
        let up = self.linear(&name("up"), [feed_forward, hidden], Bias::Without);
        let down = self.linear(&name("down"), [hidden, feed_forward], Bias::Without);
        let head_norms = match llama.head_norms {
            HeadNorms::Without => None,
            HeadNorms::With => Some([
                self.tensor(&name("q_norm"), &[head_dim], 1.0, 0.2),
                self.tensor(&name("k_norm"), &[head_dim], 1.0, 0.2),
            ]),
        };
        Layer {
            norm1,
            qkv,
            head_norms,
            out,
            norm2,
            feed_forward: FeedForward::Gated { gate, up, down },
        }
    }

    /// A plain layer (`norm1`, `qkv`, `out`, `norm2`, `ff1`, `ff2`): Kyutai's, CLIP's, a ViT's or
    /// a DiT's.
    fn plain_layer(&mut self, prefix: &str, dims: &Dims, plain: Plain) -> Layer {
        let Dims {
            hidden,
            feed_forward,
            ..
        } = *dims;
        let name = |part: &str| format!("{prefix}.{part}");
        let norm1 = self.norm(&name("norm1"), hidden, plain.norm);
        let qkv = self.linear(&name("qkv"), [dims.qkv(), hidden], plain.bias);
        let out = self.linear(&name("out"), [hidden, dims.width()], plain.bias);
        let norm2 = self.norm(&name("norm2"), hidden, plain.norm);
        let up = self.linear(&name("ff1"), [feed_forward, hidden], plain.bias);
        let down = self.linear(&name("ff2"), [hidden, feed_forward], plain.bias);
        Layer {
            norm1,
            qkv,
            head_norms: None,
            out,
            norm2,
            feed_forward: FeedForward::Plain {
                up,
                down,
                activation: plain.activation,
            },
        }
    }

    /// The bundle of every tensor added, its matrices as `matrices`, streamed into a store.
    pub fn store(self, matrices: DType) -> Store {
        let dir = std::env::temp_dir().join(format!(
            "fusor-ai-blocks-{}-{}",
            self.name,
            std::process::id()
        ));
        let mut writer = BundleWriter::create(&dir).expect("the temporary directory is writable");
        for (name, shape, values) in &self.tensors {
            let dtype = if shape.len() == 2 {
                matrices
            } else {
                DType::F32
            };
            let tensor = Tensor::encode(dtype, shape.clone(), values).expect("the shapes fit");
            writer
                .add_tensor(name, &tensor)
                .expect("each name is added once");
        }
        let source = Source {
            repo: "test/blocks".into(),
            revision: "0".into(),
            profile: "test".into(),
        };
        let manifest = writer.finish(source).expect("the bundle is written");
        let mut store = Store::new(manifest.clone()).expect("the manifest is the writer's");
        for (index, part) in manifest.parts.iter().enumerate() {
            let bytes = std::fs::read(dir.join(&part.file)).expect("the writer wrote the part");
            for piece in bytes.chunks(CHUNK) {
                store.write(index, piece).expect("the part's bytes fit it");
            }
            store.end_part(index).expect("the part hashes as written");
        }
        std::fs::remove_dir_all(&dir).expect("the temporary directory is removable");
        store
    }
}

#[derive(Clone, Copy)]
pub enum Bias {
    With,
    Without,
}

/// Whether a layer RMS-normalizes each query and key head (Qwen3).
#[derive(Clone, Copy)]
pub enum HeadNorms {
    Without,
    With,
}

#[derive(Clone, Copy)]
pub struct Llama {
    pub qkv_bias: Bias,
    pub head_norms: HeadNorms,
}

#[derive(Clone, Copy)]
pub enum Kind {
    Llama(Llama),
    Plain(Plain),
}

#[derive(Clone, Copy)]
pub enum NormKind {
    Rms,
    Layer,
}

/// A plain layer's norms, whether its linears have biases, and its activation.
#[derive(Clone, Copy)]
pub struct Plain {
    pub norm: NormKind,
    pub bias: Bias,
    pub activation: fn(f64) -> f64,
}

/// A layer's dimensions.
#[derive(Clone, Copy)]
pub struct Dims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub feed_forward: usize,
    pub eps: f64,
}

impl Dims {
    fn qkv(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

pub struct Linear {
    weight: Vec<f64>,
    bias: Option<Vec<f64>>,
    inputs: usize,
}

impl Linear {
    /// `x · Wᵀ + b` of every row of `x`.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let weights = self.weight.chunks_exact(self.inputs);
        x.chunks_exact(self.inputs)
            .flat_map(|row| {
                weights.clone().enumerate().map(move |(output, weight)| {
                    let bias = self.bias.as_ref().map_or(0.0, |bias| bias[output]);
                    bias + dot(weight, row)
                })
            })
            .collect()
    }

    /// Row `row` of the weight.
    pub fn row(&self, row: usize) -> &[f64] {
        &self.weight[row * self.inputs..(row + 1) * self.inputs]
    }
}

enum Norm {
    Rms(Vec<f64>),
    Layer(Vec<f64>, Vec<f64>),
}

impl Norm {
    fn apply(&self, x: &[f64], eps: f64) -> Vec<f64> {
        match self {
            Norm::Rms(weight) => rms_norm(x, weight, eps),
            Norm::Layer(weight, bias) => layer_norm(x, weight, bias, eps),
        }
    }
}

pub fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    x.chunks_exact(weight.len())
        .flat_map(|row| {
            let mean = dot(row, row) / row.len() as f64;
            let inverse = 1.0 / (mean + eps).sqrt();
            row.iter()
                .zip(weight)
                .map(move |(x, scale)| x * inverse * scale)
        })
        .collect()
}

fn layer_norm(x: &[f64], weight: &[f64], bias: &[f64], eps: f64) -> Vec<f64> {
    x.chunks_exact(weight.len())
        .flat_map(|row| {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let inverse = 1.0 / (variance + eps).sqrt();
            row.iter()
                .zip(weight.iter().zip(bias))
                .map(move |(x, (scale, shift))| (x - mean) * inverse * scale + shift)
        })
        .collect()
}

enum FeedForward {
    Gated {
        gate: Linear,
        up: Linear,
        down: Linear,
    },
    Plain {
        up: Linear,
        down: Linear,
        activation: fn(f64) -> f64,
    },
}

impl FeedForward {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        match self {
            FeedForward::Gated { gate, up, down } => {
                let gated: Vec<f64> = gate
                    .apply(x)
                    .into_iter()
                    .zip(up.apply(x))
                    .map(|(gate, up)| silu(gate) * up)
                    .collect();
                down.apply(&gated)
            }
            FeedForward::Plain {
                up,
                down,
                activation,
            } => {
                let inner: Vec<f64> = up.apply(x).into_iter().map(activation).collect();
                down.apply(&inner)
            }
        }
    }
}

/// How a head's elements pair under the rotary embedding.
#[derive(Clone, Copy)]
pub enum Pairing {
    /// `i` with `i + d/2`.
    Half,
    /// `2i` with `2i + 1`.
    Interleaved,
}

/// The rotary embedding of a layer's query and key heads: how they pair, and the angles of each
/// row.
pub struct Rotary<'angles> {
    pub pairing: Pairing,
    pub angles: &'angles dyn Fn(usize) -> Vec<f64>,
}

/// The angle of each frequency `base^(−2i/dims)`, `i` in `0..positions.len()`, turned by its
/// position.
pub fn angles(positions: &[usize], base: f64, dims: usize) -> Vec<f64> {
    positions
        .iter()
        .enumerate()
        .map(|(i, &position)| position as f64 * base.powf(-((2 * i) as f64) / dims as f64))
        .collect()
}

fn rotate(head: &mut [f64], angles: &[f64], pairing: Pairing) {
    let half = head.len() / 2;
    for (i, angle) in angles.iter().enumerate() {
        let (first, second) = match pairing {
            Pairing::Half => (i, i + half),
            Pairing::Interleaved => (2 * i, 2 * i + 1),
        };
        let (x, y) = (head[first], head[second]);
        head[first] = x * angle.cos() - y * angle.sin();
        head[second] = y * angle.cos() + x * angle.sin();
    }
}

/// The rows a row attends to.
#[derive(Clone, Copy)]
pub enum Mask {
    /// Itself and every row before it.
    Causal,
    /// Itself and the `n − 1` rows before it.
    Window(usize),
    /// Every row.
    Full,
}

pub struct Stack(Vec<Layer>);

impl Stack {
    /// Every layer over the rows of `x`, `rows × hidden`, in place.
    pub fn apply(&self, x: &mut [f64], dims: &Dims, rotary: Option<&Rotary>, mask: Mask) {
        for layer in &self.0 {
            layer.apply(x, dims, rotary, mask);
        }
    }
}

/// A pre-norm transformer layer: `x += out(attention(norm1(x)))`, `x += ff(norm2(x))`.
struct Layer {
    norm1: Norm,
    qkv: Linear,
    head_norms: Option<[Vec<f64>; 2]>,
    out: Linear,
    norm2: Norm,
    feed_forward: FeedForward,
}

impl Layer {
    fn apply(&self, x: &mut [f64], dims: &Dims, rotary: Option<&Rotary>, mask: Mask) {
        let qkv = self.qkv.apply(&self.norm1.apply(x, dims.eps));
        let mut heads: Vec<Vec<f64>> = qkv
            .chunks_exact(dims.head_dim)
            .map(<[f64]>::to_vec)
            .collect();
        let units = dims.heads + 2 * dims.kv_heads;
        for (index, head) in heads.iter_mut().enumerate() {
            let (row, unit) = (index / units, index % units);
            if unit >= dims.heads + dims.kv_heads {
                continue;
            }
            if let Some([query, key]) = &self.head_norms {
                let weight = if unit < dims.heads { query } else { key };
                *head = rms_norm(head, weight, dims.eps);
            }
            if let Some(rotary) = rotary {
                rotate(head, &(rotary.angles)(row), rotary.pairing);
            }
        }
        let attended = attend(&heads, dims, mask);
        add(x, &self.out.apply(&attended));
        add(x, &self.feed_forward.apply(&self.norm2.apply(x, dims.eps)));
    }
}

/// Scaled dot-product attention over `heads`, each row's query, key and value heads in turn;
/// query heads share key and value heads in equal groups.
fn attend(heads: &[Vec<f64>], dims: &Dims, mask: Mask) -> Vec<f64> {
    let units = dims.heads + 2 * dims.kv_heads;
    let rows = heads.len() / units;
    let group = dims.heads / dims.kv_heads;
    let key = |row: usize, head: usize| &heads[row * units + dims.heads + head / group];
    let value =
        |row: usize, head: usize| &heads[row * units + dims.heads + dims.kv_heads + head / group];
    let mut out = Vec::with_capacity(rows * dims.width());
    for row in 0..rows {
        let seen = match mask {
            Mask::Causal => 0..row + 1,
            Mask::Window(n) => (row + 1).saturating_sub(n)..row + 1,
            Mask::Full => 0..rows,
        };
        for head in 0..dims.heads {
            let query = &heads[row * units + head];
            let scores: Vec<f64> = seen
                .clone()
                .map(|j| dot(query, key(j, head)) / (dims.head_dim as f64).sqrt())
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut mixed = vec![0.0; dims.head_dim];
            for (j, weight) in seen.clone().zip(&weights) {
                for (mix, x) in mixed.iter_mut().zip(value(j, head)) {
                    *mix += weight / total * x;
                }
            }
            out.extend(mixed);
        }
    }
    out
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

fn add(x: &mut [f64], addend: &[f64]) {
    for (x, y) in x.iter_mut().zip(addend) {
        *x += y;
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `x · Φ(x)`, the exact GELU.
pub fn gelu_erf(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// GELU's tanh approximation.
pub fn gelu_tanh(x: f64) -> f64 {
    let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}

/// CLIP's `x · σ(1.702 x)`.
pub fn quick_gelu(x: f64) -> f64 {
    x / (1.0 + (-1.702 * x).exp())
}

/// `erf` by its Maclaurin series: within 1e-12 for `|x| ≤ 4`, beyond the tests' activations.
fn erf(x: f64) -> f64 {
    const TERMS: usize = 80;
    let mut power = x;
    let mut sum = x;
    for n in 1..TERMS {
        power *= -x * x / n as f64;
        sum += power / (2 * n + 1) as f64;
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

pub fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

/// `‖got − want‖ / ‖want‖`.
pub fn relative_error(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "as many values as the reference");
    let error: f64 = got
        .iter()
        .zip(want)
        .map(|(&got, want)| (f64::from(got) - want).powi(2))
        .sum();
    (error / dot(want, want)).sqrt()
}

/// A hybrid stack's delta rule: its heads and the taps of its convolution.
#[derive(Clone, Copy)]
pub struct DeltaDims {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
}

impl DeltaDims {
    fn keys(&self) -> usize {
        self.key_heads * self.key_dim
    }

    fn values(&self) -> usize {
        self.value_heads * self.value_dim
    }

    fn channels(&self) -> usize {
        2 * self.keys() + self.values()
    }
}

/// A hybrid stack's dimensions: its attention's (`dims`, `rotary_dim` elements of a head
/// rotated), its delta rule's, and which layers attend.
#[derive(Clone, Copy)]
pub struct HybridDims {
    pub dims: Dims,
    pub rotary_dim: usize,
    pub delta: DeltaDims,
    pub attention_interval: usize,
}

impl Weights {
    /// `LAYERS` hybrid layers with `blocks::hybrid`'s names, the `i`th under `prefix.i`: the last
    /// of every `attention_interval` with gated attention, the others with the delta rule.
    pub fn hybrid(&mut self, prefix: &str, shape: &HybridDims) -> Hybrid {
        let Dims {
            hidden,
            feed_forward,
            ..
        } = shape.dims;
        let layers = (0..LAYERS).map(|i| {
            let name = |part: &str| format!("{prefix}.{i}.{part}");
            let norm1 = self.tensor(&name("input_norm"), &[hidden], 1.0, 0.2);
            let mixer = if (i + 1) % shape.attention_interval == 0 {
                self.gated_attention(&name, &shape.dims)
            } else {
                self.delta_rule(&name, hidden, &shape.delta)
            };
            let norm2 = self.tensor(&name("post_norm"), &[hidden], 1.0, 0.2);
            let gate = self.linear(&name("gate"), [feed_forward, hidden], Bias::Without);
            let up = self.linear(&name("up"), [feed_forward, hidden], Bias::Without);
            let down = self.linear(&name("down"), [hidden, feed_forward], Bias::Without);
            HybridLayer {
                norm1,
                mixer,
                norm2,
                feed_forward: FeedForward::Gated { gate, up, down },
            }
        });
        Hybrid(layers.collect())
    }

    fn gated_attention(&mut self, name: &dyn Fn(&str) -> String, dims: &Dims) -> Mixer {
        let Dims {
            hidden,
            heads,
            kv_heads,
            head_dim,
            ..
        } = *dims;
        let columns = (2 * heads + 2 * kv_heads) * head_dim;
        Mixer::Attention {
            qkv: self.linear(&name("qkv"), [columns, hidden], Bias::Without),
            norms: [
                self.tensor(&name("q_norm"), &[head_dim], 1.0, 0.2),
                self.tensor(&name("k_norm"), &[head_dim], 1.0, 0.2),
            ],
            out: self.linear(&name("o"), [hidden, dims.width()], Bias::Without),
        }
    }

    fn delta_rule(
        &mut self,
        name: &dyn Fn(&str) -> String,
        hidden: usize,
        delta: &DeltaDims,
    ) -> Mixer {
        let columns = delta.channels() + delta.values() + 2 * delta.value_heads;
        Mixer::Delta(DeltaRule {
            input: self.linear(&name("input"), [columns, hidden], Bias::Without),
            taps: self.tensor(&name("conv"), &[delta.channels(), delta.kernel], 0.0, 0.5),
            a_log: self.tensor(&name("a_log"), &[delta.value_heads], 0.0, 1.0),
            dt_bias: self.tensor(&name("dt_bias"), &[delta.value_heads], 0.5, 0.5),
            norm: self.tensor(&name("norm"), &[delta.value_dim], 1.0, 0.2),
            out: self.linear(&name("out"), [hidden, delta.values()], Bias::Without),
        })
    }
}

pub struct Hybrid(Vec<HybridLayer>);

impl Hybrid {
    /// Every layer over the rows of `x`, `rows × hidden`, in place, the rotated part of row `r`'s
    /// query and key heads turned by `angles(r)`.
    pub fn apply(&self, x: &mut [f64], shape: &HybridDims, angles: &dyn Fn(usize) -> Vec<f64>) {
        let eps = shape.dims.eps;
        for layer in &self.0 {
            let normed = rms_norm(x, &layer.norm1, eps);
            let mixed = match &layer.mixer {
                Mixer::Attention { qkv, norms, out } => {
                    let attended = gated_attention(&qkv.apply(&normed), norms, shape, angles);
                    out.apply(&attended)
                }
                Mixer::Delta(rule) => rule.out.apply(&rule.apply(&normed, &shape.delta, eps)),
            };
            add(x, &mixed);
            add(
                x,
                &layer.feed_forward.apply(&rms_norm(x, &layer.norm2, eps)),
            );
        }
    }
}

struct HybridLayer {
    norm1: Vec<f64>,
    mixer: Mixer,
    norm2: Vec<f64>,
    feed_forward: FeedForward,
}

enum Mixer {
    Attention {
        qkv: Linear,
        norms: [Vec<f64>; 2],
        out: Linear,
    },
    Delta(DeltaRule),
}

/// Causal attention of the rows' query heads (each row's units: query heads, their gates, key
/// heads, value heads), each output times the sigmoid of its gate.
fn gated_attention(
    qkv: &[f64],
    [query_norm, key_norm]: &[Vec<f64>; 2],
    shape: &HybridDims,
    angles: &dyn Fn(usize) -> Vec<f64>,
) -> Vec<f64> {
    let dims = &shape.dims;
    let units = 2 * dims.heads + 2 * dims.kv_heads;
    let mut heads = Vec::new();
    let mut gates = Vec::new();
    for (row, units) in qkv.chunks_exact(units * dims.head_dim).enumerate() {
        for (unit, head) in units.chunks_exact(dims.head_dim).enumerate() {
            let norm = match unit {
                unit if unit < dims.heads => Some(query_norm),
                unit if unit < 2 * dims.heads => {
                    gates.extend(head.iter().map(|&gate| 1.0 / (1.0 + (-gate).exp())));
                    continue;
                }
                unit if unit < 2 * dims.heads + dims.kv_heads => Some(key_norm),
                _ => None,
            };
            let mut head = head.to_vec();
            if let Some(norm) = norm {
                head = rms_norm(&head, norm, dims.eps);
                rotate(&mut head[..shape.rotary_dim], &angles(row), Pairing::Half);
            }
            heads.push(head);
        }
    }
    let mut attended = attend(&heads, dims, Mask::Causal);
    for (value, gate) in attended.iter_mut().zip(gates) {
        *value *= gate;
    }
    attended
}

struct DeltaRule {
    input: Linear,
    taps: Vec<f64>,
    a_log: Vec<f64>,
    dt_bias: Vec<f64>,
    norm: Vec<f64>,
    out: Linear,
}

impl DeltaRule {
    /// The gated delta rule over the rows of `normed`, from an empty memory: each value head's
    /// output, gated, side by side.
    fn apply(&self, normed: &[f64], delta: &DeltaDims, eps: f64) -> Vec<f64> {
        let projected = self.input.apply(normed);
        let columns = self.input.weight.len() / self.input.inputs;
        let rows: Vec<&[f64]> = projected.chunks_exact(columns).collect();
        let channels = delta.channels();
        let convolved: Vec<Vec<f64>> = (0..rows.len())
            .map(|row| {
                (0..channels)
                    .map(|channel| {
                        let taps = &self.taps[channel * delta.kernel..][..delta.kernel];
                        let sum = taps.iter().enumerate().map(|(tap, weight)| {
                            let back = delta.kernel - 1 - tap;
                            row.checked_sub(back)
                                .map_or(0.0, |at| weight * rows[at][channel])
                        });
                        silu(sum.sum())
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![0.0; rows.len() * delta.values()];
        for head in 0..delta.value_heads {
            let key_head = head / (delta.value_heads / delta.key_heads);
            let mut memory = vec![vec![0.0; delta.key_dim]; delta.value_dim];
            for (row, qkv) in convolved.iter().enumerate() {
                let unit =
                    |offset: usize, head: usize, dim: usize| &qkv[offset + head * dim..][..dim];
                let scale = 1.0 / (delta.key_dim as f64).sqrt();
                let query = l2_normalized(unit(0, key_head, delta.key_dim), scale);
                let key = l2_normalized(unit(delta.keys(), key_head, delta.key_dim), 1.0);
                let value = unit(2 * delta.keys(), head, delta.value_dim);
                let gates = &rows[row][channels..];
                let correction = 1.0 / (1.0 + (-gates[delta.values() + head]).exp());
                let rate = (1.0
                    + (gates[delta.values() + delta.value_heads + head] + self.dt_bias[head])
                        .exp())
                .ln();
                let decay = (-self.a_log[head].exp() * rate).exp();
                let reading: Vec<f64> = memory
                    .iter_mut()
                    .zip(value)
                    .map(|(remembered, value)| {
                        remember(remembered, &key, value, decay, correction);
                        dot(remembered, &query)
                    })
                    .collect();
                let z = &gates[head * delta.value_dim..][..delta.value_dim];
                let normed = rms_norm(&reading, &self.norm, eps);
                let at = row * delta.values() + head * delta.value_dim;
                for ((out, normed), z) in out[at..].iter_mut().zip(normed).zip(z) {
                    *out = normed * silu(*z);
                }
            }
        }
        out
    }
}

/// One value's row of a head's memory decayed, then corrected toward `value` by the error of
/// its recall of `key`.
fn remember(remembered: &mut [f64], key: &[f64], value: &f64, decay: f64, correction: f64) {
    remembered.iter_mut().for_each(|weight| *weight *= decay);
    let error = (value - dot(remembered, key)) * correction;
    for (weight, key) in remembered.iter_mut().zip(key) {
        *weight += error * key;
    }
}

/// `x / ‖x‖ · scale`, with FLA's epsilon of 1e-6 under the root.
fn l2_normalized(x: &[f64], scale: f64) -> Vec<f64> {
    let inverse = 1.0 / (dot(x, x) + 1e-6).sqrt();
    x.iter().map(|value| value * inverse * scale).collect()
}
