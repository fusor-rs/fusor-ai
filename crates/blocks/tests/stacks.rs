//! Every stack of transformer layers, loaded from a bundle and run on the team, against an f64
//! reference of its architecture: with each weight format, the error stays at its noise floor.

mod support;

use std::ops::Range;

use blocks::dit::{self, Dit, Sequences};
use blocks::hybrid::{self, AttentionShape, DeltaShape, Gate, HeadNorm, Hybrid};
use blocks::interleaved::{self, Activation, Interleaved, Mixer, Norms, Window};
use blocks::lm::{self, Embedding, Lm, Positions, Reader, Rows};
use blocks::rotary::{Position, Rotation};
use blocks::transformer::{self, Architecture, KvCache, Span, Transformer};
use blocks::vit::interleaved::{self as tower, Tower};
use blocks::vit::{self, Grid, Vit};
use bundle::Load;
use kernels::DType;
use kernels::ops::Exponentials;
use team::Team;

use support::{
    Bias, DeltaDims, Dims, HeadNorms, HybridDims, Kind, LAYERS, Linear, Llama, Mask, NormKind,
    Pairing, Plain, Rotary, Weights, angles, gelu_erf, gelu_tanh, quick_gelu, relative_error,
    rms_norm, silu, widen,
};

const EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 10_000.0;
/// Grouped-query attention: two query heads per key and value head.
const DECODER: Dims = Dims {
    hidden: 64,
    heads: 4,
    kv_heads: 2,
    head_dim: 16,
    feed_forward: 128,
    eps: EPS as f64,
};
const ENCODER: Dims = Dims {
    kv_heads: 4,
    ..DECODER
};
const ROWS: usize = 8;
/// Rows per call of a decoder: the rows arrive in calls of 3, 3 and 2.
const CALL: usize = 3;
const VOCABULARY: usize = 64;
const VIT_WEIGHTS: DType = DType::Q7b32;
const DIT_WEIGHTS: DType = DType::F32;

/// The error a weight format leaves, with room: f32 weights agree with the reference to
/// rounding, 7-bit ones (and the int8 inputs they take) to about one percent.
fn tolerance(weights: DType) -> f64 {
    match weights {
        DType::F32 => 1e-6,
        _ => 3e-2,
    }
}

fn transformer_shape(architecture: Architecture, rotation: Rotation) -> transformer::Shape {
    transformer::Shape {
        architecture,
        hidden: DECODER.hidden,
        layers: LAYERS,
        heads: DECODER.heads,
        kv_heads: DECODER.kv_heads,
        head_dim: DECODER.head_dim,
        feed_forward: DECODER.feed_forward,
        rope_base: ROPE_BASE,
        rotation,
        norm_eps: EPS,
    }
}

/// The angles of a decoder's head at `position`, its frequencies in `sections` turned by the
/// temporal, height and width positions.
fn decoder_angles(position: Position, sections: [usize; 3]) -> Vec<f64> {
    let [temporal, height, width] = sections;
    let positions = [
        vec![position.temporal; temporal],
        vec![position.height; height],
        vec![position.width; width],
    ]
    .concat();
    angles(&positions, ROPE_BASE.into(), DECODER.head_dim)
}

struct Reading {
    architecture: Architecture,
    llama: Llama,
    embedding: Embedding,
    rotation: Rotation,
    weights: DType,
}

const READINGS: [Reading; 2] = [
    Reading {
        architecture: Architecture::Qwen2,
        llama: Llama {
            qkv_bias: Bias::With,
            head_norms: HeadNorms::Without,
        },
        embedding: Embedding::Own,
        rotation: Rotation::Multimodal([2, 3, 3]),
        weights: DType::F32,
    },
    Reading {
        architecture: Architecture::Qwen3,
        llama: Llama {
            qkv_bias: Bias::Without,
            head_norms: HeadNorms::With,
        },
        embedding: Embedding::Tied,
        rotation: Rotation::Sequence,
        weights: DType::Q7b32,
    },
];

#[test]
fn language_model_reads_in_chunks_and_decodes_like_the_reference() {
    for case in &READINGS {
        let error = reading_error(case);
        let architecture = case.architecture;
        assert!(
            error < tolerance(case.weights),
            "{architecture:?}: error {error:.2e}"
        );
    }
}

/// The error of the logits after reading `ROWS` rows: all but the last as a prompt, in calls of
/// `CALL` rows, then the last.
fn reading_error(case: &Reading) -> f64 {
    let Dims { hidden, .. } = DECODER;
    let mut weights = Weights::new("lm", 1);
    let own = match case.embedding {
        Embedding::Own => Some(weights.tensor("lm.embed", &[VOCABULARY, hidden], 0.0, 1.0)),
        Embedding::Tied => None,
    };
    let head = weights.linear("lm.head", [VOCABULARY, hidden], Bias::Without);
    let stack = weights.stack("lm.layers", &DECODER, Kind::Llama(case.llama));
    let norm = weights.tensor("lm.norm", &[hidden], 1.0, 0.2);
    let shape = lm::Shape {
        stack: transformer_shape(case.architecture, case.rotation),
        vocabulary: VOCABULARY,
        embedding: case.embedding,
    };
    let lm = Lm::load(&mut weights.store(case.weights), "lm", shape)
        .expect("the bundle holds every tensor the block loads");
    let ids: Vec<u32> = (0..ROWS as u32)
        .map(|row| row * 7 % VOCABULARY as u32)
        .collect();
    let positions: Vec<Position> = (0..ROWS)
        .map(|row| match case.rotation {
            Rotation::Multimodal(_) => Position {
                temporal: row,
                height: row * 2 % 5,
                width: row * 3 % 7,
            },
            _ => Position::uniform(row),
        })
        .collect();
    let logits = read(&lm, &shape, &ids, &positions);

    let mut x: Vec<f64> = ids
        .iter()
        .flat_map(|&id| match &own {
            Some(table) => &table[id as usize * hidden..(id as usize + 1) * hidden],
            None => head.row(id as usize),
        })
        .copied()
        .collect();
    let sections = match case.rotation {
        Rotation::Multimodal(sections) => sections,
        _ => [DECODER.head_dim / 2, 0, 0],
    };
    let angles = |row: usize| decoder_angles(positions[row], sections);
    let rotary = Rotary {
        pairing: Pairing::Half,
        angles: &angles,
    };
    stack.apply(&mut x, &DECODER, Some(&rotary), Mask::Causal);
    let last = rms_norm(&x[(ROWS - 1) * hidden..], &norm, DECODER.eps);
    relative_error(&logits, &head.apply(&last))
}

/// The logits after `lm` reads `ids`; the rows turn by `positions` under multimodal rotary
/// embeddings, else by their sequence positions.
fn read(lm: &Lm, shape: &lm::Shape, ids: &[u32], positions: &[Position]) -> Vec<f32> {
    let hidden = DECODER.hidden;
    let mut embeddings = vec![0.0; ROWS * hidden];
    lm.embed(ids, &mut embeddings);
    let team = Team::global();
    let mut reader = Reader::new(shape, ROWS, CALL, team.threads());
    for rows in [0..ROWS - 1, ROWS - 1..ROWS] {
        let positions = match shape.stack.rotation {
            Rotation::Multimodal(_) => Positions::Given(&positions[rows.clone()]),
            _ => Positions::Sequential,
        };
        let rows = Rows {
            embeddings: &embeddings[rows.start * hidden..rows.end * hidden],
            first: rows.start,
            positions,
        };
        reader.read(&team, lm, &rows);
    }
    reader.prediction.logits.read().to_vec()
}

struct Stream {
    architecture: Architecture,
    plain: Plain,
    pairing: Option<Pairing>,
    mask: transformer::Mask<'static>,
    capacity: usize,
    /// Rows per call.
    call: usize,
    weights: DType,
}

const WINDOW: usize = 4;
/// Rows of the state, and where each group after it ends, for `Mask::Groups`.
const GROUPED: (usize, &[usize]) = (3, &[5, 6, ROWS]);
const STREAMS: [Stream; 4] = [
    Stream {
        architecture: Architecture::Kyutai,
        plain: Plain {
            norm: NormKind::Layer,
            bias: Bias::Without,
            activation: gelu_erf,
        },
        pairing: Some(Pairing::Interleaved),
        mask: transformer::Mask::Window(WINDOW),
        capacity: WINDOW + CALL,
        call: CALL,
        weights: DType::Q7b32,
    },
    Stream {
        architecture: Architecture::Clip,
        plain: Plain {
            norm: NormKind::Layer,
            bias: Bias::With,
            activation: quick_gelu,
        },
        pairing: None,
        mask: transformer::Mask::Causal,
        capacity: ROWS,
        call: CALL,
        weights: DType::F32,
    },
    Stream {
        architecture: Architecture::Encoder,
        plain: Plain {
            norm: NormKind::Layer,
            bias: Bias::With,
            activation: gelu_erf,
        },
        pairing: None,
        mask: transformer::Mask::Full,
        capacity: ROWS,
        call: ROWS,
        weights: DType::F32,
    },
    Stream {
        architecture: Architecture::Encoder,
        plain: Plain {
            norm: NormKind::Layer,
            bias: Bias::With,
            activation: gelu_erf,
        },
        pairing: None,
        mask: transformer::Mask::Groups {
            state: GROUPED.0,
            ends: GROUPED.1,
        },
        capacity: ROWS,
        call: ROWS,
        weights: DType::F32,
    },
];

#[test]
fn transformer_attends_over_its_cache_like_the_reference() {
    for case in &STREAMS {
        let error = stream_error(case);
        let architecture = case.architecture;
        assert!(
            error < tolerance(case.weights),
            "{architecture:?}: error {error:.2e}"
        );
    }
}

/// The error of the output of `ROWS` rows through the stack in calls of `case.call` rows.
fn stream_error(case: &Stream) -> f64 {
    let mut weights = Weights::new("transformer", 2);
    let input = weights.floats(ROWS * DECODER.hidden, 0.0, 1.0);
    let stack = weights.stack("stack.layers", &DECODER, Kind::Plain(case.plain));
    let shape = transformer_shape(case.architecture, Rotation::Sequence);
    let transformer = Transformer::load(&mut weights.store(case.weights), "stack", shape)
        .expect("the bundle holds every tensor the block loads");
    let output = stream(&transformer, shape, case, &input);

    let mut x = widen(&input);
    let angles = |row: usize| decoder_angles(Position::uniform(row), [DECODER.head_dim / 2, 0, 0]);
    let rotary = case.pairing.map(|pairing| Rotary {
        pairing,
        angles: &angles,
    });
    stack.apply(&mut x, &DECODER, rotary.as_ref(), reference_mask(case.mask));
    relative_error(&output, &x)
}

/// The reference's mask for a block's.
fn reference_mask(mask: transformer::Mask<'static>) -> Mask {
    match mask {
        transformer::Mask::Causal => Mask::Causal,
        transformer::Mask::Window(n) => Mask::Window(n),
        transformer::Mask::Full => Mask::Full,
        transformer::Mask::Groups { state, ends } => Mask::Groups { state, ends },
    }
}

fn stream(
    transformer: &Transformer,
    shape: transformer::Shape,
    case: &Stream,
    input: &[f32],
) -> Vec<f32> {
    let hidden = DECODER.hidden;
    let team = Team::global();
    let cache = KvCache::new(shape, case.capacity);
    let mut buffers = shape.buffers(case.call, team.threads());
    let mut output = Vec::new();
    for first in (0..ROWS).step_by(case.call) {
        let rows = case.call.min(ROWS - first);
        buffers.hidden.get_mut()[..rows * hidden]
            .copy_from_slice(&input[first * hidden..(first + rows) * hidden]);
        let span = Span {
            rows,
            first,
            mask: case.mask,
        };
        team.run(|ctx| transformer.forward(ctx, &cache, &buffers, span));
        output.extend_from_slice(&buffers.hidden.read()[..rows * hidden]);
    }
    output
}

#[test]
fn vision_encoder_attends_among_the_patches_like_the_reference() {
    let grid = Grid {
        rows: 3,
        columns: 4,
    };
    let Dims { hidden, heads, .. } = ENCODER;
    let mut weights = Weights::new("vit", 3);
    let input = weights.floats(grid.patches() * hidden, 0.0, 1.0);
    let plain = Plain {
        norm: NormKind::Layer,
        bias: Bias::With,
        activation: gelu_tanh,
    };
    let stack = weights.stack("vit.layers", &ENCODER, Kind::Plain(plain));
    let angles = |row: usize| grid_angles(grid, ENCODER.head_dim, row);
    let rotary = Rotary {
        pairing: Pairing::Half,
        angles: &angles,
    };
    for (rope_base, reference_rotary) in [(Some(ROPE_BASE), Some(&rotary)), (None, None)] {
        let shape = vit::Shape {
            hidden,
            layers: LAYERS,
            heads,
            feed_forward: ENCODER.feed_forward,
            rope_base,
            norm_eps: EPS,
            exponentials: Exponentials::Exact,
        };
        let vit = Vit::load(&mut weights.clone().store(VIT_WEIGHTS), "vit", shape)
            .expect("the bundle holds every tensor the block loads");

        let team = Team::global();
        let mut buffers = shape.buffers(grid.patches(), team.threads());
        buffers.hidden.get_mut().copy_from_slice(&input);
        team.run(|ctx| vit.forward(ctx, &buffers, grid));

        let mut x = widen(&input);
        stack.apply(&mut x, &ENCODER, reference_rotary, Mask::Full);
        let error = relative_error(buffers.hidden.read(), &x);
        assert!(
            error < tolerance(VIT_WEIGHTS),
            "rope base {rope_base:?}: error {error:.2e}"
        );
    }
}

/// A patch's rotary angles: the first half of a head's frequencies by its row in `grid`, the
/// second by its column.
fn grid_angles(grid: Grid, head_dim: usize, row: usize) -> Vec<f64> {
    let (quarter, base) = (head_dim / 4, ROPE_BASE.into());
    let along_rows = angles(&vec![row / grid.columns; quarter], base, 2 * quarter);
    let along_columns = angles(&vec![row % grid.columns; quarter], base, 2 * quarter);
    [along_rows, along_columns].concat()
}

const PATCH_VALUES: usize = 32;
/// Side of the conv layer's square of taps.
const KERNEL: usize = 3;

/// The vision tower's attention layers (`LAYERS` of them, grouped-query), then a conv layer.
const TOWER_MIXERS: [tower::Mixer; LAYERS + 1] = {
    use tower::Mixer::{Attention, Conv};
    [Attention, Attention, Conv]
};

/// A gated 2-D conv layer's weights, as the reference applies them.
struct ConvLayer {
    norm1: Vec<f64>,
    input: Linear,
    taps: Vec<f64>,
    out: Linear,
    norm2: Vec<f64>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl ConvLayer {
    fn new(weights: &mut Weights, prefix: &str) -> Self {
        let (hidden, inner) = (DECODER.hidden, DECODER.feed_forward);
        let name = |part: &str| format!("{prefix}.{part}");
        Self {
            norm1: weights.tensor(&name("input_norm"), &[hidden], 1.0, 0.2),
            input: weights.linear(&name("input"), [3 * hidden, hidden], Bias::Without),
            taps: weights.tensor(&name("conv"), &[hidden, KERNEL * KERNEL], 0.0, 0.5),
            out: weights.linear(&name("out"), [hidden, hidden], Bias::Without),
            norm2: weights.tensor(&name("post_norm"), &[hidden], 1.0, 0.2),
            gate: weights.linear(&name("gate"), [inner, hidden], Bias::Without),
            up: weights.linear(&name("up"), [inner, hidden], Bias::Without),
            down: weights.linear(&name("down"), [hidden, inner], Bias::Without),
        }
    }

    /// `x += out(c · conv(b · x))` over the patches of `grid`, zero past its edges, then the
    /// SiLU-gated MLP.
    fn apply(&self, x: &mut [f64], grid: Grid) {
        let (hidden, eps) = (DECODER.hidden, DECODER.eps);
        let projected = self.input.apply(&rms_norm(x, &self.norm1, eps));
        let part = |patch: usize, part: usize, channel: usize| {
            projected[patch * 3 * hidden + part * hidden + channel]
        };
        let mut mixed = vec![0.0; x.len()];
        for (at, value) in mixed.iter_mut().enumerate() {
            let (patch, channel) = (at / hidden, at % hidden);
            let (top, left) = (patch / grid.columns, patch % grid.columns);
            let mut sum = 0.0;
            for tap in 0..KERNEL * KERNEL {
                let (y, x) = (top + tap / KERNEL, left + tap % KERNEL);
                if (1..=grid.rows).contains(&y) && (1..=grid.columns).contains(&x) {
                    let source = (y - 1) * grid.columns + x - 1;
                    let gated = part(source, 0, channel) * part(source, 2, channel);
                    sum += self.taps[channel * KERNEL * KERNEL + tap] * gated;
                }
            }
            *value = part(patch, 1, channel) * sum;
        }
        for (value, added) in x.iter_mut().zip(self.out.apply(&mixed)) {
            *value += added;
        }
        let normed = rms_norm(x, &self.norm2, eps);
        let gates = self.gate.apply(&normed).into_iter();
        let silu = |value: f64| value / (1.0 + (-value).exp());
        let inner: Vec<f64> = gates
            .zip(self.up.apply(&normed))
            .map(|(gate, up)| silu(gate) * up)
            .collect();
        for (value, added) in x.iter_mut().zip(self.down.apply(&inner)) {
            *value += added;
        }
    }
}

#[test]
fn vision_tower_mixes_its_patches_like_the_reference() {
    let grid = Grid {
        rows: 3,
        columns: 4,
    };
    let Dims {
        hidden,
        heads,
        kv_heads,
        ..
    } = DECODER;
    let mut weights = Weights::new("tower", 5);
    let pixels = weights.floats(grid.patches() * PATCH_VALUES, 0.0, 1.0);
    let patch = weights.linear("tower.patch", [hidden, PATCH_VALUES], Bias::Without);
    let patch_norm = weights.tensor("tower.patch_norm", &[hidden], 1.0, 0.2);
    let llama = Llama {
        qkv_bias: Bias::Without,
        head_norms: HeadNorms::With,
    };
    let stack = weights.stack("tower.layers", &DECODER, Kind::Llama(llama));
    let conv = ConvLayer::new(&mut weights, &format!("tower.layers.{LAYERS}"));
    let shape = tower::Shape {
        hidden,
        mixers: &TOWER_MIXERS,
        heads,
        kv_heads,
        feed_forward: DECODER.feed_forward,
        kernel: KERNEL,
        rope_base: ROPE_BASE,
        norm_eps: EPS,
        patch_values: PATCH_VALUES,
        exponentials: Exponentials::Exact,
    };
    let tower = Tower::load(&mut weights.store(VIT_WEIGHTS), "tower", shape)
        .expect("the bundle holds every tensor the block loads");

    let team = Team::global();
    let buffers = shape.buffers(grid.patches(), team.threads());
    team.run(|ctx| {
        tower.embed(ctx, &buffers, &pixels);
        tower.encode(ctx, &buffers, grid);
    });

    let mut x = rms_norm(&patch.apply(&widen(&pixels)), &patch_norm, DECODER.eps);
    let angles = |row: usize| grid_angles(grid, DECODER.head_dim, row);
    let rotary = Rotary {
        pairing: Pairing::Half,
        angles: &angles,
    };
    stack.apply(&mut x, &DECODER, Some(&rotary), Mask::Full);
    conv.apply(&mut x, grid);
    let encoded = &buffers.encoder.hidden.read()[..x.len()];
    let error = relative_error(encoded, &x);
    assert!(error < tolerance(VIT_WEIGHTS), "error {error:.2e}");
}

/// A kind of DiT, with the reference's activation and rotary pairing for it.
struct DitCase {
    kind: dit::Kind,
    activation: fn(f64) -> f64,
    pairing: Option<Pairing>,
}

const DIT_CASES: [DitCase; 2] = [
    DitCase {
        kind: dit::Kind::Rotary {
            rope_base: ROPE_BASE,
        },
        activation: gelu_tanh,
        pairing: Some(Pairing::Interleaved),
    },
    DitCase {
        kind: dit::Kind::Plain,
        activation: silu,
        pairing: None,
    },
];

#[test]
fn diffusion_transformer_attends_within_each_sequence_like_the_reference() {
    let sequences = Sequences {
        rows: 12,
        length: 4,
    };
    let Dims {
        hidden,
        heads,
        head_dim,
        feed_forward,
        ..
    } = ENCODER;
    for DitCase {
        kind,
        activation,
        pairing,
    } in DIT_CASES
    {
        let mut weights = Weights::new("dit", 4);
        let input = weights.floats(sequences.rows * hidden, 0.0, 1.0);
        let plain = Plain {
            norm: NormKind::Rms,
            bias: Bias::With,
            activation,
        };
        let stack = weights.stack("dit.blocks", &ENCODER, Kind::Plain(plain));
        let shape = dit::Shape {
            hidden,
            heads,
            head_dim,
            feed_forward,
            depth: LAYERS,
            norm_eps: EPS,
            kind,
        };
        let dit = Dit::load(&mut weights.store(DIT_WEIGHTS), "dit", shape)
            .expect("the bundle holds every tensor the block loads");

        let team = Team::global();
        let mut buffers = shape.buffers(sequences.rows, team.threads());
        buffers.hidden.get_mut().copy_from_slice(&input);
        team.run(|ctx| dit.forward(ctx, &buffers, sequences));

        let mut x = widen(&input);
        let angles = |row: usize| angles(&vec![row; head_dim / 2], ROPE_BASE.into(), head_dim);
        let rotary = pairing.map(|pairing| Rotary {
            pairing,
            angles: &angles,
        });
        for sequence in x.chunks_exact_mut(sequences.length * hidden) {
            stack.apply(sequence, &ENCODER, rotary.as_ref(), Mask::Full);
        }
        let error = relative_error(buffers.hidden.read(), &x);
        assert!(
            error < tolerance(DIT_WEIGHTS),
            "{kind:?}: error {error:.2e}"
        );
    }
}

/// Rows of the prompt a refiner's padding row sees.
const REFINER_PROMPT: usize = 5;

#[test]
fn refiner_pad_row_sees_the_prompt_and_itself_like_the_reference() {
    let hidden = DECODER.hidden;
    let mut weights = Weights::new("refiner", 6);
    let input = weights.floats((REFINER_PROMPT + 1) * hidden, 0.0, 1.0);
    let llama = Llama {
        qkv_bias: Bias::Without,
        head_norms: HeadNorms::With,
    };
    let stack = weights.stack("stack.layers", &DECODER, Kind::Llama(llama));
    let shape = transformer_shape(Architecture::Qwen3, Rotation::Partial(0));
    let refiner = Transformer::load(&mut weights.store(DType::F32), "stack", shape)
        .expect("the bundle holds every tensor the block loads");

    let team = Team::global();
    let cache = KvCache::new(shape, REFINER_PROMPT + 1);
    let mut prompt = shape.buffers(REFINER_PROMPT, team.threads());
    let mut padding = shape.buffers(1, team.threads());
    let (prompt_input, padding_input) = input.split_at(REFINER_PROMPT * hidden);
    prompt.hidden.get_mut().copy_from_slice(prompt_input);
    padding.hidden.get_mut().copy_from_slice(padding_input);
    let prompt_span = Span {
        rows: REFINER_PROMPT,
        first: 0,
        mask: transformer::Mask::Full,
    };
    let padding_span = Span {
        rows: 1,
        first: REFINER_PROMPT,
        mask: transformer::Mask::Causal,
    };
    team.run(|ctx| {
        refiner.forward_layers(ctx, &cache, &prompt, (prompt_span, 0..1));
        refiner.forward_layers(ctx, &cache, &prompt, (prompt_span, 1..LAYERS));
        refiner.forward(ctx, &cache, &padding, padding_span);
    });

    let mut x = widen(&input);
    stack.apply(&mut x, &DECODER, None, Mask::Prefix(REFINER_PROMPT));
    let output = [prompt.hidden.read(), padding.hidden.read()].concat();
    let error = relative_error(&output, &x);
    assert!(error < tolerance(DType::F32), "error {error:.2e}");
}

/// Two hybrid layers: the delta rule, then gated attention over half of each head rotated.
const HYBRID: HybridDims = HybridDims {
    dims: DECODER,
    rotary_dim: DECODER.head_dim / 2,
    delta: DeltaDims {
        key_heads: 2,
        value_heads: 8,
        key_dim: 16,
        value_dim: 8,
        kernel: 4,
    },
    attention_interval: 2,
};
const HYBRID_SECTIONS: [usize; 3] = [2, 1, 1];

/// How each call is read: as it is; first on other rows (the state saved before and restored
/// after); after the whole sequence reversed (the prefix before the call kept with its attention
/// rows and put back); or one layer at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Detour {
    None,
    EveryCall,
    Prefix,
    Layers,
}

/// Each mask the hybrid stack reads under, the rows per call and the detours: a sequence read with
/// every row seeing every other is read in one call.
const HYBRID_CALLS: [(transformer::Mask, usize, Detour); 5] = [
    (transformer::Mask::Causal, CALL, Detour::None),
    (transformer::Mask::Causal, CALL, Detour::EveryCall),
    (transformer::Mask::Causal, CALL, Detour::Prefix),
    (transformer::Mask::Full, ROWS, Detour::None),
    (transformer::Mask::Causal, CALL, Detour::Layers),
];

#[test]
fn hybrid_stack_carries_its_state_across_calls_like_the_reference() {
    for reading in HYBRID_CALLS {
        let error = hybrid_error(reading);
        assert!(
            error < tolerance(DType::F32),
            "{reading:?}: error {error:.2e}"
        );
    }
}

/// The error of the output of `ROWS` rows through the hybrid stack in calls of `call` rows.
fn hybrid_error(reading: (transformer::Mask<'static>, usize, Detour)) -> f64 {
    let mut weights = Weights::new("hybrid", 5);
    let input = weights.floats(ROWS * DECODER.hidden, 0.0, 1.0);
    let stack = weights.hybrid("hybrid.layers", &HYBRID);
    let shape = hybrid_shape();
    let hybrid = Hybrid::load(&mut weights.store(DType::F32), "hybrid", shape)
        .expect("the bundle holds every tensor the block loads");
    let positions: Vec<Position> = (0..ROWS)
        .map(|row| Position {
            temporal: row,
            height: row * 2 % 5,
            width: row * 3 % 7,
        })
        .collect();
    let output = read_hybrid(&hybrid, &shape, &input, &positions, reading);

    let mut x = widen(&input);
    let angles = |row: usize| {
        let Position {
            temporal,
            height,
            width,
        } = positions[row];
        let axes = [temporal, height, width, temporal];
        angles(&axes, ROPE_BASE.into(), HYBRID.rotary_dim)
    };
    stack.apply(&mut x, &HYBRID, &angles, reference_mask(reading.0));
    relative_error(&output, &x)
}

fn hybrid_shape() -> hybrid::Shape {
    let DeltaDims {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel,
    } = HYBRID.delta;
    hybrid::Shape {
        hidden: DECODER.hidden,
        layers: LAYERS,
        attention_interval: HYBRID.attention_interval,
        attention: AttentionShape {
            heads: DECODER.heads,
            kv_heads: DECODER.kv_heads,
            head_dim: DECODER.head_dim,
            rotary_dim: HYBRID.rotary_dim,
            rope_base: ROPE_BASE,
            rotation: Rotation::MultimodalInterleaved(HYBRID_SECTIONS),
            head_norm: HeadNorm::Rms,
            gate: Gate::Elementwise,
        },
        delta: DeltaShape {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel,
        },
        feed_forward: DECODER.feed_forward,
        norm_eps: EPS,
    }
}

/// `shape`'s layers with `blocks::interleaved`'s names under `prefix`.
fn write_interleaved(weights: &mut Weights, prefix: &str, shape: &interleaved::Shape) {
    let (hidden, heads) = (DECODER.hidden, DECODER.heads);
    let head_dim = DECODER.head_dim;
    let gate_heads = match shape.attention.gate {
        Gate::Headwise => heads,
        _ => 0,
    };
    for (layer, mixer) in shape.mixers.iter().enumerate() {
        let prefix = format!("{prefix}.layers.{layer}");
        let norms = [
            "input_norm",
            "post_norm",
            "mixer_out_norm",
            "feed_forward_out_norm",
        ];
        for norm in norms {
            weights.tensor(&format!("{prefix}.{norm}"), &[hidden], 1.0, 0.1);
        }
        let matrices = match mixer {
            Mixer::Conv => vec![
                ("input.weight", 3 * hidden, hidden),
                ("out.weight", hidden, hidden),
            ],
            Mixer::Full | Mixer::Window => {
                let qkv = (heads + 2 * DECODER.kv_heads) * head_dim + gate_heads;
                vec![
                    ("qkv.weight", qkv, hidden),
                    ("o.weight", hidden, heads * head_dim),
                ]
            }
        };
        let feed_forward = DECODER.feed_forward;
        let inner = [
            ("gate.weight", feed_forward, hidden),
            ("up.weight", feed_forward, hidden),
        ];
        let down = ("down.weight", hidden, feed_forward);
        for (name, outputs, inputs) in matrices.into_iter().chain(inner).chain([down]) {
            weights.tensor(&format!("{prefix}.{name}"), &[outputs, inputs], 0.0, 0.2);
        }
        match mixer {
            Mixer::Conv => {
                weights.tensor(
                    &format!("{prefix}.conv"),
                    &[hidden, shape.conv_kernel],
                    0.0,
                    0.5,
                );
            }
            Mixer::Full | Mixer::Window => {
                for norm in ["q_norm", "k_norm"] {
                    weights.tensor(&format!("{prefix}.{norm}"), &[head_dim], 1.0, 0.1);
                }
            }
        }
    }
}

/// Two layers of a bidirectional encoder in the shape of T5Gemma2's: attention between the rows
/// of a call, windowed under `mixers` or not.
const fn encoder_shape(mixers: &'static [Mixer]) -> interleaved::Shape {
    interleaved::Shape {
        mixers,
        attention: AttentionShape {
            gate: Gate::None,
            rotation: Rotation::Sequence,
            ..LFM.attention
        },
        window: Window {
            rows: 2 * ROWS,
            ..LFM.window
        },
        conv_kernel: 0,
        ..LFM
    }
}

#[test]
fn window_layers_of_an_encoder_see_every_row() {
    let windowed = encoder_shape(&[Mixer::Window; 2]);
    let full = encoder_shape(&[Mixer::Full; 2]);
    let mut weights = Weights::new("interleaved-encoder", 13);
    let input = weights.floats(ROWS * DECODER.hidden, 0.0, 1.0);
    write_interleaved(&mut weights, "encoder", &windowed);
    let outputs = [windowed, full].map(|shape| {
        let mut store = weights.clone().store(DType::F32);
        let stack = Interleaved::load(&mut store, "encoder", shape)
            .expect("the bundle holds every tensor the block loads");
        read_whole(&stack, &shape, &input)
    });
    let error = relative_error(&outputs[0], &widen(&outputs[1]));
    assert!(error < tolerance(DType::F32), "{error:.2e}");
}

/// `input`'s `ROWS` rows through the stack in one call under `Mask::Full`.
fn read_whole(stack: &Interleaved, shape: &interleaved::Shape, input: &[f32]) -> Vec<f32> {
    let team = Team::global();
    let state = interleaved::State::encoder(shape, ROWS);
    let mut buffers = shape.buffers(ROWS, team.threads());
    let positions: Vec<Position> = (0..ROWS).map(Position::uniform).collect();
    buffers.hidden.get_mut()[..input.len()].copy_from_slice(input);
    let call = hybrid::Rows {
        first: 0,
        positions: &positions,
        mask: transformer::Mask::Full,
    };
    team.run(|ctx| stack.forward(ctx, &state, &buffers, &call));
    buffers.hidden.read()[..input.len()].to_vec()
}

/// The output of `input`'s rows through the stack in calls of `call` rows under `mask`, each call
/// read first on the rows of the input reversed when it takes a detour.
fn read_hybrid(
    hybrid: &Hybrid,
    shape: &hybrid::Shape,
    input: &[f32],
    positions: &[Position],
    (mask, call, detour): (transformer::Mask, usize, Detour),
) -> Vec<f32> {
    let hidden = DECODER.hidden;
    let team = Team::global();
    let mut state = hybrid::State::new(shape, ROWS);
    let mut saved = hybrid::Saved::new(shape);
    let mut buffers = shape.buffers(call, team.threads());
    let reversed: Vec<f32> = input.rchunks(hidden).flatten().copied().collect();
    let calls: Vec<Range<usize>> = (0..ROWS)
        .step_by(call)
        .map(|first| first..ROWS.min(first + call))
        .collect();
    let mut read = |state: &hybrid::State, from: &[f32], rows: &Range<usize>| {
        buffers.stack.hidden.get_mut()[..rows.len() * hidden]
            .copy_from_slice(&from[rows.start * hidden..rows.end * hidden]);
        let rows_call = hybrid::Rows {
            first: rows.start,
            positions: &positions[rows.clone()],
            mask,
        };
        match detour {
            Detour::Layers => {
                for layer in 0..shape.layers {
                    team.run(|ctx| {
                        let layers = layer..layer + 1;
                        hybrid.forward_layers(ctx, state, &buffers, (&rows_call, layers));
                    });
                }
            }
            _ => team.run(|ctx| hybrid.forward(ctx, state, &buffers, &rows_call)),
        }
        buffers.stack.hidden.read()[..rows.len() * hidden].to_vec()
    };
    let mut output = Vec::new();
    for rows in &calls {
        match detour {
            Detour::None | Detour::Layers => {}
            Detour::EveryCall => {
                state.save(&mut saved);
                read(&state, &reversed, rows);
                state.restore(&saved);
            }
            Detour::Prefix => {
                let prefix = state.keep_prefix(shape, rows.start);
                state.clear();
                for other in &calls {
                    read(&state, &reversed, other);
                }
                state.restore_prefix(&prefix);
            }
        }
        output.extend(read(&state, input, rows));
    }
    output
}

/// LFM3's layers in small: gated short convolutions between full and window attention, every
/// sublayer's output normed (a sandwich).
const LFM: interleaved::Shape = interleaved::Shape {
    hidden: DECODER.hidden,
    mixers: &[Mixer::Conv, Mixer::Full, Mixer::Conv, Mixer::Window],
    attention: AttentionShape {
        heads: DECODER.heads,
        kv_heads: DECODER.kv_heads,
        head_dim: DECODER.head_dim,
        rotary_dim: DECODER.head_dim,
        rope_base: ROPE_BASE,
        rotation: Rotation::Partial(DECODER.head_dim / 4),
        head_norm: HeadNorm::Rms,
        gate: Gate::Headwise,
    },
    window: Window {
        rows: 2,
        rotary_dim: DECODER.head_dim,
        rope_base: ROPE_BASE,
        rotation: Rotation::Sequence,
    },
    conv_kernel: 3,
    feed_forward: DECODER.feed_forward,
    activation: Activation::Silu,
    norms: Norms::Sandwich,
    norm_eps: EPS,
};

#[test]
fn interleaved_stack_reads_on_across_calls_as_in_one() {
    let mut weights = Weights::new("interleaved", 11);
    let input = weights.floats(ROWS * DECODER.hidden, 0.0, 1.0);
    write_interleaved(&mut weights, "lfm", &LFM);
    let stack = Interleaved::load(&mut weights.store(DType::F32), "lfm", LFM)
        .expect("the bundle holds every tensor the block loads");
    let whole = widen(&read_interleaved(&stack, &input, ROWS, Detour::None));
    for detour in [Detour::None, Detour::Prefix] {
        let error = relative_error(&read_interleaved(&stack, &input, CALL, detour), &whole);
        assert!(error < tolerance(DType::F32), "{detour:?}: {error:.2e}");
    }
}

/// The output of `input`'s rows through the stack in calls of `call` rows, under `detour`.
fn read_interleaved(stack: &Interleaved, input: &[f32], call: usize, detour: Detour) -> Vec<f32> {
    let hidden = DECODER.hidden;
    let team = Team::global();
    let mut state = interleaved::State::new(&LFM, ROWS, call);
    let mut buffers = LFM.buffers(call, team.threads());
    let positions: Vec<Position> = (0..ROWS).map(Position::uniform).collect();
    let reversed: Vec<f32> = input.rchunks(hidden).flatten().copied().collect();
    let mut read = |state: &interleaved::State, from: &[f32], rows: Range<usize>| {
        buffers.hidden.get_mut()[..rows.len() * hidden]
            .copy_from_slice(&from[rows.start * hidden..rows.end * hidden]);
        let rows_call = hybrid::Rows {
            first: rows.start,
            positions: &positions[rows.clone()],
            mask: transformer::Mask::Causal,
        };
        team.run(|ctx| stack.forward(ctx, state, &buffers, &rows_call));
        buffers.hidden.read()[..rows.len() * hidden].to_vec()
    };
    let mut output = Vec::new();
    for first in (0..ROWS).step_by(call) {
        if detour == Detour::Prefix && first > 0 {
            let prefix = state.keep_prefix(first);
            state.clear();
            for other in (0..ROWS).step_by(call) {
                read(&state, &reversed, other..ROWS.min(other + call));
            }
            state.restore_prefix(&prefix);
        }
        output.extend(read(&state, input, first..ROWS.min(first + call)));
    }
    output
}
