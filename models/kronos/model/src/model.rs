//! The model assembled from a loaded bundle: the tokenizer under `tokenizer`, and the predictor's
//! embeddings `embed`, temporal embeddings `time`, stack `llm`, heads `head` and dependency layer
//! `dependency`. The compiler (`kronos-compile`) writes exactly these names.

use blocks::lm::Decoder;
use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture};
use bundle::{Load, Scope, Store};
use kernels::{Linear, Table, Vector};

use crate::Error;
use crate::tokenizer::Tokenizer;

/// Bits of each half of a token: the coarse `s1` and the fine `s2`.
pub const BITS: usize = 10;
/// Values of each half of a token.
pub const VOCABULARY: usize = 1 << BITS;
/// The most rows the predictor reads at once; a longer sequence slides a window this long.
pub const MAX_CONTEXT: usize = 512;
/// A row's values: open, high, low, close, volume and amount.
pub const FEATURES: usize = 6;
pub const HIDDEN: usize = 832;
pub const NORM_EPS: f32 = 1e-5;
/// Heads of the dependency layer's attention.
pub const DEPENDENCY_HEADS: usize = 4;

/// Kronos-base's stack: 12 Qwen2 layers of 16 heads of 52, with biases on every attention
/// projection.
pub const STACK: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen2,
    hidden: HIDDEN,
    layers: 12,
    heads: 16,
    kv_heads: 16,
    head_dim: 52,
    feed_forward: 2048,
    rope_base: 10_000.0,
    rotation: Rotation::Sequence,
    norm_eps: NORM_EPS,
};

/// A temporal embedding's table, and how many values its field takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Minute,
    Hour,
    Weekday,
    Day,
    Month,
}

impl Field {
    /// In a stamp's order.
    pub const ALL: [Field; 5] = [
        Field::Minute,
        Field::Hour,
        Field::Weekday,
        Field::Day,
        Field::Month,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Field::Minute => "minute",
            Field::Hour => "hour",
            Field::Weekday => "weekday",
            Field::Day => "day",
            Field::Month => "month",
        }
    }

    pub fn values(self) -> usize {
        match self {
            Field::Minute => 60,
            Field::Hour => 24,
            Field::Weekday => 7,
            Field::Day => 32,
            Field::Month => 13,
        }
    }
}

/// A row's time: its minute, hour, weekday (Monday 0), day of the month and month.
pub type Stamp = [u8; 5];

/// A token's input embedding: each half's row scaled by `√HIDDEN`, both fused by one linear
/// layer.
pub(crate) struct Embedding {
    pub(crate) s1: Table,
    pub(crate) s2: Table,
    pub(crate) fusion: Linear,
}

/// The learned temporal embeddings, one table per field of a stamp.
pub(crate) struct Time {
    tables: Vec<Table>,
}

impl Time {
    /// The sum of the stamp's rows, in the reference's order: hour, weekday, day, month, minute.
    pub(crate) fn embed(&self, stamp: &Stamp, out: &mut [f32]) {
        const ORDER: [Field; 5] = [
            Field::Hour,
            Field::Weekday,
            Field::Day,
            Field::Month,
            Field::Minute,
        ];
        let mut row = [0.0; HIDDEN];
        out.fill(0.0);
        for field in ORDER {
            let table = self.tables[field as usize].matrix();
            table.dequantize_row(usize::from(stamp[field as usize]), &mut row);
            for (sum, value) in out.iter_mut().zip(row) {
                *sum += value;
            }
        }
    }
}

/// The dependency layer: one query, the sampled `s1`'s embedding, attends over the stack's
/// final-normed rows without rotary turns (the reference turns the query and every key by
/// position 0); the last row plus its output, normed, feeds the `s2` head.
pub(crate) struct Dependency {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) out: Linear,
    pub(crate) norm: Vector,
}

pub struct Model {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) embedding: Embedding,
    pub(crate) time: Time,
    pub(crate) stack: Decoder,
    pub(crate) head_s1: Linear,
    pub(crate) head_s2: Linear,
    pub(crate) dependency: Dependency,
}

impl Model {
    pub(crate) fn load(store: &mut Store) -> Result<Self, Error> {
        let tokenizer = Tokenizer::load(store, "tokenizer", ())?;
        let stack = Decoder::load(store, "llm", STACK)?;
        let square = [HIDDEN, HIDDEN];
        let mut scope = Scope::new(store, "embed");
        let embedding = Embedding {
            s1: scope.load("s1", [VOCABULARY, HIDDEN])?,
            s2: scope.load("s2", [VOCABULARY, HIDDEN])?,
            fusion: scope.load("fusion", [HIDDEN, 2 * HIDDEN])?,
        };
        let mut scope = Scope::new(store, "time");
        let tables = Field::ALL
            .iter()
            .map(|field| scope.load(field.name(), [field.values(), HIDDEN]))
            .collect::<Result<_, _>>()?;
        let mut scope = Scope::new(store, "head");
        let head_s1 = scope.load("s1", [VOCABULARY, HIDDEN])?;
        let head_s2 = scope.load("s2", [VOCABULARY, HIDDEN])?;
        let mut scope = Scope::new(store, "dependency");
        let dependency = Dependency {
            query: scope.load("query", square)?,
            key: scope.load("key", square)?,
            value: scope.load("value", square)?,
            out: scope.load("out", square)?,
            norm: scope.load("norm", HIDDEN)?,
        };
        Ok(Self {
            tokenizer,
            embedding,
            time: Time { tables },
            stack,
            head_s1,
            head_s2,
            dependency,
        })
    }
}
