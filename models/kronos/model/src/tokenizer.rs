//! Kronos-Tokenizer-base: rows of six normalized values to tokens and back.
//!
//! Encoding embeds each row, runs the causal encoder, projects each row to a code of
//! `2 · BITS` values and keeps their signs: the first `BITS`, least significant first, are the
//! coarse half `s1`, the rest the fine half `s2` (binary spherical quantization; its
//! normalization keeps the signs). Decoding turns a token back into its code, `±1/√(2 · BITS)`,
//! and runs the causal decoder and the head. Neither stack ends in a norm.
//!
//! Tensors under a prefix `p`: `p.embed`, the stacks `p.encoder.layers.{i}…` and
//! `p.decoder.layers.{i}…`, `p.quantize`, `p.dequantize` and `p.head`.

use blocks::rotary::Rotation;
use blocks::transformer::{self, Architecture, KvCache, Mask, Span, Transformer};
use bundle::{Load, Scope, Store};
use kernels::{Linear, QuantizedRows};
use serde::Serialize;
use team::Team;

use crate::model::{BITS, FEATURES, MAX_CONTEXT, NORM_EPS};

/// Values of a token's code.
pub const CODE: usize = 2 * BITS;

pub const SHAPE: transformer::Shape = transformer::Shape {
    architecture: Architecture::Qwen2,
    hidden: 256,
    layers: 3,
    heads: 4,
    kv_heads: 4,
    head_dim: 64,
    feed_forward: 512,
    rope_base: 10_000.0,
    rotation: Rotation::Sequence,
    norm_eps: NORM_EPS,
};

const WIDTH: usize = SHAPE.hidden;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Token {
    pub s1: u16,
    pub s2: u16,
}

impl Token {
    /// The token of a code: the signs of its values.
    fn of(code: &[f32]) -> Self {
        let (coarse, fine) = code.split_at(BITS);
        Self {
            s1: index(coarse),
            s2: index(fine),
        }
    }

    /// The token's code as the decoder reads it.
    fn code(self) -> [f32; CODE] {
        let scale = (1.0 / (CODE as f64).sqrt()) as f32;
        let mut code = [0.0; CODE];
        for (bit, value) in code.iter_mut().enumerate() {
            let half = if bit < BITS { self.s1 } else { self.s2 };
            let set = half >> (bit % BITS) & 1 == 1;
            *value = if set { scale } else { -scale };
        }
        code
    }
}

fn index(code: &[f32]) -> u16 {
    let set = code.iter().enumerate().filter(|&(_, &value)| value > 0.0);
    set.map(|(bit, _)| 1 << bit).sum()
}

/// The tokens of codes, `rows × CODE`.
pub fn tokens(codes: &[f32]) -> Vec<Token> {
    let codes = codes.as_chunks::<CODE>().0;
    codes.iter().map(|code| Token::of(code)).collect()
}

pub struct Tokenizer {
    embed: Linear,
    encoder: Transformer,
    quantize: Linear,
    dequantize: Linear,
    decoder: Transformer,
    head: Linear,
}

/// The activations of a call over up to `MAX_CONTEXT` rows, shared by both stacks.
pub struct Buffers {
    cache: KvCache,
    stack: transformer::Buffers,
    quant: QuantizedRows,
}

impl Buffers {
    pub fn new(threads: usize) -> Self {
        Self {
            cache: KvCache::new(SHAPE, MAX_CONTEXT),
            stack: SHAPE.buffers(MAX_CONTEXT, threads),
            quant: QuantizedRows::default(),
        }
    }
}

impl Tokenizer {
    /// Each row's code before its signs are kept, `rows × CODE`, from rows of `FEATURES`
    /// values; at most `MAX_CONTEXT` rows.
    pub fn codes(&self, team: &Team, buffers: &mut Buffers, rows: &[f32]) -> Vec<f32> {
        let count = rows.len() / FEATURES;
        let hidden = &mut buffers.stack.hidden.get_mut()[..count * WIDTH];
        self.embed.apply(rows, &mut buffers.quant, hidden);
        run(team, &self.encoder, buffers, count);
        let encoded = &buffers.stack.hidden.read()[..count * WIDTH];
        let mut codes = vec![0.0; count * CODE];
        self.quantize.apply(encoded, &mut buffers.quant, &mut codes);
        codes
    }

    /// The rows of `tokens`, `tokens × FEATURES`; at most `MAX_CONTEXT` tokens.
    pub fn decode(&self, team: &Team, buffers: &mut Buffers, tokens: &[Token]) -> Vec<f32> {
        let codes: Vec<f32> = tokens.iter().flat_map(|token| token.code()).collect();
        let hidden = &mut buffers.stack.hidden.get_mut()[..tokens.len() * WIDTH];
        self.dequantize.apply(&codes, &mut buffers.quant, hidden);
        run(team, &self.decoder, buffers, tokens.len());
        let decoded = &buffers.stack.hidden.read()[..tokens.len() * WIDTH];
        let mut rows = vec![0.0; tokens.len() * FEATURES];
        self.head.apply(decoded, &mut buffers.quant, &mut rows);
        rows
    }
}

/// The first `rows` rows of the stack's buffers through `stack`, causally from position 0.
fn run(team: &Team, stack: &Transformer, buffers: &Buffers, rows: usize) {
    let span = Span {
        rows,
        first: 0,
        mask: Mask::Causal,
    };
    team.run(|ctx| stack.forward(ctx, &buffers.cache, &buffers.stack, span));
}

impl Load for Tokenizer {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, bundle::Error> {
        let encoder = Transformer::load(store, &format!("{prefix}.encoder"), SHAPE)?;
        let decoder = Transformer::load(store, &format!("{prefix}.decoder"), SHAPE)?;
        let mut scope = Scope::new(store, prefix);
        Ok(Self {
            embed: scope.load("embed", [WIDTH, FEATURES])?,
            encoder,
            quantize: scope.load("quantize", [CODE, WIDTH])?,
            dequantize: scope.load("dequantize", [WIDTH, CODE])?,
            decoder,
            head: scope.load("head", [FEATURES, WIDTH])?,
        })
    }
}
