//! Multimodal diffusion transformer blocks (FLUX.1's, HunyuanVideo's and Motif-Video's). A call's
//! rows are a stream of image or video tokens, each turned by 3-D rotary embeddings
//! (`Rotation::Axes`, interleaved pairs) at its position, then context rows without positions
//! (text, image embeddings) that the stream attends to jointly. Every block takes its own
//! modulations of the timestep (`modulate`): shifts, scales and gates around parameter-free
//! LayerNorms.
//!
//! A dual-stream block (`Dual`) gives the stream and the context their own weights: modulated
//! norms, QKV with biases, each query and key head RMS-normalized, one attention over all rows,
//! output projections into gated residuals, then modulated norms and GELU-tanh MLPs into gated
//! residuals. A single-stream block (`Single`) gives every row the same weights: one modulated
//! norm, QKV and the MLP's first projection side by side, the joint attention, and one projection
//! of the attention's output beside the MLP's activations into a gated residual. Motif-Video's
//! carry a cross-attention too (`Cross::Text`): its queries are the stream's attention output,
//! rotated, its keys and values the context's last `text` rows as the block's norm left them, and
//! its output joins the stream's attention output before the projection.
//!
//! Tensors under a block's prefix `p`: `p.modulation` (dual: the stream's six chunks, then the
//! context's six; single: three) and, dual, `p.{qkv, context_qkv, out, context_out, ff1, ff2,
//! context_ff1, context_ff2}` linears and `p.{q_norm, k_norm, context_q_norm, context_k_norm}`
//! vectors; single, `p.{qkv, mlp, out}` linears (`out` reads the attention's output, then the MLP's
//! activations), `p.{q_norm, k_norm}` and under `Cross::Text` `p.cross.{query, context, out}`
//! linears (`context`: keys, then values) and `p.cross.{query_norm, key_norm}` vectors. Every
//! linear has a bias. A modulation chunk's order: shift, scale, gate.
//!
//! Every method runs inside a team run; every phase ends with a barrier, but `modulate`'s, which
//! the caller's barrier closes.

mod buffers;
mod dual;
mod pass;
mod single;

use crate::rotary::{Position, Rotary, Rotation, check_axes};
pub use buffers::Buffers;
pub use dual::Dual;
pub use pass::normalize_heads;
pub use single::Single;

/// The modulations of one norm, `hidden` values each.
#[derive(Clone, Copy)]
enum Chunk {
    Shift,
    Scale,
    Gate,
}

const NORM_CHUNKS: usize = Chunk::Gate as usize + 1;

/// A dual-stream block's modulated norms, in the order of its modulations: the stream's before
/// its attention and its MLP, then the context's.
#[derive(Clone, Copy)]
enum DualNorm {
    StreamAttention,
    StreamFeedForward,
    ContextAttention,
    ContextFeedForward,
}

const DUAL_NORMS: usize = DualNorm::ContextFeedForward as usize + 1;

/// The dimensions of one block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    /// Units of each MLP.
    pub feed_forward: usize,
    /// Rotary frequencies turned by a stream row's frame, row and column.
    pub rope_sections: [usize; 3],
    pub rope_base: f32,
    /// The modulated LayerNorms' and the head norms' epsilon.
    pub eps: f32,
}

impl Shape {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Modulation values of one dual-stream block: six chunks for the stream, six for the context.
    pub fn dual_modulations(&self) -> usize {
        DUAL_NORMS * NORM_CHUNKS * self.hidden
    }

    /// Modulation values of one single-stream block: one norm's chunks.
    pub fn single_modulations(&self) -> usize {
        NORM_CHUNKS * self.hidden
    }

    /// The rotary embeddings of a block loaded under `prefix`.
    fn rotary(&self, prefix: &str) -> Result<Rotary, bundle::Error> {
        check_axes(prefix, (self.hidden, self.heads), self.rope_sections)?;
        let rotation = Rotation::Axes(self.rope_sections);
        Ok(Rotary::new(rotation, self.rope_base, self.head_dim()))
    }
}

/// The most rows a call holds: all of them, the stream's, and the context's text rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    pub rows: usize,
    pub stream: usize,
    pub text: usize,
}

/// One call's rows: the stream's, rotated by `positions` (one each), then `context` rows whose
/// last `text` the cross-attentions read.
#[derive(Debug, Clone, Copy)]
pub struct Rows<'call> {
    pub positions: &'call [Position],
    pub context: usize,
    pub text: usize,
}

/// Whether a single-stream block has Motif-Video's cross-attention to the context's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cross {
    None,
    Text,
}
