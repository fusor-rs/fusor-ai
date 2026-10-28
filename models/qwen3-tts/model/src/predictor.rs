//! The code predictor: given a frame's first code and the talker's state, five Qwen3 layers draw
//! its other 15 codes one at a time. Its input is the talker's final-normed state, the first
//! code's codec embedding, then each drawn code through its codebook's own embedding table; the
//! head of the step reads the last row. A frame is predicted from a fresh cache, positions 0 to
//! 15.
//!
//! Tensors: the stack's (`predictor.layers.{i}…`), its final norm `predictor.norm`, and for each
//! of the 15 codebooks drawn, the embedding table `predictor.embed.{g}` of its code and the head
//! `predictor.heads.{g}` over the next codebook's codes.

use blocks::lm::{Lm, Prediction};
use blocks::transformer::{Buffers, KvCache, Mask, Span, Transformer};
use bundle::{Load, Store};
use kernels::{Linear, Table, Vector};
use team::Team;

use crate::Error;
use crate::model::{CODE_GROUPS, CODES, Frame, HIDDEN, PREDICTOR};
use crate::sampling::Sampler;

/// The codes the predictor draws.
const DRAWN: usize = CODE_GROUPS - 1;
/// The state and the first code's embedding, which the first pass reads.
const OPENING_ROWS: usize = 2;

pub(crate) struct Predictor {
    stack: Transformer,
    norm: Vector,
    embed: Vec<Table>,
    heads: Vec<Linear>,
    cache: KvCache,
    buffers: Buffers,
    prediction: Prediction,
    logits: Vec<f32>,
}

impl Predictor {
    pub(crate) fn load(store: &mut Store, threads: usize) -> Result<Self, Error> {
        let embed = (0..DRAWN)
            .map(|group| Table::load(store, &format!("predictor.embed.{group}"), [CODES, HIDDEN]))
            .collect::<Result<_, _>>()?;
        let heads = (0..DRAWN)
            .map(|group| Linear::load(store, &format!("predictor.heads.{group}"), [CODES, HIDDEN]))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            stack: Transformer::load(store, "predictor", PREDICTOR)?,
            norm: Vector::load(store, "predictor.norm", HIDDEN)?,
            embed,
            heads,
            cache: KvCache::new(PREDICTOR, CODE_GROUPS),
            buffers: PREDICTOR.buffers(OPENING_ROWS, threads),
            prediction: Prediction::new(HIDDEN, CODES, threads),
            logits: Vec::new(),
        })
    }

    /// The row the talker reads for `frame`: its codes' embeddings summed.
    pub(crate) fn frame_row(&self, lm: &Lm, frame: &Frame) -> Vec<f32> {
        let mut row = vec![0.0; HIDDEN];
        lm.embed(&frame[..1], &mut row);
        let mut code_row = vec![0.0; HIDDEN];
        for (table, &code) in self.embed.iter().zip(&frame[1..]) {
            table.matrix().dequantize_row(code as usize, &mut code_row);
            for (sum, value) in row.iter_mut().zip(&code_row) {
                *sum += value;
            }
        }
        row
    }

    /// The frame whose first code is `code0`, from the talker's final-normed state `hidden`, each
    /// other code drawn by `sampler`.
    pub(crate) fn frame(
        &mut self,
        (team, lm): (&Team, &Lm),
        hidden: &[f32],
        code0: u32,
        sampler: &mut Sampler,
    ) -> Frame {
        let mut frame = [code0; CODE_GROUPS];
        self.pass((team, lm), hidden, &mut frame, |logits| {
            sampler.draw(logits)
        });
        frame
    }

    /// The logits of each of the 15 passes over `frame`, read as the reference's teacher-forced
    /// pass reads it, `15 × CODES`.
    pub(crate) fn forced(
        &mut self,
        (team, lm): (&Team, &Lm),
        hidden: &[f32],
        frame: &Frame,
    ) -> Vec<f32> {
        let mut logits = Vec::with_capacity(DRAWN * CODES);
        let mut group = 0;
        let mut forced = *frame;
        self.pass((team, lm), hidden, &mut forced, |pass_logits| {
            logits.extend_from_slice(pass_logits);
            group += 1;
            frame[group]
        });
        logits
    }

    /// Fills `frame[1..]`: `choose` takes each pass's logits and returns the code the next pass
    /// reads.
    fn pass(
        &mut self,
        (team, lm): (&Team, &Lm),
        hidden: &[f32],
        frame: &mut Frame,
        mut choose: impl FnMut(&mut [f32]) -> u32,
    ) {
        let rows = self.buffers.hidden.get_mut();
        rows[..HIDDEN].copy_from_slice(hidden);
        lm.embed(&frame[..1], &mut rows[HIDDEN..OPENING_ROWS * HIDDEN]);
        for group in 0..DRAWN {
            let (rows_read, first) = if group == 0 {
                (OPENING_ROWS, 0)
            } else {
                (1, group + 1)
            };
            self.run(
                team,
                group,
                Span {
                    rows: rows_read,
                    first,
                    mask: Mask::Causal,
                },
            );
            self.logits.clear();
            self.logits.extend_from_slice(self.prediction.logits.read());
            let code = choose(&mut self.logits);
            frame[group + 1] = code;
            let row = &mut self.buffers.hidden.get_mut()[..HIDDEN];
            self.embed[group]
                .matrix()
                .dequantize_row(code as usize, row);
        }
    }

    /// One pass of the stack over `span`'s rows, then head `group` over the last row.
    fn run(&self, team: &Team, group: usize, span: Span) {
        let last = (span.rows - 1) * HIDDEN;
        team.run(|ctx| {
            self.stack.forward(ctx, &self.cache, &self.buffers, span);
            let hidden = &self.buffers.hidden.read()[last..last + HIDDEN];
            let norm = (&self.norm, PREDICTOR.norm_eps);
            self.prediction
                .predict(ctx, hidden, norm, &self.heads[group]);
        });
    }
}
