//! The text engine: a prompt's tokens through T5Gemma2, and a first image through SigLIP.

use blocks::interleaved::State;
use blocks::transformer::Buffers;
use blocks::vit::Grid;
use blocks::vit::tower::{self, Tower};
use bundle::{Load, Store, Tokenizer};
use team::{SharedBuf, Team};

use crate::image::{patches, vision_pixels};
use crate::model::{
    IMAGE_CHANNELS, IMAGE_HIDDEN, IMAGE_TOKENS, TEXT_HIDDEN, TEXT_SHAPE, TEXT_TOKENS, TOWER,
    VISION, VISION_PATCH, VISION_SIDE, VOCABULARY,
};
use crate::prompt::tokenize;
use crate::session::check_size;
use crate::text::TextEncoder;
use crate::{Error, Size};

pub struct TextEngine {
    encoder: TextEncoder,
    tower: Tower,
    tokenizer: Tokenizer,
    team: Team,
    state: State,
    buffers: Buffers,
    states: SharedBuf,
    vision: tower::Buffers,
    /// The tower's learned positions at its own grid: no resampling.
    positions: Vec<f32>,
}

impl bundle::Engine for TextEngine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let encoder = TextEncoder::load(store)?;
        let tower = Tower::load(store, VISION, TOWER)?;
        let tokenizer = Tokenizer::take(store)?;
        let team = Team::global();
        let threads = team.threads();
        Ok(Self {
            encoder,
            positions: tower.positions(vision_grid()),
            tower,
            tokenizer,
            state: State::encoder(&TEXT_SHAPE, TEXT_TOKENS),
            buffers: TEXT_SHAPE.buffers(TEXT_TOKENS, threads),
            states: SharedBuf::new(TEXT_TOKENS * TEXT_HIDDEN),
            vision: tower::Buffers::new(&TOWER, IMAGE_TOKENS, threads),
            team,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {}
}

/// SigLIP's patch grid: `VISION_SIDE / VISION_PATCH` patches along each side.
fn vision_grid() -> Grid {
    Grid {
        rows: VISION_SIDE / VISION_PATCH,
        columns: VISION_SIDE / VISION_PATCH,
    }
}

impl TextEngine {
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        tokenize(&self.tokenizer, text)
    }

    /// T5Gemma2's last hidden states of `tokens` (1 to `TEXT_TOKENS`, each below the
    /// vocabulary), rows of `TEXT_HIDDEN`.
    pub fn encode_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        if !(1..=TEXT_TOKENS).contains(&tokens.len())
            || tokens.iter().any(|&token| token as usize >= VOCABULARY)
        {
            return Err(Error::Request(format!(
                "{} tokens: 1 to {TEXT_TOKENS}, each below {VOCABULARY}",
                tokens.len()
            )));
        }
        let Self {
            encoder,
            team,
            state,
            buffers,
            states,
            ..
        } = self;
        team.run(|ctx| encoder.encode(ctx, (state, buffers), tokens, states));
        Ok(states.read()[..tokens.len() * TEXT_HIDDEN].to_vec())
    }

    /// The states of `text`'s tokens.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        let tokens = self.tokens(text)?;
        self.encode_tokens(&tokens)
    }

    /// SigLIP's last hidden states of `pixels` (`vision_pixels`'), `IMAGE_TOKENS` rows of
    /// `IMAGE_HIDDEN`.
    pub fn embed_pixels(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        let expected = VISION_SIDE * VISION_SIDE * IMAGE_CHANNELS;
        if pixels.len() != expected {
            return Err(Error::Request(format!(
                "{} pixel values: {expected}, {VISION_SIDE}×{VISION_SIDE} RGB",
                pixels.len()
            )));
        }
        let patches = patches(pixels);
        let Self {
            tower,
            team,
            vision,
            positions,
            ..
        } = self;
        team.run(|ctx| {
            tower.embed(ctx, vision, &patches, positions);
            ctx.barrier();
            tower.encode(ctx, vision, vision_grid());
        });
        Ok(vision.encoder.hidden.read()[..IMAGE_TOKENS * IMAGE_HIDDEN].to_vec())
    }

    /// `embed_pixels` of a first frame of `size` (`image::first_frame`'s).
    pub fn embed_frame(&mut self, frame: &[f32], size: Size) -> Result<Vec<f32>, Error> {
        check_size(size.width, size.height)?;
        if frame.len() != size.width * size.height * IMAGE_CHANNELS {
            return Err(Error::Request(format!(
                "{} frame values for {}×{} RGB pixels",
                frame.len(),
                size.width,
                size.height
            )));
        }
        self.embed_pixels(&vision_pixels(frame, size))
    }
}
