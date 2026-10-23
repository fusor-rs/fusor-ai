//! The page-orientation classifier: a stride-2 stem and units down to 1/64 of a square ink
//! thumbnail of the page, the mean over its pixels and a linear head; the probabilities of the
//! page's content turned by 0, 1, 2 or 3 quarter turns counter-clockwise, or set in vertical
//! columns.
//!
//! Tensors under `orient.m.`: `body.0.0` (the stem), the units `body.{1..=10}`, `head`.

use bundle::{Load, Store};
use kernels::conv::{self, Conv2d, Grid};
use kernels::ops::{hard_swish, softmax};
use kernels::{Linear, QuantizedRows};
use team::{SharedBuf, Team};

use crate::Error;
use crate::detector::{down, residual};
use crate::mobile::{self, Unit, UnitBuffers, UnitShape};
use crate::plane::{Plane, Scratch};

/// The thumbnail's side.
pub const SIDE: usize = 768;
/// The classes: quarter turns 0 to 3, then a page set in vertical columns.
pub const CLASSES: usize = 5;
const STEM: usize = 16;
const STEM_KERNEL: usize = 3;
const UNITS: [UnitShape; 10] = [
    down(STEM, 24, 3),
    residual(24, 48, 3),
    down(24, 32, 3),
    residual(32, 64, 3),
    down(32, 48, 3),
    residual(48, 96, 3),
    down(48, 64, 3),
    residual(64, 128, 3),
    down(64, 96, 3),
    residual(96, 192, 3),
];

pub struct Orientation {
    stem: Conv2d,
    units: Vec<Unit>,
    head: Linear,
}

impl Orientation {
    /// The class probabilities of an ink thumbnail (`SIDE × SIDE`, `1 − gray/255`).
    pub fn forward(&self, team: &Team, scratch: &Scratch, ink: &[f32]) -> [f32; CLASSES] {
        let grid = Grid {
            images: 1,
            height: SIDE,
            width: SIDE,
        };
        let mut input = SharedBuf::new(ink.len());
        input.get_mut().copy_from_slice(ink);
        let half = self.stem.output_grid(grid);
        let stem = SharedBuf::new(half.pixels() * STEM);
        let buffers = UnitBuffers::new(&[&self.units], half);
        let last = UNITS.iter().fold(half, |grid, shape| match shape {
            UnitShape::Down { .. } => Grid {
                height: grid.height.div_ceil(2),
                width: grid.width.div_ceil(2),
                ..grid
            },
            UnitShape::Residual { .. } => grid,
        });
        let channels = self.head.matrix().cols;
        let output = SharedBuf::new(last.pixels() * channels);
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            let input = Plane {
                values: &input,
                grid,
                channels: 1,
            };
            let stem = layers.conv(&self.stem, input, &stem, |_, _, value| hard_swish(value));
            mobile::run(&layers, &self.units, stem, &buffers, &output);
        });
        let mut mean = vec![0.0; channels];
        for pixel in output.read().chunks_exact(channels) {
            for (sum, value) in mean.iter_mut().zip(pixel) {
                *sum += value;
            }
        }
        mean.iter_mut().for_each(|sum| *sum /= last.pixels() as f32);
        let mut probabilities = [0.0; CLASSES];
        self.head
            .apply(&mean, &mut QuantizedRows::default(), &mut probabilities);
        softmax(&mut probabilities);
        probabilities
    }
}

impl Load for Orientation {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let units = UNITS
            .iter()
            .enumerate()
            .map(|(index, &shape)| {
                Unit::load(store, &format!("{prefix}.body.{}", index + 1), shape)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            stem: Conv2d::load(
                store,
                &format!("{prefix}.body.0.0"),
                conv::Shape::halving(1, STEM, STEM_KERNEL),
            )?,
            units,
            head: Linear::load(
                store,
                &format!("{prefix}.head"),
                [CLASSES, UNITS[9].outputs()],
            )?,
        })
    }
}
