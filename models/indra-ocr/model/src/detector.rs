//! The line detector: a stride-2 stem, four stages of units down to 1/32 of the input, a
//! top-down pyramid of laterals back to 1/4 (each stage's pointwise lateral plus the coarser one
//! upsampled), smoothed and joined with the stem's lateral folded from 1/2 to 1/4 (each 2 × 2
//! block's values as channels), and a head whose twelve channels unfold into three maps at 1/2:
//! the line-core logit and the distances to the line's top and bottom (`8 · softplus`).
//!
//! Tensors under `det.`: `stem.0`, the units `stages.{s}.{u}`, `lat.{s}.0`, `smooth.0`,
//! `stem_lat.0`, `head.0.0`, `head.1.0` and `head.2`.

use bundle::{Load, Store};
use kernels::conv::{self, Conv2d, Depthwise2d, Grid};
use kernels::ops::hard_swish;
use team::{SharedBuf, Team};

use crate::Error;
use crate::mobile::{self, Unit, UnitBuffers, UnitShape, depthwise, pointwise};
use crate::plane::{Layers, Plane, Scratch};

pub(crate) const PREFIX: &str = "det";
/// The input's height and width divide by this.
pub const INPUT_MULTIPLE: usize = 32;
/// The maps' stride, in input pixels.
pub const OUT_STRIDE: usize = 2;
/// The maps: the core logit, the distance up to the line's top, down to its bottom.
pub const MAPS: usize = 3;
const STEM: usize = 16;
const STEM_KERNEL: usize = 3;
const STEM_LATERAL: usize = 4;
const LATERAL: usize = 24;
const SMOOTH_KERNEL: usize = 3;
/// The stem's lateral folded to 1/4: each 2 × 2 block's values become channels.
const FOLD: usize = 2;
/// The pyramid's channels and the stem lateral's folded ones.
const JOINED: usize = LATERAL + STEM_LATERAL * FOLD * FOLD;
const HEAD: usize = 24;
/// The distances are `DISTANCE_SCALE · softplus` of the head's.
const DISTANCE_SCALE: f32 = 8.0;
const STAGES: [&[UnitShape]; 4] = [
    &[down(STEM, 24, 3), residual(24, 48, 3)],
    &[down(24, 32, 3), residual(32, 64, 3), residual(32, 64, 3)],
    &[down(32, 48, 5), residual(48, 96, 5), residual(48, 96, 5)],
    &[down(48, 64, 5), residual(64, 128, 5), residual(64, 128, 5)],
];

pub(crate) const fn down(inputs: usize, outputs: usize, kernel: usize) -> UnitShape {
    UnitShape::Down {
        inputs,
        outputs,
        kernel,
    }
}

pub(crate) const fn residual(channels: usize, expanded: usize, kernel: usize) -> UnitShape {
    UnitShape::Residual {
        channels,
        expanded,
        kernel,
    }
}

pub struct Detector {
    stem: Conv2d,
    stages: Vec<Vec<Unit>>,
    laterals: Vec<Conv2d>,
    smooth: Depthwise2d,
    stem_lateral: Conv2d,
    head_in: Conv2d,
    head_depthwise: Depthwise2d,
    head_out: Conv2d,
}

/// The pyramid's buffers: each stage's output and lateral.
struct Pyramid {
    stages: Vec<SharedBuf>,
    laterals: Vec<SharedBuf>,
}

impl Detector {
    /// The maps of an ink map (`1 − gray/255`, `grid.height × grid.width`, multiples of
    /// `INPUT_MULTIPLE`): `MAPS` values per pixel at stride `OUT_STRIDE`, pixel-major.
    pub fn forward(&self, team: &Team, scratch: &Scratch, ink: &[f32], grid: Grid) -> Vec<f32> {
        let mut input = SharedBuf::new(ink.len());
        input.get_mut().copy_from_slice(ink);
        let half = self.stem.output_grid(grid);
        let quarter = quartered(half);
        let stem = SharedBuf::new(half.pixels() * STEM);
        let stage_refs: Vec<&[Unit]> = self.stages.iter().map(Vec::as_slice).collect();
        let units = UnitBuffers::new(&stage_refs, half);
        let pyramid = self.pyramid(half);
        let stem_lateral = SharedBuf::new(half.pixels() * STEM_LATERAL);
        let joined = SharedBuf::new(quarter.pixels() * JOINED);
        let head = [0, 1, 2].map(|_| SharedBuf::new(quarter.pixels() * HEAD));
        let maps = SharedBuf::new(half.pixels() * MAPS);
        team.run(|ctx| {
            let layers = scratch.layers(ctx);
            let input = Plane {
                values: &input,
                grid,
                channels: 1,
            };
            let stem = layers.conv(&self.stem, input, &stem, activated);
            let top = self.top_down(&layers, stem, &units, &pyramid);
            let smooth = layers.depthwise(&self.smooth, top, &head[0], activated);
            let lateral = layers.conv(&self.stem_lateral, stem, &stem_lateral, activated);
            let joined = join(&layers, smooth, lateral, &joined);
            let inner = layers.conv(&self.head_in, joined, &head[1], activated);
            let inner = layers.depthwise(&self.head_depthwise, inner, &head[2], activated);
            let out = layers.conv(&self.head_out, inner, &head[0], |_, _, value| value);
            unfold(&layers, out, &maps, half);
        });
        maps.read().to_vec()
    }

    /// Buffers for the stages' outputs and laterals over a stem output of `grid`.
    fn pyramid(&self, grid: Grid) -> Pyramid {
        let mut grid = grid;
        let mut stages = Vec::new();
        let mut laterals = Vec::new();
        for (shapes, lateral) in STAGES.iter().zip(&self.laterals) {
            grid = halved(grid);
            stages.push(SharedBuf::new(grid.pixels() * stage_outputs(shapes)));
            laterals.push(SharedBuf::new(grid.pixels() * lateral.outputs()));
        }
        Pyramid { stages, laterals }
    }

    /// The stages down from the stem, then the laterals back up to the finest stage's.
    fn top_down<'out>(
        &self,
        layers: &Layers,
        stem: Plane,
        units: &UnitBuffers,
        pyramid: &'out Pyramid,
    ) -> Plane<'out> {
        let mut plane = stem;
        let mut outputs = Vec::new();
        for (stage, buffer) in self.stages.iter().zip(&pyramid.stages) {
            plane = mobile::run(layers, stage, plane, units, buffer);
            outputs.push(plane);
        }
        let mut coarser: Option<Plane> = None;
        for ((lateral, output), buffer) in self
            .laterals
            .iter()
            .zip(outputs)
            .zip(&pyramid.laterals)
            .rev()
        {
            let finer = layers.conv(lateral, output, buffer, |pixel, channel, value| {
                let upsampled = coarser.map_or(0.0, |coarse| {
                    coarse.at(upsampled(pixel, output.grid, coarse.grid), channel)
                });
                hard_swish(value) + upsampled
            });
            coarser = Some(finer);
        }
        coarser.expect("the detector has stages")
    }
}

fn activated(_: usize, _: usize, value: f32) -> f32 {
    hard_swish(value)
}

fn halved(grid: Grid) -> Grid {
    Grid {
        height: grid.height.div_ceil(2),
        width: grid.width.div_ceil(2),
        ..grid
    }
}

fn quartered(half: Grid) -> Grid {
    Grid {
        height: half.height / FOLD,
        width: half.width / FOLD,
        ..half
    }
}

fn stage_outputs(shapes: &[UnitShape]) -> usize {
    shapes
        .last()
        .map(|shape| shape.outputs())
        .expect("a stage holds at least one unit")
}

/// The pixel of `coarse` (half of `fine`'s height and width) that nearest upsampling copies to
/// `pixel` of `fine`.
fn upsampled(pixel: usize, fine: Grid, coarse: Grid) -> usize {
    let (row, column) = (pixel / fine.width, pixel % fine.width);
    (row / 2) * coarse.width + column / 2
}

/// The smoothed pyramid's channels, then the stem lateral's, folded from 1/2 to 1/4: channel
/// `c · 4 + dy · 2 + dx` of a pixel is the lateral's channel `c` at `(2y + dy, 2x + dx)`.
fn join<'out>(
    layers: &Layers,
    smooth: Plane,
    lateral: Plane,
    into: &'out SharedBuf,
) -> Plane<'out> {
    let grid = smooth.grid;
    let channels = smooth.channels + lateral.channels * FOLD * FOLD;
    for pixel in layers.ctx.split(grid.pixels(), 1) {
        let (row, column) = (pixel / grid.width, pixel % grid.width);
        let out = into.part(pixel * channels..(pixel + 1) * channels);
        let (smoothed, folded) = out.split_at_mut(smooth.channels);
        for (channel, value) in smoothed.iter_mut().enumerate() {
            *value = smooth.at(pixel, channel);
        }
        for (index, value) in folded.iter_mut().enumerate() {
            let (channel, dy, dx) = (index / (FOLD * FOLD), index / FOLD % FOLD, index % FOLD);
            let source = (row * FOLD + dy) * lateral.grid.width + column * FOLD + dx;
            *value = lateral.at(source, channel);
        }
    }
    layers.ctx.barrier();
    Plane {
        values: into,
        grid,
        channels,
    }
}

/// The head's output unfolded to the maps at 1/2 (`DepthToSpace`, channel `c · 4 + dy · 2 +
/// dx` of a pixel to map `c` at `(2y + dy, 2x + dx)`), the distances through `8 · softplus`.
fn unfold(layers: &Layers, head: Plane, into: &SharedBuf, half: Grid) {
    for pixel in layers.ctx.split(half.pixels(), 1) {
        let (row, column) = (pixel / half.width, pixel % half.width);
        let source = (row / FOLD) * head.grid.width + column / FOLD;
        let offset = (row % FOLD) * FOLD + column % FOLD;
        for map in 0..MAPS {
            let value = head.at(source, map * FOLD * FOLD + offset);
            let value = if map == 0 {
                value
            } else {
                DISTANCE_SCALE * softplus(value)
            };
            into.set(pixel * MAPS + map, value);
        }
    }
    layers.ctx.barrier();
}

/// ONNX Runtime's softplus, `log(1 + eˣ)` without overflow.
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

impl Load for Detector {
    type Shape = ();

    fn load(store: &mut Store, prefix: &str, (): ()) -> Result<Self, Error> {
        let name = |part: &str| format!("{prefix}.{part}");
        let stem = Conv2d::load(
            store,
            &name("stem.0"),
            conv::Shape::halving(1, STEM, STEM_KERNEL),
        )?;
        let mut stages = Vec::new();
        let mut laterals = Vec::new();
        for (stage, shapes) in STAGES.iter().enumerate() {
            let units = shapes.iter().enumerate().map(|(unit, &shape)| {
                Unit::load(store, &name(&format!("stages.{stage}.{unit}")), shape)
            });
            stages.push(units.collect::<Result<_, _>>()?);
            let lateral = name(&format!("lat.{stage}.0"));
            laterals.push(pointwise(store, &lateral, stage_outputs(shapes), LATERAL)?);
        }
        Ok(Self {
            stem,
            stages,
            laterals,
            smooth: depthwise(store, &name("smooth.0"), LATERAL, SMOOTH_KERNEL, 1)?,
            stem_lateral: pointwise(store, &name("stem_lat.0"), STEM, STEM_LATERAL)?,
            head_in: pointwise(store, &name("head.0.0"), JOINED, HEAD)?,
            head_depthwise: depthwise(store, &name("head.1.0"), HEAD, SMOOTH_KERNEL, 1)?,
            head_out: pointwise(store, &name("head.2"), HEAD, MAPS * FOLD * FOLD)?,
        })
    }
}
