//! The inverted-residual units of the line detector and the orientation classifier, each
//! convolution's batch norm folded in and followed by a HardSwish but a residual's projection.
//!
//! Tensors under a unit's prefix `p`: a down unit's depthwise `p.0.0` and pointwise `p.1.0`, a
//! residual unit's `p.body.0.0` (expand), `p.body.1.0` (depthwise) and `p.body.2.0` (project).

use bundle::{Load, Store};
use kernels::conv::{self, Conv2d, Depthwise2d, DepthwiseShape, Grid};
use kernels::ops::hard_swish;
use team::SharedBuf;

use crate::Error;
use crate::plane::{Layers, Plane};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitShape {
    /// Halves the grid: a depthwise `kernel` window at stride 2, then a pointwise convolution
    /// to `outputs` channels.
    Down {
        inputs: usize,
        outputs: usize,
        kernel: usize,
    },
    /// `x + project(depthwise(expand(x)))`, through `expanded` channels.
    Residual {
        channels: usize,
        expanded: usize,
        kernel: usize,
    },
}

impl UnitShape {
    pub(crate) const fn outputs(self) -> usize {
        match self {
            UnitShape::Down { outputs, .. } => outputs,
            UnitShape::Residual { channels, .. } => channels,
        }
    }
}

pub(crate) enum Unit {
    Down {
        depthwise: Depthwise2d,
        pointwise: Conv2d,
    },
    Residual {
        expand: Conv2d,
        depthwise: Depthwise2d,
        project: Conv2d,
    },
}

/// The buffers units run through: two for a unit's inner activations, two its outputs
/// alternate between.
pub(crate) struct UnitBuffers {
    inner: [SharedBuf; 2],
    outputs: [SharedBuf; 2],
}

impl UnitBuffers {
    /// Buffers for `units` over an input of `grid`.
    pub(crate) fn new(units: &[&[Unit]], grid: Grid) -> Self {
        let (mut inner, mut outer) = (0, 0);
        let mut grid = grid;
        for unit in units.iter().flat_map(|units| units.iter()) {
            let (output, unit_inner) = unit.sizes(grid);
            inner = inner.max(unit_inner);
            outer = outer.max(output.pixels() * unit.outputs());
            grid = output;
        }
        Self {
            inner: [SharedBuf::new(inner), SharedBuf::new(inner)],
            outputs: [SharedBuf::new(outer), SharedBuf::new(outer)],
        }
    }
}

impl Unit {
    fn outputs(&self) -> usize {
        match self {
            Unit::Down { pointwise, .. } => pointwise.outputs(),
            Unit::Residual { project, .. } => project.outputs(),
        }
    }

    /// The unit's output grid over `grid`, and the values of its largest inner activation.
    fn sizes(&self, grid: Grid) -> (Grid, usize) {
        match self {
            Unit::Down { depthwise, .. } => {
                let output = depthwise.output_grid(grid);
                (output, output.pixels() * depthwise.channels())
            }
            Unit::Residual { expand, .. } => (grid, grid.pixels() * expand.outputs()),
        }
    }

    fn forward<'out>(
        &self,
        layers: &Layers,
        input: Plane,
        inner: &[SharedBuf; 2],
        into: &'out SharedBuf,
    ) -> Plane<'out> {
        let activated = |_: usize, _: usize, value: f32| hard_swish(value);
        match self {
            Unit::Down {
                depthwise,
                pointwise,
            } => {
                let thin = layers.depthwise(depthwise, input, &inner[0], activated);
                layers.conv(pointwise, thin, into, activated)
            }
            Unit::Residual {
                expand,
                depthwise,
                project,
            } => {
                let wide = layers.conv(expand, input, &inner[0], activated);
                let wide = layers.depthwise(depthwise, wide, &inner[1], activated);
                layers.conv(project, wide, into, |pixel, channel, value| {
                    value + input.at(pixel, channel)
                })
            }
        }
    }
}

/// `units` over `input`, the last one's output into `last`.
pub(crate) fn run<'out>(
    layers: &Layers,
    units: &[Unit],
    input: Plane,
    buffers: &UnitBuffers,
    last: &'out SharedBuf,
) -> Plane<'out> {
    let mut plane = input;
    let mut slot = 0;
    let (final_unit, leading) = units.split_last().expect("a stage holds at least one unit");
    for unit in leading {
        plane = unit.forward(layers, plane, &buffers.inner, &buffers.outputs[slot]);
        slot = 1 - slot;
    }
    final_unit.forward(layers, plane, &buffers.inner, last)
}

impl Load for Unit {
    type Shape = UnitShape;

    fn load(store: &mut Store, prefix: &str, shape: UnitShape) -> Result<Self, Error> {
        Ok(match shape {
            UnitShape::Down {
                inputs,
                outputs,
                kernel,
            } => Unit::Down {
                depthwise: depthwise(store, &format!("{prefix}.0.0"), inputs, kernel, 2)?,
                pointwise: pointwise(store, &format!("{prefix}.1.0"), inputs, outputs)?,
            },
            UnitShape::Residual {
                channels,
                expanded,
                kernel,
            } => Unit::Residual {
                expand: pointwise(store, &format!("{prefix}.body.0.0"), channels, expanded)?,
                depthwise: depthwise(store, &format!("{prefix}.body.1.0"), expanded, kernel, 1)?,
                project: pointwise(store, &format!("{prefix}.body.2.0"), expanded, channels)?,
            },
        })
    }
}

/// A pointwise convolution of `inputs` to `outputs` channels.
pub(crate) fn pointwise(
    store: &mut Store,
    name: &str,
    inputs: usize,
    outputs: usize,
) -> Result<Conv2d, Error> {
    Conv2d::load(store, name, conv::Shape::same(inputs, outputs, 1))
}

/// A depthwise convolution of `channels` over a square `kernel` window.
pub(crate) fn depthwise(
    store: &mut Store,
    name: &str,
    channels: usize,
    kernel: usize,
    stride: usize,
) -> Result<Depthwise2d, Error> {
    let shape = DepthwiseShape {
        channels,
        window: [kernel, kernel],
        stride,
    };
    Depthwise2d::load(store, name, shape)
}
