//! A UNet's levels: per path, a level's residual blocks, the transformer after each (at the
//! levels that have them) and its resampling convolution; and the middle block.

use bundle::{Load, Store, load_each};
use kernels::conv::{self, Conv2d};

use super::Shape;
use super::transformer::{self, Transformer};
use crate::resnet::{self, Resnet};

impl Shape {
    pub(super) fn down(&self, level: usize) -> Stage {
        let channels = self.channels[level];
        let first = self.level_channels(level as isize - 1);
        let resnets = (0..self.blocks)
            .map(|block| self.resnet(if block == 0 { first } else { channels }, channels))
            .collect();
        let resample = (level + 1 < self.levels())
            .then(|| conv::Shape::halving(channels, channels, self.kernel));
        Stage {
            resnets,
            transformer: self.transformer(channels, self.depths[level]),
            resample,
        }
    }

    pub(super) fn up(&self, level: usize) -> Stage {
        let channels = self.channels[level];
        let below = self.level_channels(level as isize + 1);
        let above = self.level_channels(level as isize - 1);
        let resnets = (0..=self.blocks)
            .map(|block| {
                let hidden = if block == 0 { below } else { channels };
                let skip = if block < self.blocks { channels } else { above };
                self.resnet(hidden + skip, channels)
            })
            .collect();
        let resample = (level > 0).then(|| conv::Shape::same(channels, channels, self.kernel));
        Stage {
            resnets,
            transformer: self.transformer(channels, self.depths[level]),
            resample,
        }
    }
}

/// The residual blocks of one level on one path, the transformer each is followed by (where
/// the level has them) and the resampling convolution after them.
pub(super) struct Stage {
    pub(super) resnets: Vec<resnet::Shape>,
    pub(super) transformer: transformer::Shape,
    pub(super) resample: Option<conv::Shape>,
}

pub(super) struct Level {
    pub(super) resnets: Vec<Resnet>,
    pub(super) transformers: Vec<Transformer>,
    pub(super) resample: Option<Conv2d>,
}

pub(super) struct Middle {
    pub(super) first: Resnet,
    pub(super) transformer: Transformer,
    pub(super) second: Resnet,
}

impl Load for Level {
    type Shape = Stage;

    fn load(store: &mut Store, prefix: &str, stage: Stage) -> Result<Self, bundle::Error> {
        let resnets = (stage.resnets.iter().enumerate())
            .map(|(block, &shape)| Resnet::load(store, &format!("{prefix}.resnets.{block}"), shape))
            .collect::<Result<Vec<_>, _>>()?;
        let count = if stage.transformer.depth > 0 {
            resnets.len()
        } else {
            0
        };
        let transformers = load_each(
            store,
            &format!("{prefix}.transformers"),
            count,
            |store, name| Transformer::load(store, name, stage.transformer),
        )?;
        let resample = match stage.resample {
            Some(shape) => Some(Conv2d::load(store, &format!("{prefix}.resample"), shape)?),
            None => None,
        };
        Ok(Self {
            resnets,
            transformers,
            resample,
        })
    }
}
