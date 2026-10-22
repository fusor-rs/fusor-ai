//! The sampler presets: how many flow-matching steps a frame takes and which of them are guided.

use serde::{Deserialize, Serialize};

use crate::flow_head::{FlowHead, Steps};

const CFG_SCALE: f32 = 2.0;
/// EPSS knots in units of 1/32 (`fm/CFM.py`).
const EPSS_10: [u32; 11] = [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
const EPSS_7: [u32; 8] = [0, 2, 4, 6, 8, 16, 24, 32];
/// Guidance interval of the faster presets: CFG only for `t ∈ [0.05, 0.8)`.
const GUIDED: std::ops::Range<f32> = 0.05..0.8;

/// In the order of `ALL`, which `Schedules` relies on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    /// 10 steps, CFG on every step: the reference sampler.
    #[default]
    Quality,
    /// 10 steps, CFG on 5 of them.
    Balanced,
    /// 7 steps, CFG on 3 of them (may slur numbers).
    Fast,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Quality, Preset::Balanced, Preset::Fast];

    fn steps(self, flow: &FlowHead) -> Steps {
        match self {
            Preset::Quality => flow.steps(&EPSS_10, |_, _| true, CFG_SCALE),
            Preset::Balanced => flow.steps(&EPSS_10, |_, time| GUIDED.contains(&time), CFG_SCALE),
            Preset::Fast => flow.steps(&EPSS_7, |_, time| GUIDED.contains(&time), CFG_SCALE),
        }
    }
}

/// Every preset's steps, prepared once.
pub(crate) struct Schedules([Steps; Preset::ALL.len()]);

impl Schedules {
    pub(crate) fn new(flow: &FlowHead) -> Self {
        Self(Preset::ALL.map(|preset| preset.steps(flow)))
    }

    pub(crate) fn get(&self, preset: Preset) -> &Steps {
        &self.0[preset as usize]
    }
}
