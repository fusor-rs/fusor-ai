//! How __name__ runs, stage by stage.

/// One stage of the model's work, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Forward,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[Stage::Forward];

    fn name(self) -> &'static str {
        match self {
            Stage::Forward => "forward",
        }
    }
}
