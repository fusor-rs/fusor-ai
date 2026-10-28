//! How d1 answers a request, stage by stage.

/// One stage of a request, with what it consumes and produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// One of the request's images through the vision tower and the projector: its tokens.
    See,
    /// One of its clips through the audio encoder: its tokens (d1-Omni-2B).
    Hear,
    /// One pack of its questions after the state, the media's tokens in their rows, through the
    /// language model: every row's final-normed state, and each question's answer from its
    /// slots'.
    Read,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[Stage::See, Stage::Hear, Stage::Read];

    fn name(self) -> &'static str {
        match self {
            Stage::See => "see",
            Stage::Hear => "hear",
            Stage::Read => "read",
        }
    }
}
