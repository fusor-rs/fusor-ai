//! How indra-ocr reads a page, stage by stage.

/// One stage of reading a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The skew search and the orientation network on the preview; the page turned upright
    /// and straightened.
    Orient,
    /// The line detector, at a larger scale too when the lines come out small.
    Detect,
    /// The line crops, the script classifier routing each line to a recognizer, and the math
    /// classifier when math is read.
    Classify,
    /// One batch of lines of similar width through a recognizer.
    Recognize,
    /// One batch of math lines through the math encoder and decoder.
    Math,
    /// Lines split and tidied, the reading-order network, layout blocks and Markdown.
    Assemble,
}

impl bundle::Stages for Stage {
    const ALL: &'static [Stage] = &[
        Stage::Orient,
        Stage::Detect,
        Stage::Classify,
        Stage::Recognize,
        Stage::Math,
        Stage::Assemble,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Orient => "orient",
            Stage::Detect => "detect",
            Stage::Classify => "classify",
            Stage::Recognize => "recognize",
            Stage::Math => "math",
            Stage::Assemble => "assemble",
        }
    }
}
