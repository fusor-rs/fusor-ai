//! What a reply writes next, as the model's own generation decides it: text, speech, or blocks
//! of 6 text tokens and 12 spoken frames in turn. A text reply ends at `<|im_end|>` or
//! `<|text_end|>`; a spoken one goes on as text after the frame that ends its speech; in an
//! interleaved one, `<|text_end|>` ends the text and the speech goes on to its own end.

use crate::audio::{self, Frame};
use crate::prompt::{
    AUDIO_OUT_TOKEN, AUDIO_START_TOKEN, AUDIO_TOKEN, MIXED_START_TOKEN, Mode, REPLY_END, TEXT_END,
};

/// Text tokens of an interleaved block.
const BLOCK_TEXT: usize = 6;
/// Spoken frames of an interleaved block.
const BLOCK_FRAMES: usize = 12;

/// Whether text may turn into speech, by writing a speech marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Markers {
    Suppressed,
    Allowed,
}

/// Where an interleaved reply is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Block {
    speaking: bool,
    /// The reply's text has ended: the rest is speech.
    text_done: bool,
    /// Rows left of the block's text or speech.
    left: usize,
}

impl Block {
    fn opened() -> Self {
        Self {
            speaking: false,
            text_done: false,
            left: BLOCK_TEXT,
        }
    }

    /// One more row of the reply's text or speech, before its text ends.
    fn count(&mut self) {
        if self.text_done {
            return;
        }
        self.left -= 1;
        if self.left == 0 {
            self.speaking = !self.speaking;
            self.left = if self.speaking {
                BLOCK_FRAMES
            } else {
                BLOCK_TEXT
            };
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Output {
    Text(Markers),
    Speech,
    Interleaved(Block),
}

impl Output {
    pub(crate) fn of(mode: Mode) -> Self {
        match mode {
            Mode::Text | Mode::Transcribe => Self::Text(Markers::Suppressed),
            Mode::Speech => Self::Speech,
            Mode::Interleaved => Self::Interleaved(Block::opened()),
        }
    }

    /// Whether the reply's next row is a spoken frame.
    pub(crate) fn speaks(self) -> bool {
        match self {
            Self::Speech => true,
            Self::Interleaved(block) => block.speaking,
            Self::Text(_) => false,
        }
    }

    /// Sets the logits of the tokens the reply may not write next to negative infinity: the
    /// placeholders of heard and spoken rows, and the speech markers unless text may turn into
    /// speech.
    pub(crate) fn suppress(self, logits: &mut [f32]) {
        let markers = match self {
            Self::Text(Markers::Suppressed) => [AUDIO_START_TOKEN, MIXED_START_TOKEN].as_slice(),
            Self::Text(Markers::Allowed) | Self::Speech | Self::Interleaved(_) => &[],
        };
        for &token in [AUDIO_TOKEN, AUDIO_OUT_TOKEN].iter().chain(markers) {
            logits[token as usize] = f32::NEG_INFINITY;
        }
    }

    /// The reply wrote `token`: whether it ends the reply.
    pub(crate) fn wrote(&mut self, token: u32) -> bool {
        match (token, &mut *self) {
            (TEXT_END, Self::Interleaved(block)) => {
                block.text_done = true;
                block.speaking = true;
                false
            }
            (TEXT_END | REPLY_END, _) => true,
            (_, Self::Interleaved(block)) => {
                block.count();
                false
            }
            (AUDIO_START_TOKEN, _) => {
                *self = Self::Speech;
                false
            }
            (MIXED_START_TOKEN, _) => {
                *self = Self::Interleaved(Block::opened());
                false
            }
            _ => false,
        }
    }

    /// The reply spoke `frame`: the end of its speech goes on as text, which may speak again.
    pub(crate) fn spoke(&mut self, frame: &Frame) {
        if audio::ends(frame) {
            *self = Self::Text(Markers::Allowed);
        } else if let Self::Interleaved(block) = self {
            block.count();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Output;
    use crate::audio::{END_CODE, Frame};
    use crate::prompt::{Mode, TEXT_END};

    /// An interleaved reply writes 6 text tokens, speaks 12 frames, writes again, then speaks on
    /// past its `<|text_end|>` until a frame ends its speech.
    #[test]
    fn interleaves_six_tokens_and_twelve_frames() {
        let (token, frame, end): (u32, Frame, Frame) = (1, [0; 12], [END_CODE; 12]);
        let mut output = Output::of(Mode::Interleaved);
        let mut rows = String::new();
        for _ in 0..20 {
            if output.speaks() {
                output.spoke(&frame);
                rows.push('f');
            } else {
                output.wrote(token);
                rows.push('t');
            }
        }
        assert_eq!(rows, "ttttttfffffffffffftt");
        assert!(!output.wrote(TEXT_END));
        assert!(output.speaks());
        output.spoke(&end);
        assert!(!output.speaks());
        assert!(output.wrote(TEXT_END));
    }
}
