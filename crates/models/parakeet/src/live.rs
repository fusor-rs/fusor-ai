//! Live transcription. Audio arrives in pieces (`hear`) and each `update` reads what is pending:
//! once `COMMIT_SECONDS` are pending, the audio up to the middle of their last pause becomes a
//! segment of its own, transcribed once and final (cut as the file segmenter cuts, so a segment
//! is never longer than `SEGMENT_SECONDS`); the rest is transcribed again on every update,
//! provisionally. `finish` makes everything pending final.

use serde::Serialize;

use crate::Error;
use crate::segment::{self, next_cut, pauses, samples, seconds, transcribable};
use crate::session::Engine;
use crate::speech::Span;
use crate::vocabulary::Timed;

/// Pending audio is cut at a pause once it is this long.
pub const COMMIT_SECONDS: f64 = 8.0;

/// The audio heard and not yet final.
#[derive(Default)]
pub(crate) struct Live {
    pending: Vec<f32>,
    /// Samples heard before `pending`.
    offset: usize,
}

/// What an update or the end of listening adds to the transcript.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Update {
    /// Sentences that are final, timed from the start of listening.
    pub committed: Vec<Timed>,
    /// The sentences of the audio after them, replaced by the next update.
    pub provisional: Vec<Timed>,
    /// Seconds of audio heard.
    pub heard: f64,
}

impl Engine {
    /// Starts listening afresh.
    pub fn listen(&mut self) {
        self.live = Live::default();
    }

    /// The next samples of the live audio (16 kHz mono).
    pub fn hear(&mut self, samples: &[f32]) {
        self.live.pending.extend_from_slice(samples);
    }

    /// Commits the segments that are ready and transcribes the rest provisionally.
    pub fn update(&mut self) -> Result<Update, Error> {
        let mut pending = std::mem::take(&mut self.live.pending);
        let update = self.read(&mut pending);
        self.live.pending = pending;
        update
    }

    /// Makes everything pending final.
    pub fn finish(&mut self) -> Result<Update, Error> {
        let mut update = self.update()?;
        update.committed.append(&mut update.provisional);
        self.live.offset += self.live.pending.len();
        self.live.pending.clear();
        Ok(update)
    }

    fn read(&mut self, pending: &mut Vec<f32>) -> Result<Update, Error> {
        let mut committed = Vec::new();
        let regions = loop {
            let marked = pending.len().min(samples(segment::BLOCK_SECONDS));
            let regions = self.speech_regions(&pending[..marked])?;
            let Some(cut) = commit_cut(&regions, marked) else {
                break regions;
            };
            committed.extend(self.commit(pending, cut, &regions)?);
        };
        let mut provisional = Vec::new();
        if transcribable(&regions, pending.len()) {
            provisional = self.sentences(pending, seconds(self.live.offset))?;
        }
        Ok(Update {
            committed,
            provisional,
            heard: seconds(self.live.offset + pending.len()),
        })
    }

    /// The first `cut` samples of `pending` as a segment: their sentences, if they hold speech.
    fn commit(
        &mut self,
        pending: &mut Vec<f32>,
        cut: usize,
        regions: &[Span],
    ) -> Result<Vec<Timed>, Error> {
        let offset = seconds(self.live.offset);
        let sentences = if transcribable(regions, cut) {
            self.sentences(&pending[..cut], offset)?
        } else {
            Vec::new()
        };
        pending.drain(..cut);
        self.live.offset += cut;
        Ok(sentences)
    }
}

/// Where pending audio is cut, given the speech of its first `len` samples: once they are
/// `COMMIT_SECONDS` long and have a pause to cut at (or reached a segment's length).
fn commit_cut(regions: &[Span], len: usize) -> Option<usize> {
    let duration = seconds(len);
    if duration < COMMIT_SECONDS {
        return None;
    }
    let cut = next_cut(&pauses(regions, duration));
    (cut <= duration).then(|| samples(cut).min(len))
}
