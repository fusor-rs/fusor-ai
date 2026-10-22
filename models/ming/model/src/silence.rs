//! Trailing silence trimmed per fragment as the reference does (`trim_trailing_silence`): 0.1 s
//! chunks, a chunk whose mean `|x|` is at most 1e-3 is silent, and 0.3 s are kept after the last
//! loud chunk.

use crate::model::SAMPLE_RATE;

const CHUNK: usize = SAMPLE_RATE as usize / 10;
const SILENCE: f32 = 1e-3;
const TAIL: usize = 3 * CHUNK;

/// Streams a fragment's audio while trimming its trailing silence: silent chunks are held until a
/// loud one follows; at the end at most `TAIL` of them remain and the partial last chunk is
/// dropped.
#[derive(Default)]
pub(crate) struct SilenceGate {
    chunk: Vec<f32>,
    held: Vec<f32>,
}

impl SilenceGate {
    pub(crate) fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        for &sample in samples {
            self.chunk.push(sample);
            if self.chunk.len() < CHUNK {
                continue;
            }
            let loud =
                self.chunk.iter().map(|sample| sample.abs()).sum::<f32>() / CHUNK as f32 > SILENCE;
            if loud {
                out.append(&mut self.held);
                out.append(&mut self.chunk);
            } else {
                self.held.append(&mut self.chunk);
            }
        }
    }

    pub(crate) fn finish(&mut self, out: &mut Vec<f32>) {
        let keep = self.held.len().min(TAIL);
        out.extend_from_slice(&self.held[..keep]);
        *self = Self::default();
    }
}
