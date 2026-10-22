//! Cutting a recording at pauses into segments of at most `SEGMENT_SECONDS`, as the reference
//! does: the speech head marks speech, a cut falls in the middle of the last pause that fits in
//! the segment, and a segment without speech is not transcribed.

use std::ops::Range;

use crate::Error;
use crate::features::MIN_SAMPLES;
use crate::model::SAMPLE_RATE;
use crate::speech::Span;

pub(crate) const SEGMENT_SECONDS: f64 = 30.0;
/// Shorter silences are not pauses.
const MIN_PAUSE_SECONDS: f64 = 0.2;
/// No cut falls before this.
const MIN_SEGMENT_SECONDS: f64 = 1.0;
/// The speech head reads a long recording in blocks of this many seconds (no fewer than a
/// segment's).
pub(crate) const BLOCK_SECONDS: f64 = 120.0;
/// Pause edges lie on the speech head's frame grid, so a pause of exactly `MIN_PAUSE_SECONDS`
/// can measure a rounding error short.
const PAUSE_EPSILON: f64 = 1e-6;

/// `seconds` of 16 kHz audio as a sample count.
pub(crate) fn samples(seconds: f64) -> usize {
    (seconds * f64::from(SAMPLE_RATE)).round_ties_even() as usize
}

pub(crate) fn seconds(samples: usize) -> f64 {
    samples as f64 / f64::from(SAMPLE_RATE)
}

/// The silences of at least `MIN_PAUSE_SECONDS` between, before and after the speech of a
/// recording of `duration` seconds (its spans in order).
pub(crate) fn pauses(speech: &[Span], duration: f64) -> Vec<Span> {
    let least = MIN_PAUSE_SECONDS - PAUSE_EPSILON;
    let mut pauses = Vec::new();
    let mut previous = 0.0;
    for span in speech {
        if span.start - previous >= least {
            pauses.push(Span {
                start: previous,
                end: span.start,
            });
        }
        previous = f64::max(previous, span.end);
    }
    if duration - previous >= least {
        pauses.push(Span {
            start: previous,
            end: duration,
        });
    }
    pauses
}

/// Where a segment starting at 0 ends, given the pauses ahead of it: the middle of the last
/// pause lying within `MIN_SEGMENT_SECONDS..SEGMENT_SECONDS`, else of the last whose middle
/// does, else `SEGMENT_SECONDS`.
pub(crate) fn next_cut(pauses: &[Span]) -> f64 {
    let middle = |pause: &Span| (pause.start + pause.end) / 2.0;
    let within = pauses
        .iter()
        .rfind(|pause| pause.start >= MIN_SEGMENT_SECONDS && pause.end <= SEGMENT_SECONDS);
    let centred = || {
        let window = MIN_SEGMENT_SECONDS..=SEGMENT_SECONDS;
        pauses.iter().rfind(|pause| window.contains(&middle(pause)))
    };
    within.or_else(centred).map_or(SEGMENT_SECONDS, middle)
}

/// Whether the first `samples` of audio are worth transcribing: long enough for the features,
/// with speech in them.
pub(crate) fn transcribable(speech: &[Span], samples: usize) -> bool {
    let duration = seconds(samples);
    samples >= MIN_SAMPLES
        && speech
            .iter()
            .any(|span| span.end > 0.0 && span.start < duration)
}

/// The speech after the first `seconds`, from there.
pub(crate) fn after(speech: &[Span], seconds: f64) -> Vec<Span> {
    let kept = speech.iter().filter(|span| span.end > seconds);
    kept.map(|span| Span {
        start: f64::max(span.start - seconds, 0.0),
        end: span.end - seconds,
    })
    .collect()
}

/// The sample ranges of a recording to transcribe. A recording that fits in one segment is one,
/// whole. A longer one is walked in blocks of `BLOCK_SECONDS`, each marked by `speech`, and cut
/// as soon as more than a segment of it is in hand; segments without speech are left out.
pub(crate) fn plan(
    pcm: &[f32],
    mut speech: impl FnMut(&[f32]) -> Result<Vec<Span>, Error>,
) -> Result<Vec<Range<usize>>, Error> {
    let cap = samples(SEGMENT_SECONDS);
    if pcm.len() <= cap {
        #[expect(
            clippy::single_range_in_vec_init,
            reason = "one segment, the whole recording"
        )]
        return Ok(vec![0..pcm.len()]);
    }
    let mut segments = Vec::new();
    let mut regions: Vec<Span> = Vec::new();
    let mut carried = 0..0;
    for block in pcm.chunks(samples(BLOCK_SECONDS)) {
        let offset = seconds(carried.len());
        let marked = speech(block)?.into_iter().map(|span| Span {
            start: span.start + offset,
            end: span.end + offset,
        });
        regions.extend(marked);
        carried.end += block.len();
        while carried.len() > cap {
            let cut = samples(next_cut(&pauses(&regions, seconds(carried.len()))));
            if transcribable(&regions, cut) {
                segments.push(carried.start..carried.start + cut);
            }
            carried.start += cut;
            regions = after(&regions, seconds(cut));
        }
    }
    if transcribable(&regions, carried.len()) {
        segments.push(carried);
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::{plan, samples};
    use crate::speech::Span;

    #[test]
    fn cuts_long_recordings_in_the_middle_of_their_last_pause() {
        let speech = [
            (0.5, 10.0),
            (10.5, 25.0),
            (26.0, 40.0),
            (41.0, 60.0),
            (61.0, 69.0),
        ];
        let speech: Vec<Span> = speech
            .into_iter()
            .map(|(start, end)| Span { start, end })
            .collect();
        let pcm = vec![0.0; samples(70.0)];
        let segments = plan(&pcm, |_| Ok(speech.clone())).unwrap();
        let cuts = [0.0, 25.5, 40.5, 70.0].map(samples);
        assert_eq!(
            segments,
            [cuts[0]..cuts[1], cuts[1]..cuts[2], cuts[2]..cuts[3]]
        );
    }
}
