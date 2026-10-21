//! Transcription: a recording is cut at pauses (`segment`) and each segment recognized whole —
//! features, subsampler, encoder, greedy transducer — then read as timed sentences. A file is
//! started with its plan and stepped one segment at a time; live audio is `live`'s.

use std::ops::Range;

use blocks::conformer;
use blocks::conformer::subsampling::{self, Frames};
use bundle::Store;
use serde::Serialize;
use team::Team;

use crate::Error;
use crate::decoder::{self, Step};
use crate::features::{MIN_FRAMES, MIN_SAMPLES};
use crate::live::Live;
use crate::model::{
    DECODER_HIDDEN, ENCODER_SHAPE, HOP, MELS, Model, SUBSAMPLING_SHAPE, VOCABULARY_PART,
};
use crate::segment::{self, BLOCK_SECONDS, SEGMENT_SECONDS, samples, seconds};
use crate::speech::{self, Span};
use crate::vocabulary::{Timed, Vocabulary, sentences};

/// Encoder frames each thread computes through the subsampler at a time: its working set, about
/// 2 MB.
const SUBSAMPLING_BLOCK_FRAMES: usize = 8;

pub struct Engine {
    model: Model,
    vocabulary: Vocabulary,
    team: Team,
    /// The features of the audio being read, `frames × MELS`.
    features: Vec<f32>,
    subsampling: subsampling::Buffers,
    speech: speech::Buffers,
    encoder: conformer::Buffers,
    decoder: decoder::State,
    job: Option<Job>,
    pub(crate) live: Live,
}

/// One segment's recognition.
#[derive(Debug, Clone, Serialize)]
pub struct Recognition {
    pub steps: Vec<Step>,
    pub text: String,
    /// Timed from the start of the segment.
    pub words: Vec<Timed>,
}

/// The segments a recording is cut into, in seconds.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub segments: Vec<Span>,
    pub seconds: f64,
}

/// One segment of a recording transcribed.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub segment: usize,
    pub segments: usize,
    /// The segment's sentences, timed from the start of the recording.
    pub sentences: Vec<Timed>,
    pub done: bool,
}

struct Job {
    pcm: Vec<f32>,
    segments: Vec<Range<usize>>,
    next: usize,
}

impl Engine {
    pub fn new(store: &mut Store) -> Result<Self, Error> {
        store.require_complete()?;
        let model = Model::load(store)?;
        let vocabulary = Vocabulary::parse(&store.take_blob(VOCABULARY_PART)?)?;
        let team = Team::global();
        let threads = team.threads();
        let block_frames = mel_frames(BLOCK_SECONDS);
        let frames = Frames::of(&SUBSAMPLING_SHAPE, mel_frames(SEGMENT_SECONDS)).valid;
        Ok(Self {
            model,
            vocabulary,
            features: Vec::new(),
            subsampling: subsampling::Buffers::new(
                SUBSAMPLING_SHAPE,
                block_frames,
                SUBSAMPLING_BLOCK_FRAMES,
                threads,
            ),
            speech: speech::Buffers::new(Frames::of(&SUBSAMPLING_SHAPE, block_frames).all),
            encoder: conformer::Buffers::new(ENCODER_SHAPE, frames, threads),
            decoder: decoder::State::new(frames, threads),
            team,
            job: None,
            live: Live::default(),
        })
    }

    pub fn threads(&self) -> usize {
        self.team.threads()
    }

    /// The normalized log-mel features of `pcm` (16 kHz mono), `frames × MELS`.
    pub fn features(&mut self, pcm: &[f32]) -> Result<Vec<f32>, Error> {
        self.model.frontend.compute(pcm, &mut self.features)?;
        Ok(self.features.clone())
    }

    /// Takes `features` as those of the audio being read.
    fn read_features(&mut self, features: &[f32]) {
        self.features.clear();
        self.features.extend_from_slice(features);
    }

    /// The subsampler's output for `features`, `Frames::all` rows of the encoder's width.
    pub fn subsample(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        self.read_features(features);
        let frames = frames_of(&self.features, self.subsampling.mel_frames())?;
        let Self {
            model,
            subsampling,
            features,
            team,
            ..
        } = self;
        team.run(|ctx| {
            model.subsampling.forward(ctx, subsampling, features);
        });
        Ok(subsampling.hidden.read()[..frames.all * ENCODER_SHAPE.hidden].to_vec())
    }

    /// The speech probability of every encoder frame of `features`.
    pub fn speech(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        self.read_features(features);
        self.run_speech()
    }

    /// The encoder's frames of `features` through the joint's encoder projection,
    /// `Frames::valid × DECODER_HIDDEN`.
    pub fn encode(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        self.read_features(features);
        let frames = self.run_encoder()?;
        Ok(self.decoder.encoded.read()[..frames * DECODER_HIDDEN].to_vec())
    }

    /// The transducer's decisions over encoded frames (`frames × DECODER_HIDDEN`).
    pub fn decode(&mut self, encoded: &[f32]) -> Result<Vec<Step>, Error> {
        let room = self.decoder.encoded.get_mut();
        if encoded.len() > room.len() || !encoded.len().is_multiple_of(DECODER_HIDDEN) {
            return Err(Error::Request(format!(
                "{} encoded values: whole rows of {DECODER_HIDDEN}, at most {}",
                encoded.len(),
                room.len()
            )));
        }
        room[..encoded.len()].copy_from_slice(encoded);
        Ok(self.run_decoder(encoded.len() / DECODER_HIDDEN))
    }

    /// Recognizes one segment of 16 kHz mono audio, at most `SEGMENT_SECONDS` long.
    pub fn recognize(&mut self, pcm: &[f32]) -> Result<Recognition, Error> {
        if pcm.len() > samples(SEGMENT_SECONDS) {
            return Err(Error::Request(format!(
                "{:.1} s is more than a segment's {SEGMENT_SECONDS} s",
                seconds(pcm.len())
            )));
        }
        self.model.frontend.compute(pcm, &mut self.features)?;
        let frames = self.run_encoder()?;
        let steps = self.run_decoder(frames);
        Ok(Recognition {
            text: self.vocabulary.text(&steps),
            words: self.vocabulary.words(&steps),
            steps,
        })
    }

    /// Recognizes `pcm` as timed sentences, `offset` seconds into the recording (none past the
    /// end of `pcm`).
    pub(crate) fn sentences(&mut self, pcm: &[f32], offset: f64) -> Result<Vec<Timed>, Error> {
        let limit = offset + seconds(pcm.len());
        let recognition = self.recognize(pcm)?;
        let shift = |time: f64| f64::min(time + offset, limit);
        let words: Vec<Timed> = recognition
            .words
            .into_iter()
            .map(|word| Timed {
                start: shift(word.start),
                end: shift(word.end),
                ..word
            })
            .collect();
        let text = recognition.text.trim();
        if words.is_empty() && !text.is_empty() {
            return Ok(vec![Timed {
                text: text.into(),
                start: offset,
                end: limit,
            }]);
        }
        Ok(sentences(&words))
    }

    /// The speech regions of `pcm` (16 kHz mono, at most `BLOCK_SECONDS`), as the speech head
    /// marks them.
    pub(crate) fn speech_regions(&mut self, pcm: &[f32]) -> Result<Vec<Span>, Error> {
        let duration = seconds(pcm.len());
        if pcm.len() < MIN_SAMPLES {
            return Ok(vec![Span {
                start: 0.0,
                end: duration,
            }]);
        }
        self.model.frontend.compute(pcm, &mut self.features)?;
        let regions = speech::regions(&self.run_speech()?);
        let clipped = regions.into_iter().map(|span| Span {
            end: span.end.min(duration),
            ..span
        });
        Ok(clipped.collect())
    }

    /// Starts transcribing a recording (16 kHz mono): its segments, cut at pauses.
    pub fn start(&mut self, pcm: Vec<f32>) -> Result<Plan, Error> {
        if pcm.len() < MIN_SAMPLES {
            return Err(Error::Request(
                "the recording is too short to transcribe".into(),
            ));
        }
        self.job = None;
        let segments = segment::plan(&pcm, |block| self.speech_regions(block))?;
        let spans = segments.iter().map(|range| Span {
            start: seconds(range.start),
            end: seconds(range.end),
        });
        let plan = Plan {
            segments: spans.collect(),
            seconds: seconds(pcm.len()),
        };
        self.job = Some(Job {
            pcm,
            segments,
            next: 0,
        });
        Ok(plan)
    }

    /// Transcribes the next segment of the recording.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no transcription in progress".into()))?;
        let segment = job.next;
        let sentences = match job.segments.get(segment) {
            Some(range) => self.sentences(&job.pcm[range.clone()], seconds(range.start))?,
            None => Vec::new(),
        };
        job.next += 1;
        let done = job.next >= job.segments.len();
        let segments = job.segments.len();
        if !done {
            self.job = Some(job);
        }
        Ok(Progress {
            segment,
            segments,
            sentences,
            done,
        })
    }

    pub fn cancel(&mut self) {
        self.job = None;
    }

    /// The subsampler and the speech head over `features`: the valid frames' probabilities.
    fn run_speech(&mut self) -> Result<Vec<f32>, Error> {
        let frames = frames_of(&self.features, self.subsampling.mel_frames())?;
        let Self {
            model,
            features,
            subsampling,
            speech,
            team,
            ..
        } = self;
        team.run(|ctx| {
            model.subsampling.forward(ctx, subsampling, features);
            model
                .speech
                .forward(ctx, &subsampling.hidden, frames, speech);
        });
        Ok(speech.probabilities.read()[..frames.valid].to_vec())
    }

    /// The subsampler, the encoder and the joint's encoder projection over `features`; returns
    /// the encoder frames.
    fn run_encoder(&mut self) -> Result<usize, Error> {
        let frames = frames_of(&self.features, mel_frames(SEGMENT_SECONDS))?;
        let Self {
            model,
            features,
            subsampling,
            encoder,
            decoder,
            team,
            ..
        } = self;
        let width = ENCODER_SHAPE.hidden;
        team.run(|ctx| {
            model.subsampling.forward(ctx, subsampling, features);
            let rows = ctx.split(frames.valid, 1);
            let range = rows.start * width..rows.end * width;
            let subsampled = &subsampling.hidden.read()[range.clone()];
            encoder.hidden.part(range).copy_from_slice(subsampled);
            ctx.barrier();
            model.encoder.forward(ctx, encoder, frames.valid);
            model.project_encoded(ctx, &encoder.hidden, frames.valid, decoder);
        });
        Ok(frames.valid)
    }

    /// The transducer over the first `frames` encoded frames, from a reset state.
    fn run_decoder(&mut self, frames: usize) -> Vec<Step> {
        self.decoder.reset();
        let (model, state) = (&self.model, &self.decoder);
        self.team.run(|ctx| model.decode(ctx, state, frames));
        self.decoder.take_steps()
    }
}

/// The mel frames of `seconds` of audio.
fn mel_frames(seconds: f64) -> usize {
    samples(seconds) / HOP
}

/// The frames of `features`, which must be whole rows of `MELS`, `MIN_FRAMES` to `capacity` of
/// them.
fn frames_of(features: &[f32], capacity: usize) -> Result<Frames, Error> {
    let mel_frames = features.len() / MELS;
    if !features.len().is_multiple_of(MELS) || !(MIN_FRAMES..=capacity).contains(&mel_frames) {
        return Err(Error::Request(format!(
            "{} feature values: whole rows of {MELS}, {MIN_FRAMES} to {capacity} of them",
            features.len()
        )));
    }
    Ok(Frames::of(&SUBSAMPLING_SHAPE, mel_frames))
}
