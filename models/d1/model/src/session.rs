//! Answering a request: `start` checks it, reads its images into patches and its clips into
//! features, and packs its questions after its state; then each `step` does one stage — an image
//! through the vision tower and the projector, a clip through the audio encoder, or one pack
//! through the language model, its full attention layers reading the state and every question
//! both ways, each question the state and itself (`Mask::Groups`), the window and conv layers
//! causally, and each of the pack's questions answered from its slots. The stages also run one
//! by one, on given inputs, for checks against the reference.

use ::media::image::Pixels;
use ::media::mel::MelSpectrogram;
use blocks::audio::{self, Encoder};
use blocks::hybrid::Rows;
use blocks::interleaved::State;
use blocks::rotary::Position;
use blocks::transformer::{Buffers, Mask};
use blocks::vit::preprocessing::Patches;
use blocks::vit::{Grid, merger, tower};
use bundle::{Error, Load, Stages as _, Store, Tokenizer};
use kernels::dot_f32;
use kernels::ops::rms_norm;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::answer::{self, Boxing, Output};
use crate::calibration::Calibration;
use crate::media::{self, IMAGE, MAX_MEL_FRAMES, MEL};
use crate::model::{
    AUDIO, CALIBRATION, CONFIG, Config, HIDDEN, LANGUAGE, Model, NORM_EPS, PROJECTOR, TOWER,
    Variant, Vision,
};
use crate::pipeline::Stage;
use crate::pointer::StatePositions;
use crate::prompt::{self, Packed, Prompter, Readout, Suffix, Table, Tables};
use crate::request::{Question, Request};
use team::Team;

/// The longest pack, in rows: the state, its media's tokens and its questions.
pub const MAX_ROWS: usize = 4096;
pub const MAX_IMAGES: usize = 4;
pub const MAX_CLIPS: usize = 4;
/// The token of an image's rows, `<|image|>`, and of a clip's, `<|audio|>`.
const IMAGE_TOKEN: u32 = 125_030;
const AUDIO_TOKEN: u32 = 125_031;
/// Mel frames a thread of the audio encoder's subsampler takes at once.
const BLOCK_FRAMES: usize = 64;

/// How a request is read: each image's patch grid, each clip's tokens, and its packs' rows.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub model: &'static str,
    pub images: Vec<[usize; 2]>,
    pub clips: Vec<usize>,
    pub packs: Vec<usize>,
    pub questions: usize,
}

/// What `system_one` returns: each question's answer under its name, in request order, and the
/// tokens read.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// One step done: the stages done of all, and the response once every question is answered.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub done: usize,
    pub steps: usize,
    pub response: Option<Response>,
}

/// A request ready to read: its state and questions, its slots' tables, and its packs.
pub struct Prepared {
    pub state: Value,
    pub questions: Vec<(String, Question)>,
    pub tables: Tables,
    pub packs: Vec<Pack>,
}

/// Questions `first..first + slots.len()` packed after the state, and what each slot reads; a
/// pack with a read keeps the state's positions.
pub struct Pack {
    pub packed: Packed,
    pub first: usize,
    pub readouts: Vec<Vec<Readout>>,
    pub positions: Option<StatePositions>,
}

/// d1-Omni-2B's audio encoder, its activations and its features' spectrogram.
struct Hearing {
    encoder: Encoder,
    buffers: audio::Buffers,
    spectrogram: MelSpectrogram,
}

pub struct Engine {
    variant: Variant,
    config: Config,
    model: Model,
    prompter: Prompter,
    calibration: Calibration,
    vision: Vision,
    hearing: Option<Hearing>,
    team: Team,
    tower: tower::Buffers,
    projector: merger::Buffers,
    buffers: Buffers,
    state: State,
    job: Option<Job>,
}

struct Job {
    prepared: Prepared,
    images: Vec<Patches>,
    clips: Vec<Vec<f32>>,
    /// The tokens of the images and clips seen so far, in order, `tokens × hidden`.
    image_tokens: Vec<f32>,
    audio_tokens: Vec<f32>,
    /// Stages done: images, then clips, then packs.
    done: usize,
    /// The answers of the packs read, in the questions' order.
    answers: Map<String, Value>,
}

impl Job {
    fn steps(&self) -> usize {
        self.images.len() + self.clips.len() + self.prepared.packs.len()
    }
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let config = Config::read(&store.take_blob(CONFIG)?)?;
        let calibration = Calibration::read(&store.take_blob(CALIBRATION)?)?;
        let model = Model::load(store)?;
        let vision = Vision::load(store)?;
        let team = Team::global();
        let threads = team.threads();
        let hearing = match config.model_type {
            Variant::Vision => None,
            Variant::Omni => Some(Hearing {
                encoder: Encoder::load(store, "audio", AUDIO)?,
                buffers: AUDIO.buffers(MAX_MEL_FRAMES, BLOCK_FRAMES, threads),
                spectrogram: MelSpectrogram::new(MEL),
            }),
        };
        Ok(Self {
            variant: config.model_type,
            prompter: Prompter::new(Tokenizer::take(store)?, &config)?,
            config,
            model,
            calibration,
            vision,
            hearing,
            tower: TOWER.buffers(IMAGE.max_patches(), threads),
            projector: merger::Buffers::new(&PROJECTOR, IMAGE.max_patches()),
            buffers: LANGUAGE.buffers(MAX_ROWS, threads),
            state: State::new(&LANGUAGE, MAX_ROWS, MAX_ROWS),
            team,
            job: None,
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {
        self.job = None;
    }
}

impl Engine {
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// An image's patches, scaled down to the checkpoint's budget first.
    ///
    /// # Errors
    /// When the image is empty or too narrow.
    pub fn patches(&self, image: &Pixels) -> Result<Patches, Error> {
        media::patches(image, self.config.d1_vision_max_pixels)
    }

    /// A clip's features, `frames × mels`.
    ///
    /// # Errors
    /// When the model reads no audio, or the clip is empty or too long.
    pub fn features(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let hearing = self.hearing.as_ref().ok_or_else(|| self.hears_nothing())?;
        media::features(&hearing.spectrogram, samples)
    }

    /// Each patch's embedding, RMS-normed, `patches × width`.
    pub fn embed(&mut self, patches: &Patches) -> Vec<f32> {
        let (tower, buffers) = (&self.vision.tower, &self.tower);
        self.team
            .run(|ctx| tower.embed(ctx, buffers, &patches.values));
        let len = patches.grid.patches() * TOWER.hidden;
        buffers.encoder.hidden.read()[..len].to_vec()
    }

    /// The tower's layers over `embedded` (`embed`'s output).
    pub fn encode(&mut self, embedded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..embedded.len()].copy_from_slice(embedded);
        let (tower, buffers) = (&self.vision.tower, &self.tower);
        self.team.run(|ctx| tower.encode(ctx, buffers, grid));
        buffers.encoder.hidden.read()[..embedded.len()].to_vec()
    }

    /// An image's tokens from `encoded` (`encode`'s output), `tokens × hidden`.
    pub fn project(&mut self, encoded: &[f32], grid: Grid) -> Vec<f32> {
        self.tower.encoder.hidden.get_mut()[..encoded.len()].copy_from_slice(encoded);
        let (projector, tower, buffers) = (&self.vision.projector, &self.tower, &self.projector);
        self.team
            .run(|ctx| projector.forward(ctx, tower, buffers, grid));
        self.image_tokens(grid)
    }

    /// An image's patches through the tower and the projector, in one team run: its tokens.
    pub fn see(&self, patches: &Patches) -> Vec<f32> {
        let (vision, grid) = (&self.vision, patches.grid);
        let (tower, projector) = (&self.tower, &self.projector);
        self.team.run(|ctx| {
            vision.tower.embed(ctx, tower, &patches.values);
            vision.tower.encode(ctx, tower, grid);
            vision.projector.forward(ctx, tower, projector, grid);
        });
        self.image_tokens(grid)
    }

    fn image_tokens(&self, grid: Grid) -> Vec<f32> {
        let len = IMAGE.tokens(grid.patches()) * HIDDEN;
        self.projector.tokens.read()[..len].to_vec()
    }

    /// A clip's tokens from its `features`, `tokens × hidden`.
    ///
    /// # Errors
    /// When the model reads no audio.
    pub fn hear(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        let missing = self.hears_nothing();
        let hearing = self.hearing.as_mut().ok_or(missing)?;
        hearing.buffers.clear();
        let (encoder, buffers) = (&hearing.encoder, &hearing.buffers);
        self.team.run(|ctx| encoder.forward(ctx, buffers, features));
        let tokens = clip_tokens(features);
        Ok(buffers.tokens.read()[..tokens * HIDDEN].to_vec())
    }

    fn hears_nothing(&self) -> Error {
        Error::Request(format!(
            "{} reads no audio: d1-Omni-2B does",
            self.variant.name()
        ))
    }

    /// `request`'s questions packed after its state and media, the images `image_tokens` and
    /// the clips `clip_tokens` tokens each.
    ///
    /// # Errors
    /// When a question fails its check, a box question has not exactly one image, or a pack
    /// is longer than `MAX_ROWS`.
    pub fn prepare(
        &self,
        request: &Request,
        image_tokens: &[usize],
        clip_tokens: &[usize],
    ) -> Result<Prepared, Error> {
        let questions = request.questions(self.config.d1_prompt)?;
        if questions.iter().any(|(_, question)| question.kind.boxed()) && image_tokens.len() != 1 {
            return Err(Error::Request(format!(
                "a box or boxes question reads exactly one picture; this request has {}",
                image_tokens.len()
            )));
        }
        let images = image_tokens
            .iter()
            .map(|&count| Prompter::markup(prompt::IMAGE, count));
        let clips = clip_tokens
            .iter()
            .map(|&count| Prompter::markup(prompt::AUDIO, count));
        let markup: String = images.chain(clips).collect();
        let prefix = self.prompter.prefix(&request.state, &markup);
        let header = Prompter::header(&request.state);
        let mut tables = self.prompter.tables();
        let suffixes = questions
            .iter()
            .map(|(_, question)| self.prompter.suffix(question, &mut tables))
            .collect::<Result<Vec<Suffix>, Error>>()?;
        let packs = suffixes
            .chunks(self.config.d1_set_size)
            .enumerate()
            .map(|(index, pack)| {
                let suffixes: Vec<&Suffix> = pack.iter().collect();
                let packed = self.prompter.pack(&prefix, header, &suffixes)?;
                if packed.ids.len() > MAX_ROWS {
                    return Err(Error::Request(format!(
                        "a pack of the request's questions has {} tokens, over the {MAX_ROWS} \
                         this runtime reads at once; nothing was truncated",
                        packed.ids.len()
                    )));
                }
                let reads = pack
                    .iter()
                    .any(|suffix| suffix.readouts.contains(&Readout::Pointer));
                let positions = reads.then(|| {
                    let prompt = format!("{prefix}{}", pack[0].text);
                    let state = &request.state;
                    StatePositions::new(&self.prompter, state, &prefix, &prompt, &packed.spans)
                });
                Ok(Pack {
                    packed,
                    first: index * self.config.d1_set_size,
                    readouts: pack.iter().map(|suffix| suffix.readouts.clone()).collect(),
                    positions,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Prepared {
            state: request.state.clone(),
            questions,
            tables,
            packs,
        })
    }

    /// A pack through the language model, its image rows `image_tokens` and audio rows
    /// `audio_tokens` in order: every row's final-normed state, `rows × hidden`.
    pub fn read(
        &mut self,
        packed: &Packed,
        image_tokens: &[f32],
        audio_tokens: &[f32],
    ) -> Vec<f32> {
        let rows = packed.ids.len();
        let input = &mut self.buffers.hidden.get_mut()[..rows * HIDDEN];
        let mut images = image_tokens.as_chunks::<HIDDEN>().0.iter();
        let mut clips = audio_tokens.as_chunks::<HIDDEN>().0.iter();
        for (&id, embedding) in packed.ids.iter().zip(input.as_chunks_mut::<HIDDEN>().0) {
            let token = match id {
                IMAGE_TOKEN => images.next(),
                AUDIO_TOKEN => clips.next(),
                _ => None,
            };
            match token {
                Some(token) => embedding.copy_from_slice(token),
                None => self
                    .model
                    .head
                    .matrix()
                    .dequantize_row(id as usize, embedding),
            }
        }
        let positions: Vec<Position> = (0..rows).map(Position::uniform).collect();
        let call = Rows {
            first: 0,
            positions: &positions,
            mask: Mask::Groups {
                state: packed.state,
                ends: &packed.ends,
            },
        };
        self.state.clear();
        let (stack, state, buffers) = (&self.model.stack, &self.state, &self.buffers);
        self.team
            .run(|ctx| stack.forward(ctx, state, buffers, &call));
        let mut normed = vec![0.0; rows * HIDDEN];
        let hidden = &self.buffers.hidden.read()[..rows * HIDDEN];
        rms_norm(hidden, self.model.norm.values(), NORM_EPS, &mut normed);
        normed
    }

    /// The answers of pack `index` of `prepared` from its rows' final-normed states `normed`.
    ///
    /// # Errors
    /// When a read's span leaves the state.
    pub fn answers(
        &self,
        prepared: &Prepared,
        index: usize,
        normed: &[f32],
    ) -> Result<Vec<(String, Value)>, Error> {
        let pack = &prepared.packs[index];
        let row = |at: usize| &normed[at * HIDDEN..(at + 1) * HIDDEN];
        let boxing = Boxing {
            grid: self.prompter.edge_grid(),
            edges: self.config.d1_box_edges,
        };
        let keys = pack.positions.as_ref().map(StatePositions::keys);
        let asked = pack.readouts.iter().zip(&pack.packed.slots).enumerate();
        asked
            .map(|(offset, (readouts, slots))| {
                let (name, question) = &prepared.questions[pack.first + offset];
                let temperature = if question.kind.typed() {
                    1.0
                } else {
                    self.calibration.temperature(&question.kind)
                };
                let outputs = readouts
                    .iter()
                    .zip(slots)
                    .map(|(readout, &slot)| match *readout {
                        Readout::Code(table) => {
                            let table = &prepared.tables.0[table];
                            Output::Probabilities(self.probabilities(table, row(slot), temperature))
                        }
                        Readout::Pointer => {
                            let keys = keys
                                .as_ref()
                                .expect("a pack with a read keeps its positions");
                            Output::Positions(pointed(row(slot), keys, row))
                        }
                    });
                let outputs: Vec<Output> = outputs.collect();
                let positions = pack.positions.as_ref();
                let answer =
                    answer::answer(question, &outputs, &self.calibration, positions, boxing)?;
                Ok((name.clone(), answer))
            })
            .collect()
    }

    /// The options' probabilities of `table` at a slot's final-normed state `normed`: each
    /// option's best form's logit, over `temperature`, through a softmax.
    fn probabilities(&self, table: &Table, normed: &[f32], temperature: f64) -> Vec<f64> {
        let head = self.model.head.matrix();
        let mut row = vec![0.0; HIDDEN];
        let logits: Vec<f32> = table
            .tokens
            .iter()
            .map(|&token| {
                head.dequantize_row(token as usize, &mut row);
                dot_f32(&row, normed)
            })
            .collect();
        let scores: Vec<f64> = table
            .options
            .iter()
            .map(|forms| {
                let best = forms
                    .iter()
                    .map(|&form| logits[form])
                    .fold(f32::NEG_INFINITY, f32::max);
                f64::from(best) / temperature
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
        let total: f64 = exponentials.iter().sum();
        exponentials.iter().map(|value| value / total).collect()
    }

    /// Starts answering `request` about RGB or RGBA `images` and mono 16 kHz `clips`.
    ///
    /// # Errors
    /// When there are too many images or clips, an image is empty or too narrow, a clip empty
    /// or too long, the model reads no audio and there is some, or the request is invalid.
    pub fn start(
        &mut self,
        request: &Request,
        images: &[Pixels],
        clips: &[Vec<f32>],
    ) -> Result<Plan, Error> {
        self.job = None;
        if images.len() > MAX_IMAGES || clips.len() > MAX_CLIPS {
            return Err(Error::Request(format!(
                "{} images and {} clips: at most {MAX_IMAGES} and {MAX_CLIPS}",
                images.len(),
                clips.len()
            )));
        }
        let images = images
            .iter()
            .map(|image| self.patches(image))
            .collect::<Result<Vec<_>, _>>()?;
        let clips = clips
            .iter()
            .map(|clip| self.features(clip))
            .collect::<Result<Vec<_>, _>>()?;
        let image_tokens: Vec<usize> = images
            .iter()
            .map(|image| IMAGE.tokens(image.grid.patches()))
            .collect();
        let clip_tokens: Vec<usize> = clips.iter().map(|features| clip_tokens(features)).collect();
        let prepared = self.prepare(request, &image_tokens, &clip_tokens)?;
        let plan = Plan {
            model: self.variant.name(),
            images: images
                .iter()
                .map(|image| [image.grid.rows, image.grid.columns])
                .collect(),
            clips: clip_tokens,
            packs: prepared
                .packs
                .iter()
                .map(|pack| pack.packed.ids.len())
                .collect(),
            questions: prepared.questions.len(),
        };
        self.job = Some(Job {
            prepared,
            images,
            clips,
            image_tokens: Vec::new(),
            audio_tokens: Vec::new(),
            done: 0,
            answers: Map::new(),
        });
        Ok(plan)
    }

    /// Runs the next stage: the next image, clip or pack.
    ///
    /// # Errors
    /// When no request is in progress, or a read's span leaves the state.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let mut job = self
            .job
            .take()
            .ok_or_else(|| Error::Request("no request in progress".into()))?;
        let (images, clips) = (job.images.len(), job.clips.len());
        let stage = if job.done < images {
            job.image_tokens.extend(self.see(&job.images[job.done]));
            Stage::See
        } else if job.done < images + clips {
            let tokens = self.hear(&job.clips[job.done - images])?;
            job.audio_tokens.extend(tokens);
            Stage::Hear
        } else {
            let index = job.done - images - clips;
            let packed = &job.prepared.packs[index].packed;
            let normed = self.read(packed, &job.image_tokens, &job.audio_tokens);
            job.answers
                .extend(self.answers(&job.prepared, index, &normed)?);
            Stage::Read
        };
        job.done += 1;
        let response = (job.done == job.steps()).then(|| Response {
            answers: std::mem::take(&mut job.answers),
            usage: Usage {
                input_tokens: job
                    .prepared
                    .packs
                    .iter()
                    .map(|pack| pack.packed.ids.len())
                    .sum(),
                output_tokens: 0,
            },
        });
        let progress = Progress {
            stage: stage.name(),
            done: job.done,
            steps: job.steps(),
            response,
        };
        if progress.response.is_none() {
            self.job = Some(job);
        }
        Ok(progress)
    }
}

/// The tokens of a clip of `features`.
fn clip_tokens(features: &[f32]) -> usize {
    AUDIO.tokens(features.len() / MEL.mels)
}

/// A read slot's logits over the state's positions: its state against each key's, over the
/// width's square root, minus infinity off the keys.
fn pointed<'rows>(slot: &[f32], keys: &[bool], row: impl Fn(usize) -> &'rows [f32]) -> Vec<f32> {
    let scale = (HIDDEN as f32).sqrt();
    let logits = keys.iter().enumerate().map(|(at, &key)| {
        if key {
            dot_f32(slot, row(at)) / scale
        } else {
            f32::NEG_INFINITY
        }
    });
    logits.collect()
}
