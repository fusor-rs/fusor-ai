//! Reading a page: `start` takes the image as grayscale; each `step` does one stage
//! (`pipeline::Stage`), the recognizers and the math reader one batch of lines at a time, until
//! the page is read. `Networks` runs each stage on its own, for checks against the reference.

use std::collections::VecDeque;

use bundle::{Error, Store};
use kernels::numpy::{argmax, mean};
use media::image::Pixels;
use serde::{Deserialize, Serialize};
use team::Team;

use crate::assemble::{Assembler, Page, Placed, Reading, Recognized, Token};
use crate::crops::{Crop, LineBox, PAD_X, RECOGNIZER_BATCHING, crop, is_dark, pad, width_batches};
use crate::layout::rounded;
use crate::lines::{Detection, adaptive_side, decode, detector_input, to_page};
use crate::lm::{Steps, beam_search};
use crate::math::encoder::HEIGHT as MATH_HEIGHT;
use crate::math::{self, MATH_BATCHING, MATH_MIN, MATH_MIN_CONFIDENCE};
use crate::model::{Group, Model};
use crate::page::{VERTICAL, orientation_input, preview, skew, straightened, unrotate, upright};
use crate::pipeline::Stage;
use crate::plane::Scratch;
use crate::reading::features;
use crate::recognizer::{BEST, Batch, LINE_HEIGHT, Outputs};

/// A turn is taken only above this probability: a wrong turn loses the page.
const ORIENT_MIN: f32 = 0.95;
/// A line read under this confidence gets no text.
const MIN_CONFIDENCE: f64 = 0.5;
/// The script classifier routes a line to one recognizer above this probability; under it the
/// base recognizer reads the line too, and the more confident reading wins.
const ROUTE_SURE: f32 = 0.95;
const QUARTER_TURN_DEGREES: usize = 90;

/// How a page is read.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Options {
    /// Read lines with math as LaTeX.
    #[serde(default)]
    pub math: bool,
}

/// One step done, and the page once it is read.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    /// Batches of lines read, of those planned so far.
    pub batches: usize,
    pub planned: usize,
    pub page: Option<Page>,
}

/// The page turned upright and straightened: its quarter turns undone (`VERTICAL` for a page
/// set in vertical columns) and the degrees of skew undone.
#[derive(Debug, Clone)]
pub struct Upright {
    pub page: Pixels,
    pub turn: usize,
    pub skew: f64,
}

/// The networks and the team they run on.
#[derive(Clone, Copy)]
pub struct Networks<'engine> {
    pub model: &'engine Model,
    pub team: &'engine Team,
    pub scratch: &'engine Scratch,
}

pub struct Engine {
    model: Model,
    team: Team,
    scratch: Scratch,
    job: Option<Job>,
}

struct Job {
    options: Options,
    gray: Pixels,
    state: State,
}

enum State {
    Orient,
    Detect(Upright),
    Classify(Upright, Vec<Detection>),
    Read(Box<Lines>),
    Done,
}

/// The lines of a page being read: their crops, the batches left, and the best reading of
/// each so far.
pub struct Lines {
    upright: Upright,
    boxes: Vec<LineBox>,
    crops: Vec<Crop>,
    /// The base recognizer's tall crops (its window over one band), else empty.
    tall: Vec<Crop>,
    work: VecDeque<Work>,
    math: VecDeque<Vec<usize>>,
    math_crops: Vec<Option<Crop>>,
    readings: Vec<Reading>,
    best: Vec<f64>,
    batches: usize,
    planned: usize,
}

/// Crops of one height padded to one width, and their widths.
pub struct Padded {
    pub pixels: Vec<f32>,
    pub widths: Vec<usize>,
    pub height: usize,
    pub width: usize,
}

impl Padded {
    fn new(crops: &[&Crop], height: usize) -> Self {
        let (pixels, width) = pad(crops, height);
        let widths = crops.iter().map(|crop| crop.width).collect();
        Self {
            pixels,
            widths,
            height,
            width,
        }
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch {
            pixels: &self.pixels,
            widths: &self.widths,
            height: self.height,
            width: self.width,
        }
    }
}

impl Lines {
    /// The recognizer batches left, in order.
    pub fn work(&self) -> impl Iterator<Item = &Work> {
        self.work.iter()
    }

    /// The math batches left, in order.
    pub fn math_batches(&self) -> impl Iterator<Item = &[usize]> {
        self.math.iter().map(Vec::as_slice)
    }

    /// A math batch's crops, as the encoder reads them.
    pub fn math_input(&self, batch: &[usize]) -> Padded {
        let chosen: Vec<&Crop> = batch
            .iter()
            .map(|&line| {
                self.math_crops[line]
                    .as_ref()
                    .expect("a math line has its crop")
            })
            .collect();
        Padded::new(&chosen, MATH_HEIGHT)
    }
}

/// A batch of lines for one recognizer.
#[derive(Debug, Clone)]
pub struct Work {
    pub group: usize,
    pub lines: Vec<usize>,
}

impl bundle::Engine for Engine {
    fn new(store: &mut Store) -> Result<Self, Error> {
        let team = Team::global();
        Ok(Self {
            model: Model::load(store)?,
            scratch: Scratch::new(team.threads()),
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
    pub fn networks(&self) -> Networks<'_> {
        Networks {
            model: &self.model,
            team: &self.team,
            scratch: &self.scratch,
        }
    }

    /// Starts reading `image` (any channels: turned grayscale as OpenCV does it).
    pub fn start(&mut self, image: &Pixels, options: Options) {
        self.job = Some(Job {
            options,
            gray: image.gray(),
            state: State::Orient,
        });
    }

    /// Does the next stage, or the next batch of lines.
    ///
    /// # Errors
    /// Without a page started.
    pub fn step(&mut self) -> Result<Progress, Error> {
        let networks = Networks {
            model: &self.model,
            team: &self.team,
            scratch: &self.scratch,
        };
        let job = self
            .job
            .as_mut()
            .ok_or_else(|| Error::Request("no page is being read: start one".into()))?;
        let state = std::mem::replace(&mut job.state, State::Done);
        let (stage, next) = match state {
            State::Orient => (Stage::Orient, State::Detect(networks.upright(&job.gray))),
            State::Detect(upright) => {
                let detections = networks.detect(&upright.page);
                (Stage::Detect, State::Classify(upright, detections))
            }
            State::Classify(upright, detections) => {
                let lines = networks.classify(upright, &detections, job.options);
                (Stage::Classify, State::Read(Box::new(lines)))
            }
            State::Read(mut lines) => {
                if let Some(work) = lines.work.pop_front() {
                    networks.recognize(&mut lines, &work);
                    (Stage::Recognize, State::Read(lines))
                } else if let Some(batch) = lines.math.pop_front() {
                    networks.read_math(&mut lines, &batch);
                    (Stage::Math, State::Read(lines))
                } else {
                    let (batches, planned) = (lines.batches, lines.planned);
                    let progress = Progress {
                        stage: bundle::Stages::name(Stage::Assemble),
                        batches,
                        planned,
                        page: Some(networks.assemble(*lines)),
                    };
                    self.job = None;
                    return Ok(progress);
                }
            }
            State::Done => return Err(Error::Request("the page is read: start another".into())),
        };
        let (batches, planned) = match &next {
            State::Read(lines) => (lines.batches, lines.planned),
            _ => (0, 0),
        };
        job.state = next;
        Ok(Progress {
            stage: bundle::Stages::name(stage),
            batches,
            planned,
            page: None,
        })
    }

    /// Reads `image` through every stage.
    ///
    /// # Errors
    /// As `step`.
    pub fn read(&mut self, image: &Pixels, options: Options) -> Result<Page, Error> {
        self.start(image, options);
        loop {
            if let Some(page) = self.step()?.page {
                return Ok(page);
            }
        }
    }
}

impl Networks<'_> {
    /// The orientation network's class probabilities of the preview straightened by `degrees`.
    pub fn orientation(&self, preview: &Pixels, degrees: f64) -> [f32; crate::orient::CLASSES] {
        let input = orientation_input(&straightened(preview, degrees));
        self.model
            .orientation
            .forward(self.team, self.scratch, &input)
    }

    /// The page turned upright and straightened.
    pub fn upright(&self, gray: &Pixels) -> Upright {
        let preview = preview(gray);
        let degrees = skew(&preview);
        let probabilities = self.orientation(&preview, degrees);
        let best = argmax(&probabilities);
        let turn = if probabilities[best] > ORIENT_MIN {
            best
        } else {
            0
        };
        let (page, skew) = upright(gray, &preview, turn, degrees);
        Upright { page, turn, skew }
    }

    /// The detector's maps of `page` at `long_side`, their grid and the input's scale.
    pub fn detector_maps(
        &self,
        page: &Pixels,
        long_side: usize,
    ) -> (Vec<f32>, kernels::conv::Grid, f64) {
        let (ink, grid, scale) = detector_input(page, long_side);
        let maps = self
            .model
            .detector
            .forward(self.team, self.scratch, &ink, grid);
        let half = kernels::conv::Grid {
            height: grid.height / 2,
            width: grid.width / 2,
            ..grid
        };
        (maps, half, scale)
    }

    /// The page's text lines, in page pixels.
    pub fn detect(&self, page: &Pixels) -> Vec<Detection> {
        let long_side = self.model.long_side;
        let (maps, grid, mut scale) = self.detector_maps(page, long_side);
        let mut detections = decode(&maps, grid);
        if let Some(side) = adaptive_side(&detections, long_side) {
            let (maps, grid, larger) = self.detector_maps(page, side);
            (detections, scale) = (decode(&maps, grid), larger);
        }
        to_page(&mut detections, scale, page.size);
        detections
    }
}

impl Networks<'_> {
    /// The lines' crops, each routed to its recognizers in batches of similar width, and the
    /// lines with math when `options.math`.
    pub fn classify(&self, upright: Upright, detections: &[Detection], options: Options) -> Lines {
        let page = &upright.page;
        let boxes: Vec<LineBox> = detections.iter().map(|detection| detection.line).collect();
        let dark: Vec<bool> = boxes.iter().map(|&line| is_dark(page, line)).collect();
        let crops: Vec<Crop> = boxes
            .iter()
            .zip(&dark)
            .map(|(&line, &dark)| crop(page, line, (LINE_HEIGHT, 1), dark))
            .collect();
        let window = self.model.window;
        let tall: Vec<Crop> = match window {
            1 => Vec::new(),
            _ => boxes
                .iter()
                .zip(&dark)
                .map(|(&line, &dark)| crop(page, line, (LINE_HEIGHT, window), dark))
                .collect(),
        };
        let work = self.plan(&crops, upright.turn == VERTICAL);
        let (math, math_crops) = match options.math {
            true => self.plan_math(page, (&boxes, &dark), &crops),
            false => (VecDeque::new(), (0..boxes.len()).map(|_| None).collect()),
        };
        let planned = work.len() + math.len();
        Lines {
            readings: vec![Reading::default(); boxes.len()],
            best: vec![-1.0; boxes.len()],
            upright,
            boxes,
            crops,
            tall,
            work: work.into(),
            math,
            math_crops,
            batches: 0,
            planned,
        }
    }

    /// The math lines' batches, and the math crop of each (none for a line without math).
    fn plan_math(
        &self,
        page: &Pixels,
        (boxes, dark): (&[LineBox], &[bool]),
        crops: &[Crop],
    ) -> (VecDeque<Vec<usize>>, Vec<Option<Crop>>) {
        let flags = if crops.is_empty() {
            Vec::new()
        } else {
            self.math_flags(crops)
        };
        let lines: Vec<usize> = (0..flags.len()).filter(|&line| flags[line]).collect();
        let mut math_crops: Vec<Option<Crop>> = (0..boxes.len()).map(|_| None).collect();
        for &line in &lines {
            math_crops[line] = Some(crop(page, boxes[line], (MATH_HEIGHT, 1), dark[line]));
        }
        let widths: Vec<usize> = lines
            .iter()
            .filter_map(|&line| math_crops[line].as_ref())
            .map(|crop| crop.width)
            .collect();
        let batches = width_batches(&widths, MATH_BATCHING)
            .into_iter()
            .map(|batch| batch.into_iter().map(|index| lines[index]).collect())
            .collect();
        (batches, math_crops)
    }

    /// The recognizer batches: a vertical page's lines all to the CJK recognizer; else each
    /// line to its script's group, and to the base recognizer too when the routing is unsure.
    fn plan(&self, crops: &[Crop], vertical: bool) -> Vec<Work> {
        let groups = &self.model.groups;
        let cjk = groups.iter().position(|group| group.name == "cjk");
        let everyone: Vec<usize> = (0..crops.len()).collect();
        let selections: Vec<(usize, Vec<usize>)> = match cjk {
            Some(cjk) if vertical => vec![(cjk, everyone)],
            _ if crops.is_empty() => vec![(0, everyone)],
            _ => {
                let (routes, sure) = self.route(crops);
                let base = (0..crops.len())
                    .filter(|&line| routes[line] == 0 || !sure[line])
                    .collect();
                let mut selections = vec![(0, base)];
                for group in 1..groups.len() {
                    selections.push((
                        group,
                        (0..crops.len())
                            .filter(|&line| routes[line] == group)
                            .collect(),
                    ));
                }
                selections
            }
        };
        let mut work = Vec::new();
        for (group, lines) in selections {
            let widths: Vec<usize> = lines.iter().map(|&line| crops[line].width).collect();
            for batch in width_batches(&widths, RECOGNIZER_BATCHING) {
                work.push(Work {
                    group,
                    lines: batch.into_iter().map(|index| lines[index]).collect(),
                });
            }
        }
        work
    }

    /// Each line's group (by the script classifier; the base one for a script without its
    /// recognizer) and whether the classifier is sure of it.
    pub fn route(&self, crops: &[Crop]) -> (Vec<usize>, Vec<bool>) {
        let names = &self.model.script_groups;
        let mut routes = vec![0; crops.len()];
        let mut sure = vec![true; crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for batch in width_batches(&widths, RECOGNIZER_BATCHING) {
            let logits = self.classify_batch(&self.model.script, crops, &batch);
            for (&line, logits) in batch.iter().zip(logits.chunks_exact(names.len())) {
                (routes[line], sure[line]) = self.route_line(logits);
            }
        }
        (routes, sure)
    }

    /// A line's group from its script logits, and whether the classifier is sure of it.
    fn route_line(&self, logits: &[f32]) -> (usize, bool) {
        let mut probabilities = logits.to_vec();
        kernels::ops::softmax(&mut probabilities);
        let best = argmax(&probabilities);
        let name = &self.model.script_groups[best];
        let group = self
            .model
            .groups
            .iter()
            .position(|group| group.name == *name);
        // A script without a recognizer of its own is the base recognizer's.
        (group.unwrap_or(0), probabilities[best] > ROUTE_SURE)
    }

    /// Whether each line holds math.
    pub fn math_flags(&self, crops: &[Crop]) -> Vec<bool> {
        let mut flags = vec![false; crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for batch in width_batches(&widths, RECOGNIZER_BATCHING) {
            let logits = self.classify_batch(&self.model.math.classifier, crops, &batch);
            for (&line, &logit) in batch.iter().zip(&logits) {
                flags[line] = kernels::ops::sigmoid(logit) > MATH_MIN;
            }
        }
        flags
    }

    fn classify_batch(
        &self,
        classifier: &crate::classifier::Classifier,
        crops: &[Crop],
        batch: &[usize],
    ) -> Vec<f32> {
        let chosen: Vec<&Crop> = batch.iter().map(|&line| &crops[line]).collect();
        classifier.forward(
            self.team,
            self.scratch,
            Padded::new(&chosen, LINE_HEIGHT).batch(),
        )
    }
}

impl Networks<'_> {
    /// A batch's crops as its recognizer reads them: the base recognizer's tall ones.
    pub fn recognizer_input(&self, lines: &Lines, work: &Work) -> Padded {
        let tall = work.group == 0 && !lines.tall.is_empty();
        let (source, height) = if tall {
            (&lines.tall, LINE_HEIGHT * self.model.window)
        } else {
            (&lines.crops, LINE_HEIGHT)
        };
        let chosen: Vec<&Crop> = work.lines.iter().map(|&line| &source[line]).collect();
        Padded::new(&chosen, height)
    }

    /// One batch through its recognizer; each line's reading kept when it is the most
    /// confident so far.
    pub fn recognize(&self, lines: &mut Lines, work: &Work) {
        let group = &self.model.groups[work.group];
        let padded = self.recognizer_input(lines, work);
        let outputs = group
            .recognizer
            .forward(self.team, self.scratch, padded.batch());
        let stride = padded.width as f64 / outputs.steps.max(1) as f64;
        for (index, &line) in work.lines.iter().enumerate() {
            let placement = Placement {
                line: lines.boxes[line],
                crop_width: lines.crops[line].width,
                stride,
            };
            let reading = read_line(group, work.group > 0, (&outputs, index), placement);
            if reading.confidence <= lines.best[line] {
                continue;
            }
            lines.best[line] = reading.confidence;
            lines.readings[line] = if reading.confidence >= MIN_CONFIDENCE {
                reading
            } else {
                Reading {
                    tokens: Vec::new(),
                    text: String::new(),
                    ..reading
                }
            };
        }
        lines.batches += 1;
    }

    /// One batch of math lines read; a confident reading with math replaces its line, as one
    /// token.
    pub fn read_math(&self, lines: &mut Lines, batch: &[usize]) {
        let math = &self.model.math;
        let padded = lines.math_input(batch);
        let memory = math
            .encoder
            .forward(self.team, self.scratch, padded.batch());
        let readings = math::decode(&math.decoder, self.team, self.scratch, &memory);
        for (&line, reading) in batch.iter().zip(readings) {
            let text = math.tokens.detokenize(&reading.ids);
            if text.contains('$') && reading.confidence() >= MATH_MIN_CONFIDENCE {
                let [x0, _, x1, _] = lines.boxes[line];
                let token = Placed {
                    token: Token::Text(text.clone()),
                    x: f64::from(x0 + x1) / 2.0,
                };
                lines.readings[line] = Reading {
                    text,
                    confidence: 1.0,
                    tokens: vec![token],
                };
            }
        }
        lines.batches += 1;
    }

    /// The page from its read lines, its boxes back on the page as it was.
    pub fn assemble(&self, lines: Lines) -> Page {
        let Upright { page, turn, skew } = lines.upright;
        let size = (page.size.width, page.size.height);
        let vertical = turn == VERTICAL;
        let assembler = Assembler::new(&self.model.groups[0].vocab, &lines.readings);
        let recognized = Recognized {
            width: size.0,
            height: size.1,
            boxes: lines.boxes,
            readings: lines.readings,
        };
        let assembled = assembler.assemble(&recognized, vertical, |boxes, texts| {
            self.model
                .order
                .forward(self.team, &features(boxes, size, texts))
        });
        let mut page = Page {
            source: "ocr",
            rotation: if vertical {
                0
            } else {
                QUARTER_TURN_DEGREES * turn
            },
            skew: rounded(skew, 2),
            vertical,
            lines: assembled.lines,
            blocks: assembled.blocks,
            markdown: assembled.markdown,
        };
        let boxes = page
            .lines
            .iter_mut()
            .map(|line| &mut line.bbox)
            .chain(page.blocks.iter_mut().map(|block| &mut block.bbox));
        for bbox in boxes {
            if skew != 0.0 {
                *bbox = unrotate(*bbox, skew, size);
            }
            if vertical {
                let height = size.1 as f64;
                *bbox = [height - bbox[3], bbox[0], height - bbox[1], bbox[2]];
            }
        }
        page
    }
}

/// Where a line's steps fall on the page: its box, the width of its band crop, and the
/// recognizer's input columns per step.
#[derive(Clone, Copy)]
struct Placement {
    line: LineBox,
    crop_width: usize,
    stride: f64,
}

impl Placement {
    /// The page x of step `step`'s centre.
    fn x(&self, step: usize) -> f64 {
        let [x0, y0, x1, y1] = self.line.map(f64::from);
        let height = (y1 - y0).max(1.0);
        let left = (x0 - PAD_X * height).floor();
        let width = (x1 + PAD_X * height).ceil() - left;
        let scale = width / self.crop_width as f64;
        left + (step as f64 + 0.5) * self.stride * scale
    }
}

/// Line `index`'s reading from its recognizer's outputs: the greedy CTC path (repeats merged,
/// blanks dropped) and its confidence, the mean probability of the kept steps; reread with the
/// group's language model when unsure. `characters`: a group recognizer, whose tokens are text.
fn read_line(
    group: &Group,
    characters: bool,
    (outputs, index): (&Outputs, usize),
    placement: Placement,
) -> Reading {
    let first = index * outputs.steps;
    let length = outputs.lengths[index];
    let ids = &outputs.ids[first..first + length];
    let kept: Vec<usize> = (0..length)
        .filter(|&step| ids[step] != 0 && (step == 0 || ids[step] != ids[step - 1]))
        .collect();
    let peaks: Vec<f32> = kept
        .iter()
        .map(|&step| outputs.confidences[first + step])
        .collect();
    let confidence = if peaks.is_empty() {
        0.0
    } else {
        f64::from(mean(&peaks))
    };
    let mut classes: Vec<u32> = kept.iter().map(|&step| ids[step]).collect();
    let mut steps = kept;
    if let Some(lm) = group
        .lm
        .as_ref()
        .filter(|lm| MIN_CONFIDENCE <= confidence && confidence < lm.gate)
    {
        let best = first * BEST..(first + length) * BEST;
        let line = Steps {
            blank: &outputs.blanks[first..first + length],
            classes: &outputs.top_ids[best.clone()],
            log_probabilities: &outputs.top_values[best],
            best: BEST,
        };
        (classes, steps) = beam_search(&line, &lm.ngram, lm.weights);
    }
    let tokens = classes.iter().zip(&steps).map(|(&class, &step)| {
        let token = if characters {
            Token::Text(group.vocab.token(class).to_string())
        } else {
            Token::Class(class)
        };
        Placed {
            token,
            x: placement.x(step),
        }
    });
    Reading {
        text: group.vocab.decode(&classes),
        confidence,
        tokens: tokens.collect(),
    }
}
