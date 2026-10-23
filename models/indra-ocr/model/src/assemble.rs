//! A page from its recognized lines: lines cut at wide gaps between characters (where the
//! detector merged cells or columns, or at a column gutter other lines' edges line up with),
//! leader lines dropped and lone list markers joined to their items, then reading order,
//! layout blocks and Markdown. A right-to-left page reads its lines right to left and orders
//! them on the page mirrored left to right.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

use crate::crops::LineBox;
use crate::layout::{self, Block, Lines, rounded};
use crate::order::Relations;
use crate::reading::{blocks_order, xy_cut, xy_cut_blocks};
use crate::rtl::{flip, has_rtl};
use crate::text::{is_alpha, is_rtl};
use crate::vocab::Vocab;

/// Gaps between characters' centres, in line heights, that cut a line: anywhere, in a table,
/// and at a column gutter (where `GUTTER_EDGES` other lines' edges line up with the gap's).
const SPLIT_GAP: f64 = 3.0;
const CELL_GAP: f64 = 1.8;
const GUTTER_GAP: f64 = 1.6;
const GUTTER_EDGES: usize = 3;
/// A character's half width, in line heights: a gap's edges, a cut piece's box.
const HALF_GLYPH: f64 = 0.35;
const LEADERS: &str = ".·…‥_-–—=";
const DASHES: &str = "-–—";
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";
/// A list marker and its item are at most this many line heights apart.
const MARKER_GAP: f32 = 2.5;
const MARKER_OVERLAP: f32 = 0.3;
/// The class logit that rules a class out.
const RULED_OUT: f32 = -1e4;
/// A numbered list marker: `1.`, `12)`, `a.`, `(3)`, `(b)`, `(iv)`.
static ENUMERATOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))$")
        .expect("the enumerator pattern compiles")
});

/// What a line's reading holds: a base-recognizer class, or text (a group recognizer's
/// character, or a math reading as a whole).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Class(u32),
    Text(String),
}

/// A token and the page x of its CTC peak.
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub token: Token,
    pub x: f64,
}

/// A recognized line: its text, its confidence (the mean CTC peak probability) and tokens.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reading {
    pub text: String,
    pub confidence: f64,
    pub tokens: Vec<Placed>,
}

/// A line of the page: its box (one decimal), text and confidence (four).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Line {
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
    pub text: String,
    pub conf: f64,
}

/// What reading the page made of it, as the reference's JSON writes it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Page {
    pub source: &'static str,
    pub rotation: usize,
    pub skew: f64,
    pub vertical: bool,
    pub lines: Vec<Line>,
    pub blocks: Vec<Block>,
    pub markdown: String,
}

/// Decodes tokens: the base vocabulary's classes, or characters in visual order.
pub struct Assembler<'vocab> {
    vocab: &'vocab Vocab,
    space: Option<u32>,
    /// The page reads right to left.
    rtl: bool,
}

/// A piece of a line: its box, text and tokens.
type Piece = (LineBox, String, Vec<Placed>);

impl<'vocab> Assembler<'vocab> {
    /// The assembler of a page whose lines read `readings`: right to left when most of their
    /// letters are.
    pub fn new(vocab: &'vocab Vocab, readings: &[Reading]) -> Self {
        let letters: Vec<char> = readings
            .iter()
            .flat_map(|reading| reading.text.chars())
            .filter(|&character| is_alpha(character))
            .collect();
        let right_to_left = letters.iter().filter(|&&letter| is_rtl(letter)).count();
        Self {
            vocab,
            space: vocab.space(),
            rtl: right_to_left > letters.len() - right_to_left,
        }
    }

    fn is_space(&self, token: &Token) -> bool {
        match token {
            Token::Class(class) => Some(*class) == self.space,
            Token::Text(text) => text == " ",
        }
    }

    /// The text of `tokens`: characters joined, runs of white space as one space, composed,
    /// right-to-left runs in logical order; or the base vocabulary's classes decoded.
    fn decode(&self, tokens: &[Placed]) -> String {
        let texts: Vec<&str> = tokens
            .iter()
            .filter_map(|placed| match &placed.token {
                Token::Text(text) => Some(text.as_str()),
                Token::Class(_) => None,
            })
            .collect();
        if texts.is_empty() {
            let classes: Vec<u32> = tokens
                .iter()
                .filter_map(|placed| match placed.token {
                    Token::Class(class) => Some(class),
                    Token::Text(_) => None,
                })
                .collect();
            return self.vocab.decode(&classes);
        }
        let joined = texts.concat();
        let text: String = joined
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .nfc()
            .collect();
        if has_rtl(&text) {
            flip(&text, self.rtl.then_some(true))
        } else {
            text
        }
    }

    /// The line cut where two neighbouring characters are over `gap` line heights apart, or
    /// on either side of a cut in `cuts`; the whole line when nothing cuts it.
    fn split(&self, line: LineBox, tokens: &[Placed], gap: f64, cuts: &[f64]) -> Vec<Piece> {
        let height = f64::from(line[3] - line[1]).max(1.0);
        let mut pieces: Vec<Vec<Placed>> = vec![Vec::new()];
        let mut previous: Option<f64> = None;
        for placed in tokens {
            if !self.is_space(&placed.token) {
                let x = placed.x;
                let cut = previous.is_some_and(|previous| {
                    x - previous > gap * height || cuts.iter().any(|&cut| previous < cut && cut < x)
                });
                if cut {
                    pieces.push(Vec::new());
                }
                previous = Some(x);
            }
            pieces
                .last_mut()
                .expect("there is a piece")
                .push(placed.clone());
        }
        if pieces.len() == 1 {
            return vec![(line, self.decode(tokens), tokens.to_vec())];
        }
        pieces
            .into_iter()
            .filter_map(|piece| {
                let text = self.decode(&piece).trim().to_string();
                let xs: Vec<f64> = piece
                    .iter()
                    .filter(|placed| !self.is_space(&placed.token))
                    .map(|placed| placed.x)
                    .collect();
                let first = xs.iter().copied().reduce(f64::min)?;
                let last = xs.iter().copied().reduce(f64::max)?;
                let left = f64::from(line[0]).max(first - HALF_GLYPH * height) as f32;
                let right = f64::from(line[2]).min(last + HALF_GLYPH * height) as f32;
                (!text.is_empty()).then_some(([left, line[1], right, line[3]], text, piece))
            })
            .collect()
    }
}

impl Assembler<'_> {
    /// Per line, the x of each gap between characters over `GUTTER_GAP` line heights that
    /// `GUTTER_EDGES` other lines' right edges (at its left side) or left edges (at its right)
    /// line up with.
    fn gutter_cuts(&self, boxes: &[LineBox], readings: &[Reading]) -> Vec<Vec<f64>> {
        boxes
            .iter()
            .zip(readings)
            .enumerate()
            .map(|(index, (line, reading))| {
                let height = f64::from(line[3] - line[1]).max(1.0);
                let xs: Vec<f64> = reading
                    .tokens
                    .iter()
                    .filter(|placed| !self.is_space(&placed.token))
                    .map(|placed| placed.x)
                    .collect();
                xs.windows(2)
                    .filter(|pair| pair[1] - pair[0] > GUTTER_GAP * height)
                    .filter(|pair| {
                        let (left, right) = (
                            (pair[0] + HALF_GLYPH * height) as f32,
                            (pair[1] - HALF_GLYPH * height) as f32,
                        );
                        let reach = (0.5 * height) as f32;
                        let edges = boxes.iter().enumerate().filter(|&(other, edges)| {
                            other != index
                                && ((edges[2] - left).abs() < reach
                                    || (edges[0] - right).abs() < reach)
                        });
                        edges.count() >= GUTTER_EDGES
                    })
                    .map(|pair| (pair[0] + pair[1]) / 2.0)
                    .collect()
            })
            .collect()
    }

    /// The lines without text of their own handled: a line of leader characters only dropped
    /// (three or more, a box three times wider than tall, or any but a lone dash), a lone list
    /// marker joined to the line that starts within `MARKER_GAP` line heights after it on its
    /// row (on its left on a right-to-left page), its tokens too on a left-to-right page.
    fn tidy(&self, boxes: &mut [LineBox], readings: &mut [Reading]) -> Vec<usize> {
        let mut keep: Vec<bool> = boxes
            .iter()
            .zip(readings.iter())
            .map(|(line, reading)| !is_leader_line(line, &reading.text))
            .collect();
        for index in 0..readings.len() {
            let marker = readings[index].text.trim().to_string();
            let is_marker = (marker.chars().count() == 1 && BULLETS.contains(&marker))
                || ENUMERATOR.is_match(&marker);
            if !keep[index] || !is_marker {
                continue;
            }
            let Some(item) = self.item_of(index, boxes, readings, &keep) else {
                continue;
            };
            let (marker_box, item_box) = (boxes[index], boxes[item]);
            let marker_tokens = readings[index].tokens.clone();
            let item_reading = &mut readings[item];
            let item_chars = item_reading
                .tokens
                .first()
                .is_some_and(|placed| matches!(placed.token, Token::Text(_)));
            let space = if item_chars {
                Some(Token::Text(" ".into()))
            } else {
                self.space.map(Token::Class)
            };
            let same_kind = marker_tokens
                .first()
                .is_some_and(|placed| matches!(placed.token, Token::Text(_)) == item_chars);
            if let Some(space) =
                space.filter(|_| !self.rtl && same_kind && !item_reading.tokens.is_empty())
            {
                let x = f64::from(marker_box[2] + item_box[0]) / 2.0;
                let mut tokens = marker_tokens;
                tokens.push(Placed { token: space, x });
                tokens.append(&mut item_reading.tokens);
                item_reading.tokens = tokens;
            }
            item_reading.text = format!("{marker} {}", item_reading.text.trim());
            boxes[item] = [
                marker_box[0].min(item_box[0]),
                marker_box[1].min(item_box[1]),
                marker_box[2].max(item_box[2]),
                marker_box[3].max(item_box[3]),
            ];
            keep[index] = false;
        }
        (0..readings.len()).filter(|&index| keep[index]).collect()
    }

    /// The line a marker's item is: the nearest kept line with text on the marker's row that
    /// starts within `MARKER_GAP` line heights after it.
    fn item_of(
        &self,
        marker: usize,
        boxes: &[LineBox],
        readings: &[Reading],
        keep: &[bool],
    ) -> Option<usize> {
        let height = |line: &LineBox| line[3] - line[1];
        let own = boxes[marker];
        let mut best: Option<(usize, f32)> = None;
        for (other, line) in boxes.iter().enumerate() {
            if other == marker || !keep[other] || readings[other].text.trim().is_empty() {
                continue;
            }
            let overlap = own[3].min(line[3]) - own[1].max(line[1]);
            if overlap < 0.5 * height(&own).min(height(line)) {
                continue;
            }
            let gap = if self.rtl {
                own[0] - line[2]
            } else {
                line[0] - own[2]
            };
            let within = -MARKER_OVERLAP * height(&own) <= gap
                && gap <= MARKER_GAP * height(&own).max(height(line));
            if within && best.is_none_or(|(_, best_gap)| gap < best_gap) {
                best = Some((other, gap));
            }
        }
        best.map(|(other, _)| other)
    }
}

/// A line of leader characters (dots, rules) that is not an empty table cell: three or more,
/// a box three times wider than tall, or any of them but a lone dash.
fn is_leader_line(line: &LineBox, text: &str) -> bool {
    let core: Vec<char> = text.chars().filter(|&character| character != ' ').collect();
    if core.is_empty() || !core.iter().all(|&character| LEADERS.contains(character)) {
        return false;
    }
    let wide = line[2] - line[0] >= 3.0 * (line[3] - line[1]);
    core.len() >= 3 || wide || !core.iter().any(|&character| DASHES.contains(character))
}

/// A page's recognized lines: its size, each line's box and reading.
pub struct Recognized {
    pub width: usize,
    pub height: usize,
    pub boxes: Vec<LineBox>,
    pub readings: Vec<Reading>,
}

/// The page's lines in reading order, its blocks and Markdown.
pub struct Assembled {
    pub lines: Vec<Line>,
    pub blocks: Vec<Block>,
    pub markdown: String,
}

impl Assembler<'_> {
    /// Lines cut at wide gaps and tidied, ordered by `relate` (the order network over the
    /// lines' features), then blocks and Markdown. `vertical`: a vertical page read turned,
    /// whose blocks and order come from XY-cut.
    pub fn assemble(
        &self,
        page: &Recognized,
        vertical: bool,
        relate: impl FnOnce(&[LineBox], &[String]) -> Relations,
    ) -> Assembled {
        let (boxes, readings) = self.cut_and_tidy(page);
        let texts: Vec<String> = readings
            .iter()
            .map(|reading| reading.text.clone())
            .collect();
        let mirrored: Vec<LineBox> = boxes
            .iter()
            .map(|line| self.mirror(line, page.width))
            .collect();
        if boxes.len() < 2 {
            let order = xy_cut(&mirrored);
            return Assembled {
                lines: lines(&order, &boxes, &readings),
                blocks: Vec::new(),
                markdown: texts.join("\n"),
            };
        }
        let mut relations = relate(&mirrored, &texts);
        let order = if vertical {
            read_vertically(&boxes, &mut relations)
        } else {
            blocks_order(
                &relations.order,
                &relations.same,
                &mirrored,
                &relations.classes,
            )
        };
        let layout = Lines {
            boxes: &mirrored,
            texts: &texts,
            classes: &relations.classes,
            same: &relations.same,
        };
        let blocks = layout::blocks(&layout, &order)
            .into_iter()
            .map(|block| Block {
                bbox: self.mirror_back(block.bbox, page.width),
                ..block
            })
            .collect();
        let markdown = layout::markdown(&layout, &order, |line| {
            let pieces = self.split(boxes[line], &readings[line].tokens, CELL_GAP, &[]);
            pieces
                .into_iter()
                .map(|(piece, text, _)| (self.mirror(&piece, page.width), text))
                .collect()
        });
        Assembled {
            lines: lines(&order, &boxes, &readings),
            blocks,
            markdown,
        }
    }

    /// The lines cut at wide gaps and tidied.
    fn cut_and_tidy(&self, page: &Recognized) -> (Vec<LineBox>, Vec<Reading>) {
        let cuts = self.gutter_cuts(&page.boxes, &page.readings);
        let mut boxes = Vec::new();
        let mut readings = Vec::new();
        for ((line, reading), cuts) in page.boxes.iter().zip(&page.readings).zip(&cuts) {
            for (piece, text, tokens) in self.split(*line, &reading.tokens, SPLIT_GAP, cuts) {
                boxes.push(piece);
                readings.push(Reading {
                    text,
                    confidence: reading.confidence,
                    tokens,
                });
            }
        }
        let kept = self.tidy(&mut boxes, &mut readings);
        let boxes = kept.iter().map(|&index| boxes[index]).collect();
        let readings = kept
            .into_iter()
            .map(|index| readings[index].clone())
            .collect();
        (boxes, readings)
    }

    /// A box as the layout sees it: mirrored left to right on a right-to-left page.
    fn mirror(&self, line: &LineBox, width: usize) -> LineBox {
        let width = width as f32;
        if self.rtl {
            [width - line[2], line[1], width - line[0], line[3]]
        } else {
            *line
        }
    }

    /// A block's rounded box back on the page as it is.
    fn mirror_back(&self, bbox: [f64; 4], width: usize) -> [f64; 4] {
        if !self.rtl {
            return bbox;
        }
        let width = width as f64;
        let mirrored = [width - bbox[2], bbox[1], width - bbox[0], bbox[3]];
        mirrored.map(|value| rounded(f64::from(value as f32), 1))
    }
}

/// The lines in `order`, their boxes to one decimal and confidences to four.
fn lines(order: &[usize], boxes: &[LineBox], readings: &[Reading]) -> Vec<Line> {
    order
        .iter()
        .map(|&index| Line {
            bbox: boxes[index].map(|value| rounded(f64::from(value), 1)),
            text: readings[index].text.clone(),
            conf: rounded(readings[index].confidence, 4),
        })
        .collect()
}

/// A vertical page's order and blocks from XY-cut: the order network saw no vertical pages.
/// Its same-block logits become the blocks', and grid classes are ruled out.
fn read_vertically(boxes: &[LineBox], relations: &mut Relations) -> Vec<usize> {
    let count = boxes.len();
    let blocks = xy_cut_blocks(boxes);
    relations.same = vec![-1.0; count * count];
    for block in &blocks {
        for &first in block {
            for &second in block {
                relations.same[first * count + second] = 1.0;
            }
        }
    }
    for classes in relations.classes.as_chunks_mut::<{ layout::CLASSES }>().0 {
        for class in (0..layout::CLASSES).filter(|&class| layout::is_grid(class)) {
            classes[class] = RULED_OUT;
        }
    }
    blocks.into_iter().flatten().collect()
}
