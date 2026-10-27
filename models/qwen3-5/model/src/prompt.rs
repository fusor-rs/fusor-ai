//! A conversation's prompt, byte for byte as the reference renders it (the chat template and the
//! processor's placeholder expansion): its system block, with its tools, when it has one; each
//! turn's user message
//! with its images' tokens, then its videos' groups, each after its time, then its text; each
//! earlier reply as the template writes it back (its reasoning dropped); and the reply's opener,
//! its thinking block empty unless the turn thinks.

use std::ops::Range;

use blocks::rotary::Position;
use blocks::vit::{self, Grid};
use bundle::{Tokenizer, python};
use serde::Deserialize;

use crate::Error;
use crate::media::Item;

const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
/// The tokens each of an image's and a video's language-model tokens hold the place of.
const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 248_056);
const VIDEO_PAD: (&str, u32) = ("<|video_pad|>", 248_057);
/// What closes a reasoning.
const THINK_END: &str = "</think>";
/// The tokens a reply ends with: `<|im_end|>` and `<|endoftext|>`.
pub const ENDS: [u32; 2] = [248_046, 248_044];
/// What follows the assistant's role when a turn thinks, and when it does not.
const THINKING: &str = "<think>\n";
const NOT_THINKING: &str = "<think>\n\n</think>\n\n";
/// What the template writes before a system block's tools, and after them.
const TOOLS_START: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_END: &str = include_str!("tools_end.txt");

/// What a conversation begins with, before its first turn: the template's system block.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct System {
    pub text: String,
    /// Each tool as the template's `tojson` writes it, one JSON object a line.
    #[serde(default)]
    pub tools: Vec<String>,
}

/// A user's message: its text and media.
#[derive(Debug, Clone)]
pub struct Turn {
    pub text: String,
    pub media: Vec<Item>,
}

/// A turn of the conversation so far, and its reply as generated (its reasoning included).
#[derive(Debug, Clone)]
pub struct Replied {
    pub turn: Turn,
    pub reply: String,
}

pub struct Prompt {
    pub ids: Vec<u32>,
    /// The rows of each image's and each video group's tokens, in order, with its grid of
    /// language-model tokens.
    pub media: Vec<(Range<usize>, Grid)>,
    /// Every row's rotary position.
    pub positions: Vec<Position>,
    /// The first row of the reply's opener, after the assistant's role: the rows before it are
    /// the prompt of every later turn too.
    pub opener: usize,
}

impl Prompt {
    /// The prompt asking for the reply to `next`, after `history`.
    ///
    /// # Errors
    /// When a message holds an image or video placeholder of its own.
    pub fn new(
        tokenizer: &Tokenizer,
        system: &System,
        history: &[Replied],
        next: &Turn,
        thinking: bool,
    ) -> Result<Self, Error> {
        let mut text = system.written();
        let mut grids = Vec::new();
        for Replied { turn, reply } in history {
            text.push_str(&user(turn, &mut grids)?);
            let reply = written_back(reply);
            text.push_str(&format!("<|im_start|>assistant\n{reply}<|im_end|>\n"));
        }
        text.push_str(&user(next, &mut grids)?);
        let opener = if thinking { THINKING } else { NOT_THINKING };
        text.push_str(&format!("<|im_start|>assistant\n{opener}"));
        let ids = tokenizer.encode(&text)?;
        // The opener starts with a special token, so its ids are the same alone as after the
        // role, and the rows before it are those of the same text in any later prompt.
        let opener = ids.len() - tokenizer.encode(opener)?.len();
        let media = media_rows(&ids, grids);
        Ok(Self {
            positions: vit::positions(ids.len(), &media),
            ids,
            media,
            opener,
        })
    }

    /// The rotary position of the reply's `index`th token.
    pub fn reply_position(&self, index: usize) -> Position {
        let last = self.positions.last().expect("a prompt has rows");
        Position::uniform(last.temporal + 1 + index)
    }
}

impl System {
    /// The block as the template writes it: its tools, then its text; nothing when it has
    /// neither.
    fn written(&self) -> String {
        let text = python::strip(&self.text);
        let content = if self.tools.is_empty() {
            if text.is_empty() {
                return String::new();
            }
            text.to_owned()
        } else {
            let tools: String = self.tools.iter().map(|tool| format!("\n{tool}")).collect();
            let text = if text.is_empty() {
                String::new()
            } else {
                format!("\n\n{text}")
            };
            format!("{TOOLS_START}{tools}{TOOLS_END}{text}")
        };
        format!("<|im_start|>system\n{content}<|im_end|>\n")
    }
}

/// A user's message as the template writes it, its media's placeholders expanded and their
/// grids pushed onto `grids`.
fn user(turn: &Turn, grids: &mut Vec<Grid>) -> Result<String, Error> {
    if turn.text.contains(IMAGE_PAD.0) || turn.text.contains(VIDEO_PAD.0) {
        return Err(Error::Request(
            "a message contains a literal image or video placeholder token".into(),
        ));
    }
    let mut content = placeholders(&turn.media, grids);
    content.push_str(&turn.text);
    let content = python::strip(&content);
    Ok(format!("<|im_start|>user\n{content}<|im_end|>\n"))
}

/// A reply's reasoning, when it has one, and its answer: what follows the last `</think>`, as
/// the template writes it back.
pub(crate) fn reasoned(reply: &str) -> (Option<&str>, &str) {
    match reply.rsplit_once(THINK_END) {
        Some((reasoning, answer)) => (Some(reasoning), answer.trim_start_matches('\n')),
        None => (None, reply),
    }
}

/// A reply as the template writes it back into a later prompt: stripped, and only what follows
/// its reasoning.
fn written_back(reply: &str) -> &str {
    reasoned(python::strip(reply)).1
}

/// Each image's placeholder, then each video's, expanded: one pad per language-model token, a
/// video's groups after their times; their grids of tokens pushed onto `grids`, in order.
fn placeholders(media: &[Item], grids: &mut Vec<Grid>) -> String {
    let mut text = String::new();
    for item in media {
        match item {
            Item::Image(grid) => {
                let pads = IMAGE_PAD.0.repeat(grid.patches());
                text.push_str(&format!("{VISION_START}{pads}{VISION_END}"));
                grids.push(*grid);
            }
            Item::Video(groups) => {
                text.push_str(VISION_START);
                for (grid, time) in groups {
                    let pads = VIDEO_PAD.0.repeat(grid.patches());
                    text.push_str(&format!(
                        "<{time:.1} seconds>{VISION_START}{pads}{VISION_END}"
                    ));
                    grids.push(*grid);
                }
                text.push_str(VISION_END);
            }
        }
    }
    text
}

/// The rows of each run of pad tokens, in order, with the grids they hold the place of.
fn media_rows(ids: &[u32], grids: Vec<Grid>) -> Vec<(Range<usize>, Grid)> {
    let is_pad = |id: u32| id == IMAGE_PAD.1 || id == VIDEO_PAD.1;
    let mut rows = Vec::new();
    let mut row = 0;
    while row < ids.len() {
        let length = ids[row..].iter().take_while(|&&id| is_pad(id)).count();
        if length > 0 {
            rows.push(row..row + length);
        }
        row += length.max(1);
    }
    rows.into_iter().zip(grids).collect()
}
