//! A conversation's prompt, byte for byte as the variant's chat template and processor render it:
//! the conversation's system text when it has one, its tools listed after it (`system`), each
//! turn's user message (its images first, each its tokens between its vision markers, then its
//! audio clips, each its tokens, between the same markers for LFM3-Omni-2B, then its text), each
//! earlier reply's text as the template writes it back (its reasoning dropped, behind an empty
//! thinking block but for LFM3-Audio-2B), and the reply's opener (`Mode`): a thinking block,
//! opened when the turn thinks and empty when not, or the marker of a spoken reply or of one
//! interleaving text and speech.

use bundle::Tokenizer;
use serde::Deserialize;

use crate::Error;

const START: &str = "<|startoftext|>";
const TURN: &str = "<|im_start|>";
const END: &str = "<|im_end|>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const EMPTY_THOUGHT: &str = "<think></think>";
const TOOLS_START: &str = "<tools>\n";
const TOOLS_END: &str = "\n</tools>";
const VISION_START: &str = "<|vision_start|>";
const IMAGE: &str = "<|image|>";
const VISION_END: &str = "<|vision_end|>";
const AUDIO: &str = "<|audio|>";
const AUDIO_START: &str = "<|audio_start|>";
const MIXED_START: &str = "<|mixed_start|>";
/// The token a reply ends with, `<|im_end|>`.
pub const REPLY_END: u32 = 124_900;
/// The token of an image's rows, `<|image|>`: the projector's tokens take their place.
pub const IMAGE_TOKEN: u32 = 125_030;
/// The token of a clip's rows, `<|audio|>`: the audio projector's tokens take their place.
pub const AUDIO_TOKEN: u32 = 125_031;
/// The token of a spoken frame's row, `<|audio_out|>`: its codes' embeddings take its place.
pub const AUDIO_OUT_TOKEN: u32 = 125_032;
/// `<|audio_start|>`: the reply speaks from here.
pub const AUDIO_START_TOKEN: u32 = 125_012;
/// `<|text_end|>`: the text of a reply ends, or of an interleaved one, whose speech goes on.
pub const TEXT_END: u32 = 125_014;
/// `<|mixed_start|>`: the reply interleaves text and speech from here.
pub const MIXED_START_TOKEN: u32 = 125_015;

/// A user's message: its images' tokens and its clips' tokens, each one's count, and its text.
#[derive(Debug, Clone, Default)]
pub struct Turn {
    pub images: Vec<usize>,
    pub audio: Vec<usize>,
    pub text: String,
}

/// How a reply is written: as text (a transcript too), spoken, or text and speech in turns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Text,
    /// Text written as the template's own generation prompt opens it: speech recognition.
    Transcribe,
    Speech,
    Interleaved,
}

impl Mode {
    /// Whether the reply may speak.
    pub fn speaks(self) -> bool {
        matches!(self, Self::Speech | Self::Interleaved)
    }
}

/// Whose chat template writes the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// LFM3's, LFM3-VL-2B's and LFM3-Omni-2B's: replies written back behind an empty thinking
    /// block, clips between vision markers.
    Lfm3,
    /// LFM3-Audio-2B's: replies written back as they are, clips bare, no thinking block opened.
    Audio,
}

/// What the prompt is written with.
#[derive(Clone, Copy)]
pub struct Chat<'chat> {
    pub tokenizer: &'chat Tokenizer,
    pub template: Template,
}

/// How the reply opens.
#[derive(Debug, Clone, Copy)]
pub struct Opening {
    pub mode: Mode,
    pub thinking: bool,
}

/// A user's message, and its reply as generated (its reasoning included).
#[derive(Debug, Clone)]
pub struct Replied {
    pub turn: Turn,
    pub reply: String,
}

/// The system text as the template writes it with `tools` listed after it, each tool's JSON as
/// its `tool_to_json` writes it; the text alone without tools.
pub fn system(text: &str, tools: &[String]) -> String {
    if tools.is_empty() {
        return text.to_owned();
    }
    let separator = if text.is_empty() { "" } else { "\n\n" };
    format!(
        "{text}{separator}{TOOLS_START}{}{TOOLS_END}",
        tools.join("\n")
    )
}

/// The ids of the prompt asking for the reply to `next`, after `system` and `history`: as many
/// image and audio ids as the turns have images' and clips' tokens, in order.
///
/// # Errors
/// When the text doesn't tokenize, or holds an image's or a clip's token itself.
pub fn prompt(
    chat: Chat,
    system: &str,
    history: &[Replied],
    next: &Turn,
    opening: Opening,
) -> Result<Vec<u32>, Error> {
    let mut text = START.to_owned();
    if !system.is_empty() {
        text.push_str(&format!("{TURN}system\n{system}{END}\n"));
    }
    for Replied { turn, reply } in history {
        let answer = match reasoned(reply) {
            (Some(_), answer) => answer.trim(),
            (None, answer) => answer,
        };
        let thought = match chat.template {
            Template::Lfm3 => EMPTY_THOUGHT,
            Template::Audio => "",
        };
        push_user(&mut text, turn, chat.template);
        text.push_str(&format!("{TURN}assistant\n{thought}{answer}{END}\n"));
    }
    push_user(&mut text, next, chat.template);
    text.push_str(&format!(
        "{TURN}assistant\n{}",
        opener(opening, chat.template)
    ));
    let ids = chat.tokenizer.encode(&text)?;
    let turns: Vec<&Turn> = history
        .iter()
        .map(|replied| &replied.turn)
        .chain([next])
        .collect();
    let rows = [
        (
            IMAGE_TOKEN,
            IMAGE,
            turns.iter().flat_map(|turn| &turn.images).sum::<usize>(),
        ),
        (
            AUDIO_TOKEN,
            AUDIO,
            turns.iter().flat_map(|turn| &turn.audio).sum(),
        ),
    ];
    for (token, name, count) in rows {
        if ids.iter().filter(|&&id| id == token).count() != count {
            return Err(Error::Request(format!(
                "the conversation's text holds {name}, the token of an image's or a clip's rows"
            )));
        }
    }
    Ok(ids)
}

/// What the assistant's turn opens with: its thinking block, which only LFM3-Audio-2B's template
/// leaves out of a reply that does not write text first, then the marker of a reply that speaks.
fn opener(opening: Opening, template: Template) -> String {
    let thought = match (opening.mode, template) {
        (Mode::Text, _) if opening.thinking => THINK_START,
        (Mode::Text, _) | (_, Template::Lfm3) => EMPTY_THOUGHT,
        (_, Template::Audio) => "",
    };
    let marker = match opening.mode {
        Mode::Text | Mode::Transcribe => "",
        Mode::Speech => AUDIO_START,
        Mode::Interleaved => MIXED_START,
    };
    format!("{thought}{marker}")
}

/// A user's turn: its images, then its clips, then its text.
fn push_user(text: &mut String, turn: &Turn, template: Template) {
    text.push_str(&format!("{TURN}user\n"));
    for &tokens in &turn.images {
        text.push_str(VISION_START);
        text.push_str(&IMAGE.repeat(tokens));
        text.push_str(VISION_END);
    }
    for &tokens in &turn.audio {
        let clip = AUDIO.repeat(tokens);
        match template {
            Template::Lfm3 => text.push_str(&format!("{VISION_START}{clip}{VISION_END}")),
            Template::Audio => text.push_str(&clip),
        }
    }
    text.push_str(&format!("{}{END}\n", turn.text));
}

/// A reply's text tokens: its spoken rows and the markers that end its text left out.
pub fn written(reply: &[u32]) -> Vec<u32> {
    let markers = [AUDIO_OUT_TOKEN, TEXT_END, REPLY_END];
    let text = reply.iter().filter(|token| !markers.contains(token));
    text.copied().collect()
}

/// A reply's reasoning, when it has one, and its answer: what follows the last `</think>`, as
/// the template writes it back.
pub(crate) fn reasoned(reply: &str) -> (Option<&str>, &str) {
    match reply.rsplit_once(THINK_END) {
        Some((reasoning, answer)) => (Some(reasoning), answer),
        None => (None, reply),
    }
}

#[cfg(test)]
mod tests {
    use super::system;

    const ADD: &str = r#"{"name": "add", "description": "Adds"}"#;
    const SUBTRACT: &str = r#"{"name": "sub", "description": "Subtracts"}"#;

    #[test]
    fn lists_tools_after_the_system_text_as_the_template_writes_them() {
        let cases = [
            (
                "Fill it.",
                vec![ADD],
                "Fill it.\n\n<tools>\n{\"name\": \"add\", \"description\": \"Adds\"}\n</tools>",
            ),
            (
                "Fill it.",
                vec![ADD, SUBTRACT],
                "Fill it.\n\n<tools>\n{\"name\": \"add\", \"description\": \"Adds\"}\n\
                 {\"name\": \"sub\", \"description\": \"Subtracts\"}\n</tools>",
            ),
            (
                "",
                vec![ADD],
                "<tools>\n{\"name\": \"add\", \"description\": \"Adds\"}\n</tools>",
            ),
            ("Fill it.", vec![], "Fill it."),
        ];
        for (text, tools, expected) in cases {
            let tools: Vec<String> = tools.into_iter().map(str::to_owned).collect();
            assert_eq!(system(text, &tools), expected);
        }
    }
}
