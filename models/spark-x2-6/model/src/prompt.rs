//! A conversation's prompt, byte for byte as the chat template renders it: the system block (the
//! template's own instruction, the thinking effort when the turn thinks, then the conversation's
//! system text), each turn's user message, each earlier reply as the template writes it back
//! (its reasoning dropped), and the reply's opener, which opens a thinking block when the turn
//! thinks and closes it when not.

use bundle::Tokenizer;

use crate::Error;

const START: &str = "<｜start▁of▁sentence｜>";
const END: &str = "<｜end▁of▁sentence｜>";
const INSTRUCTION: &str = "you are a helpful assistant.";
/// What the instruction gains when a turn thinks: the template's default `reasoning_effort`,
/// high.
const THINKING_EFFORT: &str = include_str!("thinking_effort.txt");
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
/// The token a reply ends with, `<｜end▁of▁sentence｜>`.
pub const REPLY_END: u32 = 1;

/// A user's message, and its reply as generated (its reasoning included).
#[derive(Debug, Clone)]
pub struct Replied {
    pub text: String,
    pub reply: String,
}

/// The ids of the prompt asking for the reply to `next`, after `system` and `history`.
///
/// # Errors
/// When the text doesn't tokenize.
pub fn prompt(
    tokenizer: &Tokenizer,
    system: &str,
    history: &[Replied],
    next: &str,
    thinking: bool,
) -> Result<Vec<u32>, Error> {
    let effort = if thinking { THINKING_EFFORT } else { "" };
    let mut text = format!("{START}<|System|>\n{INSTRUCTION}{effort}");
    if !system.is_empty() {
        text.push_str(&format!("\n\n{system}"));
    }
    text.push_str(END);
    for Replied { text: user, reply } in history {
        let answer = reasoned(reply).1;
        text.push_str(&format!("{START}<|User|>{user}{END}"));
        text.push_str(&format!("{START}<|Bot|>{THINK_END}{answer}{END}"));
    }
    let opener = if thinking { THINK_START } else { THINK_END };
    text.push_str(&format!("{START}<|User|>{next}{END}{START}<|Bot|>{opener}"));
    tokenizer.encode(&text)
}

/// A reply's reasoning, when it has one, and its answer: what follows the last `</think>`, as
/// the template writes it back.
pub(crate) fn reasoned(reply: &str) -> (Option<&str>, &str) {
    match reply.rsplit_once(THINK_END) {
        Some((reasoning, answer)) => (Some(reasoning), answer),
        None => (None, reply),
    }
}
