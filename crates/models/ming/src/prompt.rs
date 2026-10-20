//! The prompt the checkpoint was trained on (plan §1.1; `prompt_contract.build_input_ids` and
//! `tts_tasks` in the reference). Every piece is tokenized on its own, without special tokens
//! added, and the pieces are concatenated:
//!
//! ```text
//! <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n <|im_start|>user\n
//! Please generate speech based on the following description.\n [Target language: X.\n]
//!   speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n  Text input:\n {text}
//! <|im_end|>\n <|im_start|>assistant\n {instruction}<|im_end|> <audio>
//! ```
//!
//! The `<|vision_pad|>` row is the speaker embedding, all zeros here (`use_zero_spk_emb`): every
//! voice comes from the instruction — a built-in voice (`IP`) or a voice description (`风格`) in
//! the zh-schema caption, or a free-form instruction.

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::Error;

const SYSTEM: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const USER: &str = "<|im_start|>user\n";
const TTS_PROMPT: &str = "Please generate speech based on the following description.\n";
const SPEAKER: [&str; 4] = [
    "  speaker_1:",
    "<|vision_start|>",
    "<|vision_pad|>",
    "<|vision_end|>\n",
];
const TEXT_PREFIX: &str = " Text input:\n";
const TURN_END: [&str; 2] = ["<|im_end|>\n", "<|im_start|>assistant\n"];
const INSTRUCTION_END: &str = "<|im_end|>";
const AUDIO: &str = "<audio>";

/// Built-in voices offered by the UI: `(ip, label)`. The model knows 100+ Chinese voices; any
/// other `ip` string is passed through as given.
pub const VOICES: [(&str, &str); 7] = [
    ("US_James", "James — US English, male narrator"),
    ("US_Emma", "Emma — US English, female"),
    ("US_Stella", "Stella — English, female"),
    ("UK_George", "George — UK English, male"),
    ("UK_Charlotte", "Charlotte — UK English, female"),
    ("灵小柔", "灵小柔 — Chinese, bright young female"),
    ("灵楷", "灵楷 — Chinese, warm young male"),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// A built-in voice (task `IP`).
    Builtin(String),
    /// A voice described in words (task `VOICE_DESIGN`).
    Design(String),
    /// A natural-language instruction for voice and delivery (task `FREEFORM`).
    Freeform(String),
}

pub struct Prompter {
    tokenizer: Tokenizer,
}

pub struct Prompt {
    pub ids: Vec<u32>,
    /// The row whose embedding is the (zero) speaker embedding.
    pub speaker_row: usize,
}

impl Prompter {
    pub fn new(tokenizer_json: &[u8]) -> Result<Self, Error> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| Error::Format(format!("tokenizer: {e}")))?;
        Ok(Self { tokenizer })
    }

    fn encode(&self, piece: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        let encoding = self
            .tokenizer
            .encode(piece, false)
            .map_err(|e| Error::Format(format!("tokenize: {e}")))?;
        ids.extend_from_slice(encoding.get_ids());
        Ok(())
    }

    /// `language` adds the "Target language" line (the reference requires it for de/es/fr/ja/ko).
    pub fn build(
        &self,
        text: &str,
        voice: &Voice,
        language: Option<&str>,
    ) -> Result<Prompt, Error> {
        let mut ids = Vec::new();
        let target = language.map(|l| format!("Target language: {}.\n", language_name(l)));
        for piece in [SYSTEM, USER, TTS_PROMPT]
            .into_iter()
            .chain(target.as_deref())
        {
            self.encode(piece, &mut ids)?;
        }
        let mut speaker_row = 0;
        for piece in SPEAKER {
            if piece == SPEAKER[2] {
                speaker_row = ids.len();
            }
            self.encode(piece, &mut ids)?;
        }
        for piece in [TEXT_PREFIX, text].into_iter().chain(TURN_END) {
            self.encode(piece, &mut ids)?;
        }
        for piece in [instruction(voice).as_str(), INSTRUCTION_END, AUDIO] {
            self.encode(piece, &mut ids)?;
        }
        Ok(Prompt { ids, speaker_row })
    }
}

/// `tts_tasks._TARGET_LANGUAGE_NAMES`: codes and lower-case names to the name in the prompt;
/// anything else is passed through.
fn language_name(language: &str) -> &str {
    match language.trim().to_lowercase().as_str() {
        "zh" | "zh-cn" | "chinese" | "yue" | "zh-yue" | "cantonese" => "Chinese",
        "en" | "english" => "English",
        "ja" | "japanese" => "Japanese",
        "ko" | "korean" => "Korean",
        "de" | "german" => "German",
        "fr" | "french" => "French",
        "es" | "spanish" => "Spanish",
        _ => language.trim(),
    }
}

/// `tts_tasks.preserve_multilingual_text`: languages other than Chinese and English bypass the
/// Chinese/English text normalization.
pub fn is_multilingual(language: &str) -> bool {
    !matches!(
        language.trim().to_lowercase().replace('_', "-").as_str(),
        "zh" | "zh-cn" | "chinese" | "en" | "english"
    )
}

/// The assistant-side caption: `tts_tasks.build_structured_instruction` with the zh schema
/// (key order and `json.dumps` spacing are part of the contract), or the free-form text.
fn instruction(voice: &Voice) -> String {
    let (style, ip) = match voice {
        Voice::Freeform(text) => return text.trim().to_string(),
        Voice::Builtin(ip) => (None, Some(ip.as_str())),
        Voice::Design(style) => (Some(style.as_str()), None),
    };
    let json = |v: Option<&str>| {
        v.map_or_else(
            || "null".to_string(),
            |s| serde_json::Value::from(s).to_string(),
        )
    };
    format!(
        concat!(
            r#"{{"audio_sequence": [{{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": {}, "#,
            r#""语速": null, "基频": null, "音量": null, "情感": null, "BGM": {{"Genre": null, "#,
            r#""Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}}, "IP": {}}}]}}"#
        ),
        json(style),
        json(ip)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_voice_caption_matches_the_reference() {
        let expected = r#"{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "IP": "US_James"}]}"#;
        assert_eq!(instruction(&Voice::Builtin("US_James".into())), expected);
    }
}
