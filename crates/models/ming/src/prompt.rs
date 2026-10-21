//! The prompt the checkpoint was trained on (`prompt_contract.build_input_ids` and `tts_tasks`
//! in the reference). Every piece is tokenized on its own, without special tokens
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

use bundle::Tokenizer;

use crate::Error;

const SYSTEM: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const USER: &str = "<|im_start|>user\n";
const TTS_PROMPT: &str = "Please generate speech based on the following description.\n";
const SPEAKER: [&str; 2] = ["  speaker_1:", "<|vision_start|>"];
/// The piece whose row is the speaker embedding.
const SPEAKER_EMBEDDING: &str = "<|vision_pad|>";
const SPEAKER_END: &str = "<|vision_end|>\n";
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

/// A request's target language (`tts_tasks._TARGET_LANGUAGE_NAMES`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Chinese,
    Cantonese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Spanish,
    /// Any other name, passed to the prompt as given.
    Other(String),
}

impl Language {
    /// A language code or name in any case, with `_` or `-`; `None` when blank.
    pub fn parse(text: &str) -> Option<Self> {
        let name = text.trim();
        let language = match name.to_lowercase().replace('_', "-").as_str() {
            "" => return None,
            "zh" | "zh-cn" | "chinese" => Self::Chinese,
            "yue" | "zh-yue" | "cantonese" => Self::Cantonese,
            "en" | "english" => Self::English,
            "ja" | "japanese" => Self::Japanese,
            "ko" | "korean" => Self::Korean,
            "de" | "german" => Self::German,
            "fr" | "french" => Self::French,
            "es" | "spanish" => Self::Spanish,
            _ => Self::Other(name.to_string()),
        };
        Some(language)
    }

    /// The name in the prompt's "Target language" line.
    fn name(&self) -> &str {
        match self {
            Self::Chinese | Self::Cantonese => "Chinese",
            Self::English => "English",
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::German => "German",
            Self::French => "French",
            Self::Spanish => "Spanish",
            Self::Other(name) => name,
        }
    }

    /// `tts_tasks.preserve_multilingual_text`: languages other than Chinese and English bypass
    /// the Chinese/English text normalization.
    pub(crate) fn is_multilingual(&self) -> bool {
        !matches!(self, Self::Chinese | Self::English)
    }
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
    pub(crate) fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    fn encode(&self, piece: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        ids.extend(self.tokenizer.encode(piece)?);
        Ok(())
    }

    /// `language` adds the "Target language" line (the reference requires it for de/es/fr/ja/ko).
    pub fn build(
        &self,
        text: &str,
        voice: &Voice,
        language: Option<&Language>,
    ) -> Result<Prompt, Error> {
        let mut ids = Vec::new();
        let target = language.map(|language| format!("Target language: {}.\n", language.name()));
        let head = [SYSTEM, USER, TTS_PROMPT]
            .into_iter()
            .chain(target.as_deref());
        for piece in head.chain(SPEAKER) {
            self.encode(piece, &mut ids)?;
        }
        let speaker_row = ids.len();
        let instruction = instruction(voice);
        let pieces = [SPEAKER_EMBEDDING, SPEAKER_END, TEXT_PREFIX, text]
            .into_iter()
            .chain(TURN_END)
            .chain([instruction.as_str(), INSTRUCTION_END, AUDIO]);
        for piece in pieces {
            self.encode(piece, &mut ids)?;
        }
        Ok(Prompt { ids, speaker_row })
    }
}

/// The assistant-side caption: `tts_tasks.build_structured_instruction` with the zh schema
/// (key order and `json.dumps` spacing are part of the contract), or the free-form text.
fn instruction(voice: &Voice) -> String {
    let (style, ip) = match voice {
        Voice::Freeform(text) => return text.trim().to_string(),
        Voice::Builtin(ip) => (None, Some(ip.as_str())),
        Voice::Design(style) => (Some(style.as_str()), None),
    };
    let json = |value: Option<&str>| {
        value.map_or_else(
            || "null".to_string(),
            |text| serde_json::Value::from(text).to_string(),
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
    use bundle::Tokenizer;

    use super::{Language, Prompter, Voice};

    /// The prompt for the reference's built-in voice, its target language line included.
    const PROMPT: &str = concat!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "Please generate speech based on the following description.\n",
        "Target language: German.\n  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n",
        " Text input:\nHallo.<|im_end|>\n<|im_start|>assistant\n",
        r#"{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "#,
        r#""语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "#,
        r#""Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "#,
        r#""IP": "US_James"}]}<|im_end|><audio>"#,
    );

    /// One token per character of `PROMPT`, its id the character's position in `characters`.
    fn tokenizer(characters: &[char]) -> Tokenizer {
        let vocab: serde_json::Map<String, serde_json::Value> = characters
            .iter()
            .enumerate()
            .map(|(id, character)| (character.to_string(), id.into()))
            .collect();
        let json = serde_json::json!({
            "version": "1.0",
            "pre_tokenizer": {
                "type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false
            },
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<"},
        });
        Tokenizer::parse(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn builtin_voice_prompt_matches_the_reference() {
        let mut characters: Vec<char> = PROMPT.chars().collect();
        characters.sort_unstable();
        characters.dedup();
        let prompter = Prompter::new(tokenizer(&characters));
        let voice = Voice::Builtin("US_James".into());
        let language = Language::parse(" de ");
        let prompt = prompter.build("Hallo.", &voice, language.as_ref()).unwrap();
        let text: String = prompt
            .ids
            .iter()
            .map(|&id| characters[id as usize])
            .collect();
        assert_eq!(text, PROMPT);
        let speaker: String = text.chars().skip(prompt.speaker_row).take(14).collect();
        assert_eq!(speaker, "<|vision_pad|>");
    }
}
