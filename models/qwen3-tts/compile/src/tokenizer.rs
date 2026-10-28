//! The tokenizer blob: Qwen3's byte-level BPE (`tokenizer.json` of Qwen/Qwen3-0.6B, whose
//! `vocab.json` and `merges.txt` are this checkpoint's) with the added tokens of
//! `tokenizer_config.json` that it lacks: this model's `<tts_pad>` and the like.

use std::path::Path;

use bundle::TOKENIZER_PART;
use compile::{BundleWriter, Error, read_json};
use serde_json::{Map, Value, json};

const CONFIG: &str = "tokenizer_config.json";

pub(crate) fn write(dir: &Path, writer: &mut BundleWriter) -> Result<(), Error> {
    let mut tokenizer: Value = read_json(&dir.join(TOKENIZER_PART))?;
    let config: Value = read_json(&dir.join(CONFIG))?;
    let declared = config["added_tokens_decoder"]
        .as_object()
        .ok_or_else(|| Error::Format(format!("{CONFIG}: no added_tokens_decoder")))?;
    let added = tokenizer["added_tokens"]
        .as_array_mut()
        .ok_or_else(|| Error::Format(format!("{TOKENIZER_PART}: no added_tokens")))?;
    let known: Vec<u64> = added
        .iter()
        .filter_map(|token| token["id"].as_u64())
        .collect();
    for (id, token) in sorted(declared)? {
        if !known.contains(&id) {
            added.push(json!({
                "id": id,
                "content": token["content"],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": token["special"],
            }));
        }
    }
    let bytes = serde_json::to_vec(&tokenizer)
        .map_err(|error| Error::Format(format!("{TOKENIZER_PART}: {error}")))?;
    writer.add_blob(TOKENIZER_PART, &bytes)
}

/// `declared`'s tokens by id.
fn sorted(declared: &Map<String, Value>) -> Result<Vec<(u64, &Value)>, Error> {
    let mut tokens = Vec::new();
    for (id, token) in declared {
        let id = id
            .parse()
            .map_err(|_| Error::Format(format!("{CONFIG}: the token id {id:?} is not a number")))?;
        tokens.push((id, token));
    }
    tokens.sort_by_key(|(id, _)| *id);
    Ok(tokens)
}
