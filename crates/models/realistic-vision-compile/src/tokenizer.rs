//! CLIP's tokenizer as a `tokenizer.json` (the tokenizers crate's format), from the slow
//! tokenizer's `vocab.json` and `merges.txt`: the fast tokenizer `CLIPTokenizerFast` builds from
//! them (lowercased NFC text, CLIP's word pattern, byte-level BPE with `</w>` ending each word).

use std::collections::BTreeMap;
use std::path::Path;

use compile::{Error, read_json};
use realistic_vision::model::{END, START};
use serde_json::{Value, json};

/// What `CLIPTokenizer` splits text into before BPE: the special tokens, English contractions,
/// runs of letters, single digits, runs of other non-space characters.
const WORDS: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";
const START_TOKEN: &str = "<|startoftext|>";
const END_TOKEN: &str = "<|endoftext|>";
/// The first line of `merges.txt`, a version header.
const MERGES_HEADER: &str = "#version:";

pub(crate) fn json(vocabulary: &Path, merges: &Path) -> Result<Vec<u8>, Error> {
    let vocabulary: BTreeMap<String, u32> = read_json(vocabulary)?;
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [special(START, START_TOKEN, true), special(END, END_TOKEN, false)],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                { "type": "NFC" },
                { "type": "Replace", "pattern": { "Regex": r"\s+" }, "content": " " },
                { "type": "Lowercase" },
            ],
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": WORDS },
                    "behavior": "Removed",
                    "invert": true,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": true,
                },
            ],
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": END_TOKEN,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "</w>",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabulary,
            "merges": read_merges(merges)?,
        },
    });
    serde_json::to_vec(&tokenizer).map_err(|error| Error::Format(format!("tokenizer: {error}")))
}

/// An added token: the start (normalized like text) or the end.
fn special(id: u32, content: &str, normalized: bool) -> Value {
    json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": normalized,
        "special": true,
    })
}

/// The merges of `merges.txt`, in rank order, as pairs.
fn read_merges(path: &Path) -> Result<Vec<Value>, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| Error::Format(format!("{}: {error}", path.display())))?;
    let lines = text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(MERGES_HEADER));
    lines
        .map(|line| match line.split_once(' ') {
            Some((left, right)) => Ok(json!([left, right])),
            None => Err(Error::Format(format!(
                "merge {line:?}: two symbols expected"
            ))),
        })
        .collect()
}
