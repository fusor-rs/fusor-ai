//! The talker's prompt: the assistant turn that speaks the text (`<|im_start|>assistant\n`, the
//! text, `<|im_end|>\n<|im_start|>assistant\n`), the codec side's language and speaker rows, and,
//! when the voice's transcript is given, the transcript's tokens and the reference's frames. The
//! text side of every row goes through the text projection, the codec side through the codec
//! embedding; the model reads their sums. Text tokens the prompt has no row left for follow the
//! frames one at a time (`trailing`).

use bundle::Tokenizer;

use crate::Error;
use crate::model::{
    CODEC_BOS, CODEC_PAD, CODEC_THINK, CODEC_THINK_BOS, CODEC_THINK_EOS, Frame, HIDDEN, Language,
    TTS_BOS, TTS_EOS, TTS_PAD, Talker,
};
use crate::predictor::Predictor;

/// Tokens of `<|im_start|>assistant\n`.
const ROLE_TOKENS: usize = 3;
/// Tokens of `<|im_end|>\n<|im_start|>assistant\n`, which close the text and open the reply.
const REPLY_TOKENS: usize = 5;
/// Tokens of `<|im_end|>\n`, which close a transcript.
const CLOSING_TOKENS: usize = 2;

/// A text's tokens as the speaking turn holds them.
pub struct Speech {
    /// The turn's opening tokens.
    pub role: Vec<u32>,
    pub text: Vec<u32>,
}

/// # Errors
/// When the tokenizer fails.
pub fn speech_ids(tokenizer: &Tokenizer, text: &str) -> Result<Speech, Error> {
    let turn = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
    let ids = tokenizer.encode(&turn)?;
    let text = ids[ROLE_TOKENS..ids.len() - REPLY_TOKENS].to_vec();
    Ok(Speech {
        role: ids[..ROLE_TOKENS].to_vec(),
        text,
    })
}

/// A transcript's tokens as the reference turn holds them.
///
/// # Errors
/// When the tokenizer fails.
pub fn reference_ids(tokenizer: &Tokenizer, transcript: &str) -> Result<Vec<u32>, Error> {
    let turn = format!("<|im_start|>assistant\n{transcript}<|im_end|>\n");
    let ids = tokenizer.encode(&turn)?;
    Ok(ids[ROLE_TOKENS..ids.len() - CLOSING_TOKENS].to_vec())
}

/// The reference voice's transcript and frames, when the prompt continues it.
#[derive(Clone, Copy)]
pub struct Reference<'voice> {
    pub ids: &'voice [u32],
    pub frames: &'voice [Frame],
}

/// What a prompt is made of.
#[derive(Clone, Copy)]
pub struct Conditioning<'request> {
    pub language: Language,
    pub speech: &'request Speech,
    /// The speaker encoder's embedding of the voice.
    pub speaker: &'request [f32],
    pub reference: Option<Reference<'request>>,
}

/// The rows the talker reads first, and the text rows it reads along with the frames after.
pub struct Prompt {
    /// `rows × HIDDEN`.
    pub rows: Vec<f32>,
    /// `HIDDEN` values each: the text row added to frame `i`'s row, while there are any.
    pub trailing: Vec<f32>,
    /// What follows the frames once the trailing rows are out.
    pub pad: Vec<f32>,
}

impl Prompt {
    pub(crate) fn new(talker: &Talker, predictor: &Predictor, conditioning: &Conditioning) -> Self {
        let Conditioning {
            language,
            speech,
            speaker,
            reference,
        } = *conditioning;
        let [pad, bos, eos] = [TTS_PAD, TTS_BOS, TTS_EOS].map(|id| text_rows(talker, &[id]));
        let mut rows = text_rows(talker, &speech.role);
        let tags = [
            CODEC_THINK,
            CODEC_THINK_BOS,
            language.codec_id(),
            CODEC_THINK_EOS,
        ];
        for tag in codec_rows(talker, &tags).as_chunks::<HIDDEN>().0 {
            rows.extend(add(tag, &pad));
        }
        rows.extend(add(speaker, &pad));
        rows.extend(add(&codec_rows(talker, &[CODEC_PAD]), &bos));
        let opening = codec_rows(talker, &[CODEC_BOS]);
        let trailing = match reference {
            None => {
                let text = text_rows(talker, &speech.text);
                rows.extend(add(&text[..HIDDEN], &opening));
                [&text[HIDDEN..], &eos].concat()
            }
            Some(reference) => {
                let tokens = [reference.ids, &speech.text].concat();
                let mut text = [text_rows(talker, &tokens), eos].concat();
                let mut codec = opening;
                for frame in reference.frames {
                    codec.extend(predictor.frame_row(&talker.lm, frame));
                }
                let missing = codec.len().saturating_sub(text.len()) / HIDDEN;
                text.extend(pad.repeat(missing));
                rows.extend(add(&text[..codec.len()], &codec));
                text.split_off(codec.len())
            }
        };
        Self {
            rows,
            trailing,
            pad,
        }
    }
}

/// The text embeddings of `ids`, one row each.
fn text_rows(talker: &Talker, ids: &[u32]) -> Vec<f32> {
    let mut rows = vec![0.0; ids.len() * HIDDEN];
    talker.text(ids, &mut rows);
    rows
}

/// The codec embeddings of `ids`, one row each.
fn codec_rows(talker: &Talker, ids: &[u32]) -> Vec<f32> {
    let mut rows = vec![0.0; ids.len() * HIDDEN];
    talker.lm.embed(ids, &mut rows);
    rows
}

fn add<'values>(
    left: &'values [f32],
    right: &'values [f32],
) -> impl Iterator<Item = f32> + 'values {
    left.iter().zip(right).map(|(first, second)| first + second)
}
