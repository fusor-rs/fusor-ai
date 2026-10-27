//! `spark-x2-6 parity`: each turn of a conversation, teacher-forced, against the reference's
//! goldens (`reference/`).
//!
//! Each turn reads the reference's prompt after the reference's earlier replies, so errors do not
//! compound from turn to turn. Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); each
//! turn's prompt ids must be the reference's, one for one, its reply's first token the
//! reference's, and most of the reply's next tokens, read after the reference's, the
//! reference's too.

use std::path::Path;

use anyhow::{Result, ensure};
use compile::Checkpoint;
use harness::parity::compare;
use kernels::numpy::argmax;
use spark_x2_6::model::HIDDEN;
use spark_x2_6::prompt::{REPLY_END, Replied};
use spark_x2_6::session::Engine;

use crate::Conversation;

/// The share of a reply's teacher-forced tokens whose likeliest next token must be the
/// reference's: the quantized weights move near ties.
const AGREEMENT: f64 = 0.9;

/// One turn's tensors (`reference/dump.py` documents them).
struct Answered {
    ids: Vec<i64>,
    reply: Vec<u32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

/// The reference's tensors for one conversation.
struct Goldens {
    conversation: Conversation,
    turns: Vec<Answered>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let count = (0..)
            .take_while(|index| file.contains(&format!("turn.{index}.ids")))
            .count();
        let turns = (0..count).map(|index| {
            let name = |part: &str| format!("turn.{index}.{part}");
            let reply = file.i64(&name("reply"))?.into_iter().map(u32::try_from);
            Ok(Answered {
                ids: file.i64(&name("ids"))?,
                reply: reply.collect::<Result<_, _>>()?,
                hidden: file.f32(&name("hidden"))?,
                logits: file.f32(&name("logits"))?,
            })
        });
        Ok(Self {
            conversation: serde_json::from_slice(&file.u8("conversation")?)?,
            turns: turns.collect::<Result<_>>()?,
        })
    }
}

pub fn run(engine: &mut Engine, path: &Path) -> Result<()> {
    let goldens = Goldens::read(path)?;
    println!("# {}", path.display());
    let mut history = Vec::new();
    let turns = goldens.conversation.turns.iter().zip(&goldens.turns);
    for (index, (message, golden)) in turns.enumerate() {
        let prompt = engine.prompt(&message.system, &history, &message.text, message.thinking)?;
        let ids = prompt.iter().map(|&id| i64::from(id));
        ensure!(
            ids.eq(golden.ids.iter().copied()),
            "turn {index}: the prompt's ids are not the reference's"
        );
        let prefill = engine.prefill(&prompt);
        ensure!(
            argmax(&prefill.logits) == golden.reply[0] as usize,
            "turn {index}: the reply's first token is not the reference's"
        );
        println!(
            "prefill {index}: {} ids identical\n{}\n{}",
            prompt.len(),
            compare(
                &format!("prefill {index} hidden"),
                &prefill.hidden,
                &golden.hidden[..HIDDEN]
            )?,
            compare(
                &format!("prefill {index} logits"),
                &prefill.logits,
                &golden.logits
            )?
        );
        decode(engine, &prompt, golden, index)?;
        let written = match golden.reply.split_last() {
            Some((&REPLY_END, written)) => written,
            _ => &golden.reply,
        };
        history.push(Replied {
            text: message.text.clone(),
            reply: engine.text(written)?,
        });
    }
    Ok(())
}

/// The reply's tokens read after the reference's, but its last: each one's state against the
/// reference's, and how many of the likeliest tokens after them are the reference's.
fn decode(engine: &mut Engine, prompt: &[u32], golden: &Answered, index: usize) -> Result<()> {
    let reply = &golden.reply;
    let forced = engine.force(prompt, &reply[..reply.len() - 1]);
    ensure!(!forced.is_empty(), "turn {index}: a reply of one token");
    let hidden: Vec<f32> = forced
        .iter()
        .flat_map(|prediction| prediction.hidden.iter().copied())
        .collect();
    let agreed = forced
        .iter()
        .zip(&reply[1..])
        .filter(|(prediction, token)| argmax(&prediction.logits) == **token as usize)
        .count();
    let share = agreed as f64 / forced.len() as f64;
    ensure!(
        share >= AGREEMENT,
        "decode {index}: {agreed} of {} next tokens the reference's",
        forced.len()
    );
    println!(
        "{} ({agreed} of {} next tokens the reference's)",
        compare(
            &format!("decode {index} hidden"),
            &hidden,
            &golden.hidden[HIDDEN..]
        )?,
        forced.len()
    );
    Ok(())
}
