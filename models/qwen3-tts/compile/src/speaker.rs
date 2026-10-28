//! The speaker encoder's convs as `blocks::ecapa`'s matrices.

use compile::{Checkpoint, Error, Tensors};
use qwen3_tts::speaker::ECAPA;

use crate::Profile;

const SOURCE: &str = "speaker_encoder";

pub(crate) fn write(checkpoint: &Checkpoint, tensors: &mut Tensors<Profile>) -> Result<(), Error> {
    let mut conv = |name: String, source: String| {
        tensors.conv1d(
            checkpoint,
            &format!("speaker.{name}"),
            &format!("{SOURCE}.{source}"),
        )
    };
    conv("blocks.0.conv".into(), "blocks.0.conv".into())?;
    for block in 1..ECAPA.channels.len() - 1 {
        let source = format!("blocks.{block}");
        for part in ["tdnn1", "tdnn2"] {
            conv(format!("{source}.{part}"), format!("{source}.{part}.conv"))?;
        }
        for part in 0..ECAPA.scale - 1 {
            let res2net = format!("{source}.res2net_block.blocks.{part}.conv");
            conv(format!("{source}.res2net.{part}"), res2net)?;
        }
        conv(
            format!("{source}.se.squeeze"),
            format!("{source}.se_block.conv1"),
        )?;
        conv(
            format!("{source}.se.excite"),
            format!("{source}.se_block.conv2"),
        )?;
    }
    conv("mfa".into(), "mfa.conv".into())?;
    conv("pool.attention".into(), "asp.tdnn.conv".into())?;
    conv("pool.score".into(), "asp.conv".into())?;
    conv("out".into(), "fc".into())
}
