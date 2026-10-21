//! WAV files: 16-bit PCM encoding, and decoding of 16-bit PCM or 32-bit float to mono.

use crate::Error;

const PCM: u16 = 1;
const FLOAT: u16 = 3;
/// `WAVE_FORMAT_EXTENSIBLE`: the format is the first two bytes of the subformat GUID.
const EXTENSIBLE: u16 = 0xfffe;
const EXTENSIBLE_SUBFORMAT: usize = 24;
const BITS_PER_SAMPLE: usize = 14;
/// 2¹⁵: 16-bit samples decode to `[−1, 1)`.
const PCM_SCALE: f32 = 32768.0;

pub fn encode(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&PCM.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let level = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        out.extend_from_slice(&level.to_le_bytes());
    }
    out
}

/// The samples of a 16-bit PCM or 32-bit float WAV file, its channels averaged, and its sample
/// rate in Hz.
pub fn decode(bytes: &[u8]) -> Result<(Vec<f32>, u32), Error> {
    let not_wav = || Error::Format("not a 16-bit PCM or 32-bit float WAV file".into());
    let body = bytes
        .strip_prefix(b"RIFF")
        .and_then(|rest| rest.get(4..))
        .and_then(|rest| rest.strip_prefix(b"WAVE"))
        .ok_or_else(not_wav)?;
    let format = chunk(body, b"fmt ").ok_or_else(not_wav)?;
    let data = chunk(body, b"data").ok_or_else(not_wav)?;
    let field = |at: usize| {
        format
            .get(at..at + 2)
            .map(|two| u16::from_le_bytes([two[0], two[1]]))
    };
    let tag = match field(0).ok_or_else(not_wav)? {
        EXTENSIBLE => field(EXTENSIBLE_SUBFORMAT).ok_or_else(not_wav)?,
        tag => tag,
    };
    let channels = usize::from(field(2).filter(|&count| count > 0).ok_or_else(not_wav)?);
    let rate = format.get(4..8).ok_or_else(not_wav)?;
    let rate = u32::from_le_bytes([rate[0], rate[1], rate[2], rate[3]]);
    let samples: Vec<f32> = match (tag, field(BITS_PER_SAMPLE)) {
        (PCM, Some(16)) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|two| f32::from(i16::from_le_bytes(*two)) / PCM_SCALE)
            .collect(),
        (FLOAT, Some(32)) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|four| f32::from_le_bytes(*four))
            .collect(),
        _ => return Err(not_wav()),
    };
    let mono = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, rate))
}

/// The body of the first RIFF chunk with this id (chunks are padded to even sizes).
fn chunk<'file>(mut body: &'file [u8], id: &[u8; 4]) -> Option<&'file [u8]> {
    while let Some((header, rest)) = body.split_first_chunk::<8>() {
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let content = rest.get(..size.min(rest.len()))?;
        if header[..4] == *id {
            return Some(content);
        }
        body = rest.get(size.next_multiple_of(2)..)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    #[test]
    fn decodes_what_it_encodes() {
        let (samples, rate) = decode(&encode(&[0.0, 0.5, -1.0], 16_000)).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples, [0.0, 16384.0 / 32768.0, -32767.0 / 32768.0]);
    }
}
