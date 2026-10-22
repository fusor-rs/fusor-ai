//! WAV files: 16-bit PCM encoding, and decoding of 16-bit PCM or 32-bit float to mono.

use kernels::Error;

const PCM: u16 = 1;
const FLOAT: u16 = 3;
/// The RIFF header, the `fmt ` chunk of a PCM file and the `data` chunk's header.
const HEADER_BYTES: usize = 44;
const SIZE_FIELD_BYTES: usize = 4;
/// The bytes after the RIFF size field that precede the samples.
const HEADER_AFTER_SIZE: u32 = 36;
const PCM_FORMAT_BYTES: u32 = 16;
const MONO: u16 = 1;
const SAMPLE_BYTES: u16 = 2;
const SAMPLE_BITS: u16 = 16;
const FLOAT_BITS: u16 = 32;
/// Offsets inside the `fmt ` chunk.
const FORMAT_TAG: usize = 0;
const CHANNELS: usize = 2;
const SAMPLE_RATE: usize = 4;
/// `WAVE_FORMAT_EXTENSIBLE`: the format is the first two bytes of the subformat GUID.
const EXTENSIBLE: u16 = 0xfffe;
const EXTENSIBLE_SUBFORMAT: usize = 24;
const BITS_PER_SAMPLE: usize = 14;
/// 2¹⁵: 16-bit samples decode to `[−1, 1)`.
const PCM_SCALE: f32 = 32768.0;

pub fn encode(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * usize::from(SAMPLE_BYTES);
    let mut out = Vec::with_capacity(HEADER_BYTES + data_len);
    let data_len = data_len as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(HEADER_AFTER_SIZE + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&PCM_FORMAT_BYTES.to_le_bytes());
    out.extend_from_slice(&PCM.to_le_bytes());
    out.extend_from_slice(&MONO.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * u32::from(SAMPLE_BYTES);
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&SAMPLE_BYTES.to_le_bytes());
    out.extend_from_slice(&SAMPLE_BITS.to_le_bytes());
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
        .and_then(|rest| rest.get(SIZE_FIELD_BYTES..))
        .and_then(|rest| rest.strip_prefix(b"WAVE"))
        .ok_or_else(not_wav)?;
    let format = chunk(body, b"fmt ").ok_or_else(not_wav)?;
    let data = chunk(body, b"data").ok_or_else(not_wav)?;
    let field = |at: usize| {
        format
            .get(at..at + 2)
            .map(|two| u16::from_le_bytes([two[0], two[1]]))
    };
    let tag = match field(FORMAT_TAG).ok_or_else(not_wav)? {
        EXTENSIBLE => field(EXTENSIBLE_SUBFORMAT).ok_or_else(not_wav)?,
        tag => tag,
    };
    let channels = field(CHANNELS)
        .filter(|&count| count > 0)
        .ok_or_else(not_wav)?;
    let channels = usize::from(channels);
    let rate = format
        .get(SAMPLE_RATE..)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or_else(not_wav)?;
    let rate = u32::from_le_bytes(*rate);
    let samples: Vec<f32> = match (tag, field(BITS_PER_SAMPLE)) {
        (PCM, Some(SAMPLE_BITS)) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|two| f32::from(i16::from_le_bytes(*two)) / PCM_SCALE)
            .collect(),
        (FLOAT, Some(FLOAT_BITS)) => data
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
