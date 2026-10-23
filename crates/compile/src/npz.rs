//! NumPy `.npz` archives read as checkpoints: each `.npy` array by name, little-endian f32, i64 or
//! u64, C order. The zip is read through its central directory (zip64 sizes and offsets included),
//! each member stored or deflated.

use std::collections::HashMap;
use std::path::Path;

use crate::Error;

const END_OF_DIRECTORY: u32 = 0x0605_4b50;
const ZIP64_END_OF_DIRECTORY: u32 = 0x0606_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const DIRECTORY_ENTRY: u32 = 0x0201_4b50;
const LOCAL_HEADER_LEN: usize = 30;
const END_OF_DIRECTORY_LEN: usize = 22;
const ZIP64_LOCATOR_LEN: usize = 20;
/// A 32-bit size or offset that zip64's extra field holds instead.
const ZIP64_MARK: u32 = u32::MAX;
const ZIP64_EXTRA: u16 = 0x0001;
const STORED: u16 = 0;
const DEFLATED: u16 = 8;
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
const NPY_EXTENSION: &str = ".npy";

/// One array: its shape and its values.
#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub values: Values,
}

#[derive(Debug, Clone)]
pub enum Values {
    F32(Vec<f32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

/// The arrays of the `.npz` at `path`, by name (without `.npy`).
pub fn read_npz(path: &Path) -> Result<HashMap<String, Array>, Error> {
    let bytes = std::fs::read(path).map_err(Error::io(path))?;
    let in_file = |error: Error| Error::Format(format!("{}: {error}", path.display()));
    entries(&bytes)
        .and_then(|entries| {
            entries
                .into_iter()
                .map(|entry| {
                    let name = entry.name.trim_end_matches(NPY_EXTENSION).to_string();
                    Ok((name, array(&entry.data(&bytes)?)?))
                })
                .collect()
        })
        .map_err(in_file)
}

/// A member of the archive, as its central directory entry describes it.
struct Entry {
    name: String,
    method: u16,
    compressed: usize,
    offset: usize,
}

impl Entry {
    /// The member's bytes, inflated.
    fn data(&self, archive: &[u8]) -> Result<Vec<u8>, Error> {
        let header = archive.get(self.offset..).ok_or_else(truncated)?;
        let name_len = usize::from(u16_at(header, 26)?);
        let extra_len = usize::from(u16_at(header, 28)?);
        let start = LOCAL_HEADER_LEN + name_len + extra_len;
        let data = header
            .get(start..start + self.compressed)
            .ok_or_else(truncated)?;
        match self.method {
            STORED => Ok(data.to_vec()),
            DEFLATED => miniz_oxide::inflate::decompress_to_vec(data)
                .map_err(|error| Error::Format(format!("{}: {error:?}", self.name))),
            other => Err(Error::Format(format!("{}: compression {other}", self.name))),
        }
    }
}

/// Every entry of the central directory.
fn entries(archive: &[u8]) -> Result<Vec<Entry>, Error> {
    let end = (0..=archive.len().saturating_sub(END_OF_DIRECTORY_LEN))
        .rev()
        .find(|&at| u32_at(archive, at).ok() == Some(END_OF_DIRECTORY))
        .ok_or_else(|| Error::Format("no zip directory".into()))?;
    let (mut count, mut offset) = (
        u64::from(u16_at(archive, end + 10)?),
        u64::from(u32_at(archive, end + 16)?),
    );
    if let Some(locator) = end.checked_sub(ZIP64_LOCATOR_LEN)
        && u32_at(archive, locator)? == ZIP64_LOCATOR
    {
        let record = u64_at(archive, locator + 8)? as usize;
        if u32_at(archive, record)? != ZIP64_END_OF_DIRECTORY {
            return Err(Error::Format("a zip64 locator without its record".into()));
        }
        (count, offset) = (u64_at(archive, record + 32)?, u64_at(archive, record + 48)?);
    }
    let mut at = offset as usize;
    (0..count)
        .map(|_| {
            let entry = entry(archive, at)?;
            at = entry.1;
            Ok(entry.0)
        })
        .collect()
}

/// The directory entry at `at`, and where the next one starts.
fn entry(archive: &[u8], at: usize) -> Result<(Entry, usize), Error> {
    if u32_at(archive, at)? != DIRECTORY_ENTRY {
        return Err(Error::Format("a corrupt zip directory".into()));
    }
    let method = u16_at(archive, at + 10)?;
    let mut compressed = u64::from(u32_at(archive, at + 20)?);
    let uncompressed = u32_at(archive, at + 24)?;
    let name_len = usize::from(u16_at(archive, at + 28)?);
    let extra_len = usize::from(u16_at(archive, at + 30)?);
    let comment_len = usize::from(u16_at(archive, at + 32)?);
    let mut offset = u64::from(u32_at(archive, at + 42)?);
    let name = archive
        .get(at + 46..at + 46 + name_len)
        .ok_or_else(truncated)?;
    let name = String::from_utf8_lossy(name).into_owned();
    let extra = archive
        .get(at + 46 + name_len..at + 46 + name_len + extra_len)
        .ok_or_else(truncated)?;
    let mut sizes = zip64_sizes(extra)?.into_iter();
    if uncompressed == ZIP64_MARK {
        sizes.next();
    }
    if compressed == u64::from(ZIP64_MARK) {
        compressed = sizes.next().ok_or_else(truncated)?;
    }
    if offset == u64::from(ZIP64_MARK) {
        offset = sizes.next().ok_or_else(truncated)?;
    }
    let entry = Entry {
        name,
        method,
        compressed: compressed as usize,
        offset: offset as usize,
    };
    Ok((entry, at + 46 + name_len + extra_len + comment_len))
}

/// The 64-bit values of an extra field's zip64 record, in order (sizes, then offset).
fn zip64_sizes(mut extra: &[u8]) -> Result<Vec<u64>, Error> {
    while extra.len() >= 4 {
        let (id, len) = (u16_at(extra, 0)?, usize::from(u16_at(extra, 2)?));
        let body = extra.get(4..4 + len).ok_or_else(truncated)?;
        if id == ZIP64_EXTRA {
            return Ok(body
                .as_chunks::<8>()
                .0
                .iter()
                .map(|word| u64::from_le_bytes(*word))
                .collect());
        }
        extra = &extra[4 + len..];
    }
    Ok(Vec::new())
}

/// A `.npy` file's array.
fn array(npy: &[u8]) -> Result<Array, Error> {
    if !npy.starts_with(NPY_MAGIC) {
        return Err(Error::Format("not a .npy array".into()));
    }
    let (header_len, start) = match npy.get(6) {
        Some(1) => (usize::from(u16_at(npy, 8)?), 10),
        _ => (u32_at(npy, 8)? as usize, 12),
    };
    let header = npy.get(start..start + header_len).ok_or_else(truncated)?;
    let header = String::from_utf8_lossy(header);
    if header.contains("'fortran_order': True") {
        return Err(Error::Format("a Fortran-ordered array".into()));
    }
    let data = &npy[start + header_len..];
    let shape = shape(&header)?;
    let values = match descr(&header)? {
        "<f4" => Values::F32(words(data, f32::from_le_bytes)),
        "<i8" => Values::I64(words(data, i64::from_le_bytes)),
        "<u8" => Values::U64(words(data, u64::from_le_bytes)),
        other => return Err(Error::Format(format!("dtype {other} unread"))),
    };
    Ok(Array { shape, values })
}

/// The header's `'descr'`, such as `<f4`.
fn descr(header: &str) -> Result<&str, Error> {
    let after = header
        .split_once("'descr': '")
        .ok_or_else(|| Error::Format("a .npy header without descr".into()))?
        .1;
    Ok(after.split_once('\'').map_or(after, |(descr, _)| descr))
}

/// The header's `'shape'`: `()` for a scalar, `(552,)`, `(3, 4)`.
fn shape(header: &str) -> Result<Vec<usize>, Error> {
    let after = header
        .split_once("'shape': (")
        .ok_or_else(|| Error::Format("a .npy header without shape".into()))?
        .1;
    let inside = after.split_once(')').map_or(after, |(inside, _)| inside);
    inside
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| {
            dim.parse()
                .map_err(|_| Error::Format(format!("a .npy shape of {dim:?}")))
        })
        .collect()
}

fn words<const N: usize, T>(data: &[u8], decode: fn([u8; N]) -> T) -> Vec<T> {
    data.as_chunks::<N>()
        .0
        .iter()
        .map(|word| decode(*word))
        .collect()
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16, Error> {
    Ok(u16::from_le_bytes(at_least(bytes, at)?))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(at_least(bytes, at)?))
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64, Error> {
    Ok(u64::from_le_bytes(at_least(bytes, at)?))
}

fn at_least<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], Error> {
    bytes
        .get(at..at + N)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(truncated)
}

fn truncated() -> Error {
    Error::Format("a truncated zip".into())
}
