use std::collections::HashMap;

use fusor_ai_kernels::Tensor;

use crate::Error;
use crate::manifest::{Manifest, Part};

/// A byte range of a part and where its bytes go.
struct Slot {
    offset: u64,
    len: u64,
    target: Target,
}

enum Target {
    Tensor(String),
    Blob,
}

struct Progress {
    received: u64,
    hasher: blake3::Hasher,
    done: bool,
}

impl Progress {
    fn new() -> Self {
        Self {
            received: 0,
            hasher: blake3::Hasher::new(),
            done: false,
        }
    }
}

pub struct Store {
    manifest: Manifest,
    tensors: HashMap<String, Tensor>,
    /// Bytes of the blob parts, by part index.
    blobs: HashMap<usize, Vec<u8>>,
    /// Per part, its slots sorted by offset.
    slots: Vec<Vec<Slot>>,
    progress: Vec<Progress>,
}

impl Store {
    /// Allocates every tensor (zeroed) and the routing table of every part.
    pub fn new(manifest: Manifest) -> Result<Self, Error> {
        let mut slots: Vec<Vec<Slot>> = manifest.parts.iter().map(|_| Vec::new()).collect();
        let mut tensors = HashMap::new();
        for (name, entry) in &manifest.tensors {
            let tensor = Tensor::zeroed(entry.dtype, entry.shape.clone());
            let part = slots
                .get_mut(entry.part)
                .ok_or_else(|| Error::Format(format!("{name}: no part {}", entry.part)))?;
            part.push(Slot {
                offset: entry.offset,
                len: tensor.bytes().len() as u64,
                target: Target::Tensor(name.clone()),
            });
            tensors.insert(name.clone(), tensor);
        }
        let mut blobs = HashMap::new();
        for (index, part) in manifest.parts.iter().enumerate() {
            if manifest.is_blob(index) {
                slots[index].push(Slot {
                    offset: 0,
                    len: part.bytes,
                    target: Target::Blob,
                });
                blobs.insert(index, Vec::new());
            }
        }
        for (part, list) in manifest.parts.iter().zip(&mut slots) {
            list.sort_by_key(|slot| slot.offset);
            check_layout(part, list)?;
        }
        let progress = manifest.parts.iter().map(|_| Progress::new()).collect();
        Ok(Self {
            manifest,
            tensors,
            blobs,
            slots,
            progress,
        })
    }

    /// The next `bytes` of part `part` (writes arrive in order).
    pub fn write(&mut self, part: usize, bytes: &[u8]) -> Result<(), Error> {
        let progress = &mut self.progress[part];
        let start = progress.received;
        let end = start + bytes.len() as u64;
        if progress.done || end > self.manifest.parts[part].bytes {
            let file = &self.manifest.parts[part].file;
            return Err(Error::Format(format!("{file}: write past its end")));
        }
        progress.hasher.update(bytes);
        progress.received = end;
        let first = self.slots[part].partition_point(|slot| slot.offset + slot.len <= start);
        for slot in &self.slots[part][first..] {
            if slot.offset >= end {
                break;
            }
            let from = start.max(slot.offset);
            let to = end.min(slot.offset + slot.len);
            let source = &bytes[(from - start) as usize..(to - start) as usize];
            match &slot.target {
                Target::Tensor(name) => {
                    let tensor = self
                        .tensors
                        .get_mut(name)
                        .expect("every slot's tensor exists");
                    let at = (from - slot.offset) as usize;
                    tensor.bytes_mut()[at..at + source.len()].copy_from_slice(source);
                }
                Target::Blob => {
                    let blob = self
                        .blobs
                        .get_mut(&part)
                        .expect("every blob part has bytes");
                    blob.extend_from_slice(source);
                }
            }
        }
        Ok(())
    }

    /// Checks part `part`'s size and BLAKE3 once all its bytes were written.
    pub fn end_part(&mut self, part: usize) -> Result<(), Error> {
        let entry = &self.manifest.parts[part];
        let progress = &mut self.progress[part];
        if progress.received != entry.bytes {
            return Err(Error::Format(format!(
                "{}: {} of {} bytes",
                entry.file, progress.received, entry.bytes
            )));
        }
        let hash = progress.hasher.finalize().to_hex();
        if hash.as_str() != entry.blake3 {
            return Err(Error::Integrity(format!(
                "{}: BLAKE3 {hash} ≠ {}",
                entry.file, entry.blake3
            )));
        }
        progress.done = true;
        Ok(())
    }

    /// Forgets what was written to part `part`, so it can be written again (a cached copy that
    /// failed its hash is refetched).
    pub fn restart_part(&mut self, part: usize) {
        self.progress[part] = Progress::new();
        if let Some(blob) = self.blobs.get_mut(&part) {
            blob.clear();
        }
    }

    pub fn complete(&self) -> bool {
        self.progress.iter().all(|progress| progress.done)
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn take(&mut self, name: &str) -> Result<Tensor, Error> {
        self.tensors
            .remove(name)
            .ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    /// A tensor that only some models, profiles or layers have.
    pub fn take_optional(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    /// The bytes of the blob part named `file`, once it was ended.
    pub fn take_blob(&mut self, file: &str) -> Result<Vec<u8>, Error> {
        let part = self.manifest.part_index(file)?;
        if !self.progress[part].done {
            return Err(Error::Format(format!("{file} is not loaded")));
        }
        self.blobs
            .remove(&part)
            .ok_or_else(|| Error::Format(format!("{file} is not a blob, or was taken")))
    }
}

/// Slots of one part must not overlap and must fit inside it.
fn check_layout(part: &Part, slots: &[Slot]) -> Result<(), Error> {
    let mut end = 0;
    for slot in slots {
        if slot.offset < end || slot.offset + slot.len > part.bytes {
            return Err(Error::Format(format!(
                "{}: overlapping or out-of-range tensors",
                part.file
            )));
        }
        end = slot.offset + slot.len;
    }
    Ok(())
}
