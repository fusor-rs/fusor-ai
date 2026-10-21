use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bundle::{FORMAT, Manifest, Part, Source, TensorEntry};
use kernels::Tensor;

use crate::Error;

/// Parts close once they would grow past this size (a single larger tensor gets its own part).
const PART_BYTES: usize = 48 << 20;
const PARTS_DIR: &str = "parts";
const MANIFEST: &str = "manifest.json";

/// Writes a bundle directory: `manifest.json`, tensor parts under `parts/`, and blobs.
pub struct BundleWriter {
    out: PathBuf,
    parts: Vec<Part>,
    current: Vec<u8>,
    tensors: BTreeMap<String, TensorEntry>,
}

impl BundleWriter {
    pub fn create(out: &Path) -> Result<Self, Error> {
        let parts_dir = out.join(PARTS_DIR);
        std::fs::create_dir_all(&parts_dir).map_err(Error::io(parts_dir))?;
        Ok(Self {
            out: out.to_path_buf(),
            parts: Vec::new(),
            current: Vec::new(),
            tensors: BTreeMap::new(),
        })
    }

    pub fn add_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<(), Error> {
        let bytes = tensor.bytes();
        if !self.current.is_empty() && self.current.len() + bytes.len() > PART_BYTES {
            self.flush()?;
        }
        let entry = TensorEntry {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            part: self.parts.len(),
            offset: self.current.len() as u64,
        };
        self.current.extend_from_slice(bytes);
        if self.tensors.insert(name.to_owned(), entry).is_some() {
            return Err(Error::Format(format!("tensor {name} added twice")));
        }
        Ok(())
    }

    /// Adds a file the runtime reads whole (a tokenizer, a vocabulary) as its own part.
    pub fn add_blob(&mut self, file: &str, bytes: &[u8]) -> Result<(), Error> {
        self.flush()?;
        self.write_part(file, bytes)
    }

    pub fn add_blob_file(&mut self, file: &str, path: &Path) -> Result<(), Error> {
        let bytes = std::fs::read(path).map_err(Error::io(path))?;
        self.add_blob(file, &bytes)
    }

    /// Writes the last tensor part and the manifest.
    pub fn finish(mut self, model: Source) -> Result<Manifest, Error> {
        self.flush()?;
        let manifest = Manifest {
            format: FORMAT,
            model,
            parts: self.parts,
            tensors: self.tensors,
        };
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|error| Error::Format(format!("manifest: {error}")))?;
        let path = self.out.join(MANIFEST);
        std::fs::write(&path, json).map_err(Error::io(path))?;
        Ok(manifest)
    }

    fn flush(&mut self) -> Result<(), Error> {
        if self.current.is_empty() {
            return Ok(());
        }
        let file = format!("{PARTS_DIR}/{:03}.bin", self.parts.len());
        let bytes = std::mem::take(&mut self.current);
        self.write_part(&file, &bytes)
    }

    fn write_part(&mut self, file: &str, bytes: &[u8]) -> Result<(), Error> {
        let path = self.out.join(file);
        std::fs::write(&path, bytes).map_err(Error::io(path))?;
        self.parts.push(Part {
            file: file.into(),
            bytes: bytes.len() as u64,
            blake3: blake3::hash(bytes).to_hex().to_string(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bundle::{Error, Source, Store};
    use kernels::{DType, Tensor};

    use super::BundleWriter;

    fn source() -> Source {
        Source {
            repo: "test/model".into(),
            revision: "0".into(),
            profile: "q7".into(),
        }
    }

    /// Pushes every part in `chunk`-byte writes, the way a streaming loader does.
    fn load(dir: &Path, manifest: &bundle::Manifest, chunk: usize) -> Result<Store, Error> {
        let mut store = Store::new(manifest.clone())?;
        for (index, part) in manifest.parts.iter().enumerate() {
            let bytes = std::fs::read(dir.join(&part.file)).unwrap();
            for piece in bytes.chunks(chunk) {
                store.write(index, piece)?;
            }
            store.end_part(index)?;
        }
        Ok(store)
    }

    #[test]
    fn written_bundle_streams_back_into_tensors_and_blobs() {
        let dir = std::env::temp_dir().join(format!("fusor-ai-writer-{}", std::process::id()));
        let values: Vec<f32> = (0..3 * 64).map(|i| (i as f32 - 90.0) / 9.0).collect();
        let mut writer = BundleWriter::create(&dir).unwrap();
        let norm = Tensor::encode(DType::F32, vec![3], &[1.0, 2.0, 3.0]).unwrap();
        writer.add_tensor("norm", &norm).unwrap();
        let weight = Tensor::encode(DType::Q7b32, vec![3, 64], &values).unwrap();
        writer.add_tensor("weight", &weight).unwrap();
        writer.add_blob("vocab.json", b"{\"a\": 1}").unwrap();
        let manifest = writer.finish(source()).unwrap();

        let mut store = load(&dir, &manifest, 7).unwrap();
        assert_eq!(store.take("norm").unwrap().bytes(), norm.bytes());
        assert_eq!(store.take("weight").unwrap().bytes(), weight.bytes());
        assert_eq!(store.take_blob("vocab.json").unwrap(), b"{\"a\": 1}");

        let mut tampered = manifest;
        tampered.parts[0].blake3 = "0".repeat(64);
        assert!(matches!(load(&dir, &tampered, 7), Err(Error::Integrity(_))));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
