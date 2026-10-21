//! The checkpoint's ternary projections, as `ternary.json` describes them (thrush's export,
//! format `thrush-ternary-v2`): `name.qweight`, `[out, ⌈in / 5⌉]` bytes of five base-3 codes
//! each (element `i` of a row is digit `i % 5` of byte `i / 5`, least significant first), and
//! `name.scales`, one per row and group of 128; the weight is `scale · (code − 1)`.

use std::collections::HashMap;
use std::path::Path;

use compile::{Checkpoint, Error};
use kernels::TERNARY_GROUP;
use serde::Deserialize;

const MANIFEST: &str = "ternary.json";
const FORMAT: &str = "thrush-ternary-v2";
const NAMES: &str = "hf";
const CODES_PER_BYTE: usize = 5;
const BASE: u8 = 3;

#[derive(Deserialize)]
struct Manifest {
    format: String,
    names: String,
    quantized_modules: Vec<Module>,
}

#[derive(Deserialize)]
struct Module {
    name: String,
    in_features: usize,
    out_features: usize,
    group_size: usize,
}

/// A ternary projection's weight, row-major.
pub(crate) struct Weight {
    pub(crate) values: Vec<f32>,
    pub(crate) cols: usize,
}

/// The ternary projections of the checkpoint, by module name.
pub(crate) struct Export {
    modules: HashMap<String, Module>,
}

impl Export {
    pub(crate) fn read(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(MANIFEST);
        let json = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let manifest: Manifest = serde_json::from_slice(&json)
            .map_err(|error| Error::Format(format!("{}: {error}", path.display())))?;
        if manifest.format != FORMAT || manifest.names != NAMES {
            return Err(Error::Format(format!(
                "{}: a {} export with {} names, not {FORMAT} with {NAMES}",
                path.display(),
                manifest.format,
                manifest.names
            )));
        }
        let modules = manifest.quantized_modules.into_iter();
        Ok(Self {
            modules: modules
                .map(|module| (module.name.clone(), module))
                .collect(),
        })
    }

    /// Module `name`'s weight, `[out, in]`.
    pub(crate) fn weight(&self, checkpoint: &Checkpoint, name: &str) -> Result<Weight, Error> {
        let module = self
            .modules
            .get(name)
            .ok_or_else(|| Error::Format(format!("{name} is not ternary in {MANIFEST}")))?;
        let Module {
            in_features,
            out_features,
            ..
        } = *module;
        let row_bytes = in_features.div_ceil(CODES_PER_BYTE);
        let codes = checkpoint.u8(&format!("{name}.qweight"))?;
        let scales = checkpoint.f32(&format!("{name}.scales"))?;
        let groups = in_features / TERNARY_GROUP;
        if module.group_size != TERNARY_GROUP
            || codes.len() != out_features * row_bytes
            || scales.len() != out_features * groups
        {
            return Err(Error::Format(format!(
                "{name}: {} code bytes and {} scales for [{out_features}, {in_features}] in \
                 groups of {}",
                codes.len(),
                scales.len(),
                module.group_size
            )));
        }
        let mut values = Vec::with_capacity(out_features * in_features);
        for (row, bytes) in codes.chunks_exact(row_bytes).enumerate() {
            let row_scales = &scales[row * groups..(row + 1) * groups];
            values.extend((0..in_features).map(|column| {
                let place = BASE.pow((column % CODES_PER_BYTE) as u32);
                let code = bytes[column / CODES_PER_BYTE] / place % BASE;
                row_scales[column / TERNARY_GROUP] * (f32::from(code) - 1.0)
            }));
        }
        Ok(Weight {
            values,
            cols: in_features,
        })
    }
}
