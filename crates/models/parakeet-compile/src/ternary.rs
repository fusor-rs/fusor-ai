//! The checkpoint's ternary projections, as `ternary.json` describes them (thrush's export,
//! format `thrush-ternary-v2`): `name.qweight`, `[out, ⌈in / 5⌉]` bytes of five base-3 codes
//! each (element `i` of a row is digit `i % 5` of byte `i / 5`, least significant first), and
//! `name.scales`, one per row and group of 128; the weight is `scale · (code − 1)`.

use std::collections::HashMap;
use std::path::Path;

use compile::{Checkpoint, Error, read_json};
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
        let manifest: Manifest = read_json(&path)?;
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
            || !in_features.is_multiple_of(TERNARY_GROUP)
            || codes.len() != out_features * row_bytes
            || scales.len() != out_features * groups
        {
            return Err(Error::Format(format!(
                "{name}: {} code bytes and {} scales for [{out_features}, {in_features}] in \
                 groups of {} (whole groups of {TERNARY_GROUP} expected)",
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

#[cfg(test)]
mod tests {
    use compile::Checkpoint;

    use super::Export;

    #[test]
    fn unpacks_base_3_codes_by_row_and_group() {
        let (rows, cols) = (2, 256);
        let code = |row: usize, column: usize| (row + column) % 3;
        let mut codes = Vec::new();
        for row in 0..rows {
            for first in (0..cols).step_by(5) {
                let digits = (first..(first + 5).min(cols)).rev();
                codes.push(digits.fold(0u8, |byte, column| byte * 3 + code(row, column) as u8));
            }
        }
        let scales = [0.5f32, 0.25, 2.0, 4.0];
        let dir = std::env::temp_dir().join(format!("fusor-ai-ternary-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = format!(
            r#"{{"format": "thrush-ternary-v2", "names": "hf", "quantized_modules": [
                {{"name": "m", "in_features": {cols}, "out_features": {rows}, "group_size": 128}}
            ]}}"#
        );
        std::fs::write(dir.join("ternary.json"), manifest).unwrap();
        let header = format!(
            r#"{{"m.qweight": {{"dtype": "U8", "shape": [{rows}, {}], "data_offsets": [0, {}]}},
                "m.scales": {{"dtype": "F32", "shape": [{rows}, 2], "data_offsets": [{}, {}]}}}}"#,
            codes.len() / rows,
            codes.len(),
            codes.len(),
            codes.len() + 16
        );
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header.as_bytes());
        file.extend(&codes);
        file.extend(scales.iter().flat_map(|scale| scale.to_le_bytes()));
        std::fs::write(dir.join("model.safetensors"), file).unwrap();

        let checkpoint = Checkpoint::read(&dir.join("model.safetensors")).unwrap();
        let weight = Export::read(&dir)
            .unwrap()
            .weight(&checkpoint, "m")
            .unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(weight.cols, cols);
        for (at, value) in weight.values.iter().enumerate() {
            let (row, column) = (at / cols, at % cols);
            let scale = scales[row * 2 + column / 128];
            assert_eq!(
                *value,
                scale * (code(row, column) as f32 - 1.0),
                "{row}, {column}"
            );
        }
    }
}
