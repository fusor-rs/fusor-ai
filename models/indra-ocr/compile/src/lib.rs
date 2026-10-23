//! `indra-ocr compile`: the checkpoint (its ONNX networks, n-gram language models and vocabularies)
//! → a browser bundle.
//!
//! Every network's weights are read through the node that uses them and named `<network>.<name>`:
//! a convolution's by its node's module path (`/stages.0/stages.0.1/body/body.0/body.0.0` →
//! `stages.0.1.body.0.0`), dense ones as `[outputs, kh · kw · inputs]` matrices tap-major (each
//! tap's inputs together, as `kernels::conv` reads them), depthwise ones as `[channels, kh · kw]`;
//! a matrix product's weight transposed to `[outputs, inputs]` and named after the bias added to
//! it; a batch norm folded into a `scale` and a `shift`; layer norms, `Gemm` heads, embeddings and
//! the other initializers by their own names. The order network's layers take the names
//! `blocks::transformer` loads (`ORDER_RENAMES`), and its table heads, which the pipeline leaves
//! off, are not kept. The language models are converted to the runtime's tables
//! (`indra_ocr::lm`), the vocabularies kept as they are; every weight stays f32.

use std::collections::HashSet;
use std::path::Path;

use bundle::{Manifest, Source};
use compile::{BundleWriter, Error, Node, Onnx, Profile as _, Tensors, Values, read_npz};
use indra_ocr::lm::{DEFAULT_BITS, Level, Ngram};
use indra_ocr::model::{LANGUAGE_MODELS, MATH_META, META, NETWORKS, ORDER_FREQUENCIES};
use kernels::DType;

const REPO: &str = "datalab-to/indra-ocr";
const REVISION: &str = "bd02167fdf2005a19962cc15188f8496a2c87d3b";
/// The `Constant` holding the order network's Fourier frequencies.
const ORDER_FREQUENCIES_NODE: &str = "/Constant_5_output_0";
/// The order network's layer tensors as `blocks::transformer` names them.
const ORDER_RENAMES: [(&str, &str); 5] = [
    (".n1.", ".norm1."),
    (".n2.", ".norm2."),
    (".proj", ".out"),
    (".ff.0", ".ff1"),
    (".ff.2", ".ff2"),
];
/// The order network's table heads, which the pipeline leaves off.
const UNUSED: [&str; 2] = ["order.tu", "order.tw"];
const ORDER: &str = "order";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Every weight f32: the networks are small (7 M parameters) and their products narrow.
    F32,
}

impl compile::Profile for Profile {
    const ALL: &'static [Self] = &[Self::F32];
    const ALWAYS_F32: &'static [&'static str] = &[];

    fn name(self) -> &'static str {
        "f32"
    }

    fn matrix_dtype(self, _name: &str) -> DType {
        DType::F32
    }
}

/// One network's tensors on their way into the bundle.
struct Network<'tensors, 'bundle> {
    prefix: &'static str,
    graph: &'tensors Onnx,
    tensors: &'tensors mut Tensors<'bundle, Profile>,
    /// Initializers already written, or folded into another tensor (a product's bias).
    written: HashSet<String>,
}

impl Network<'_, '_> {
    /// The bundle's name of the network's tensor `stem`, or none for one it does not keep.
    fn name(&self, stem: &str) -> Option<String> {
        let mut name = format!("{}.{stem}", self.prefix);
        if UNUSED
            .iter()
            .any(|unused| name.starts_with(&format!("{unused}.")))
        {
            return None;
        }
        if self.prefix == ORDER && stem.starts_with("layers.") {
            for (from, to) in ORDER_RENAMES {
                name = name.replace(from, to);
            }
        }
        Some(name)
    }

    fn write(&mut self) -> Result<(), Error> {
        let biases = self.biases();
        for node in &self.graph.nodes {
            match node.op.as_str() {
                "Conv" => self.convolution(node)?,
                "MatMul" => self.product(node, &biases)?,
                "BatchNormalization" => self.batch_norm(node)?,
                "Add" if biases.iter().any(|(_, bias)| node.inputs.contains(bias)) => {}
                _ => self.as_named(node)?,
            }
        }
        Ok(())
    }

    /// Each matrix product's output and the initializer added to it right after.
    fn biases(&self) -> Vec<(String, String)> {
        let nodes = &self.graph.nodes;
        let added = |output: &str| {
            nodes.iter().find_map(|node| {
                let reads = node.op == "Add" && node.inputs.iter().any(|input| input == output);
                let bias = node
                    .inputs
                    .iter()
                    .find(|input| self.graph.is_initializer(input));
                bias.filter(|_| reads).cloned()
            })
        };
        nodes
            .iter()
            .filter(|node| node.op == "MatMul" && self.graph.is_initializer(&node.inputs[1]))
            .filter_map(|node| Some((node.outputs[0].clone(), added(&node.outputs[0])?)))
            .collect()
    }

    fn convolution(&mut self, node: &Node) -> Result<(), Error> {
        let Some(name) = self.name(&module_path(&node.outputs[0])) else {
            return Ok(());
        };
        let weight = self.graph.tensor(&node.inputs[1])?;
        let outputs = weight.dims[0];
        let per_group = weight.dims[1];
        let taps: usize = weight.dims[2..].iter().product();
        let values = if node.ints("group", &[1])[0] > 1 {
            weight.values.clone()
        } else {
            tap_major(&weight.values, outputs, per_group, taps)
        };
        self.tensors.linear(&name, outputs, &values, None)?;
        if let Some(bias) = node.inputs.get(2) {
            self.tensors
                .vector(&format!("{name}.bias"), &self.graph.tensor(bias)?.values)?;
        }
        Ok(())
    }

    fn product(&mut self, node: &Node, biases: &[(String, String)]) -> Result<(), Error> {
        let source = &node.inputs[1];
        if !self.graph.is_initializer(source) {
            return Ok(());
        }
        let bias = biases
            .iter()
            .find(|(output, _)| *output == node.outputs[0])
            .map(|(_, bias)| bias)
            .ok_or_else(|| Error::Format(format!("{source}: a product without a bias")))?;
        let stem = bias.trim_end_matches(".bias").trim_end_matches("_bias");
        let weight = self.graph.tensor(source)?;
        let (inputs, outputs) = (weight.dims[0], weight.dims[1]);
        let transposed: Vec<f32> = (0..outputs * inputs)
            .map(|at| weight.values[(at % inputs) * outputs + at / inputs])
            .collect();
        self.written.insert(bias.clone());
        let Some(name) = self.name(stem) else {
            return Ok(());
        };
        let bias_values = &self.graph.tensor(bias)?.values;
        self.tensors
            .linear(&name, outputs, &transposed, Some(bias_values))
    }

    /// `x · scale + shift`, the batch norm `(x − mean) / √(var + ε) · weight + bias`.
    fn batch_norm(&mut self, node: &Node) -> Result<(), Error> {
        let [weight, bias, mean, variance] =
            [1, 2, 3, 4].map(|at| self.graph.tensor(&node.inputs[at]));
        let (weight, bias, mean, variance) = (weight?, bias?, mean?, variance?);
        let eps = node.attributes.get("epsilon").map_or(1e-5, |eps| eps[0]) as f32;
        let scale: Vec<f32> = (weight.values.iter().zip(&variance.values))
            .map(|(weight, variance)| weight / (variance + eps).sqrt())
            .collect();
        let shift: Vec<f32> = (bias.values.iter().zip(&mean.values).zip(&scale))
            .map(|((bias, mean), scale)| bias - mean * scale)
            .collect();
        let Some(name) = self.name(node.inputs[1].trim_end_matches(".weight")) else {
            return Ok(());
        };
        self.tensors.vector(&format!("{name}.scale"), &scale)?;
        self.tensors.vector(&format!("{name}.shift"), &shift)
    }

    /// The node's other initializers, each once, by its own name: a matrix as rows of its first
    /// dimension, anything else as a vector.
    fn as_named(&mut self, node: &Node) -> Result<(), Error> {
        for input in &node.inputs {
            if !self.graph.is_initializer(input) || !self.written.insert(input.clone()) {
                continue;
            }
            let Some(name) = self.name(input) else {
                continue;
            };
            let tensor = self.graph.tensor(input)?;
            match tensor.dims.as_slice() {
                [rows, cols] => self
                    .tensors
                    .add(&name, vec![*rows, *cols], &tensor.values)?,
                _ => self.tensors.vector(&name, &tensor.values)?,
            }
        }
        Ok(())
    }
}

/// A node's module path from its output's name: the segments before the operator's, a segment
/// that extends the one before it (`stages.0` → `stages.0.1`) taking its place.
fn module_path(output: &str) -> String {
    let segments = output.trim_start_matches('/').split('/');
    let mut path: Vec<&str> = Vec::new();
    for segment in segments.clone().take(segments.count() - 1) {
        match path.last_mut() {
            Some(last) if segment.starts_with(&format!("{last}.")) => *last = segment,
            _ => path.push(segment),
        }
    }
    path.join(".")
}

/// A dense kernel `[outputs, inputs, taps]` as `[outputs, taps · inputs]`, tap-major.
fn tap_major(values: &[f32], outputs: usize, inputs: usize, taps: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for output in 0..outputs {
        for tap in 0..taps {
            out.extend((0..inputs).map(|input| values[(output * inputs + input) * taps + tap]));
        }
    }
    out
}

/// The `.npz` language model at `path` as the runtime's tables.
fn language_model(path: &Path) -> Result<Ngram, Error> {
    let mut arrays = read_npz(path)?;
    let mut take = |name: &str| {
        arrays
            .remove(name)
            .ok_or_else(|| Error::Format(format!("{}: no array {name}", path.display())))
    };
    let scalar = |values: Values| match values {
        Values::I64(values) => Ok(values[0] as usize),
        _ => Err(Error::Format(format!(
            "{}: a scalar that is not an integer",
            path.display()
        ))),
    };
    let order = scalar(take("order")?.values)?;
    let vocabulary = scalar(take("vocab_size")?.values)?;
    let bits = match take("bits") {
        Ok(bits) => scalar(bits.values)?,
        // A checkpoint without `bits` packs tokens at the default width.
        Err(_) => DEFAULT_BITS,
    };
    let keys = |values: Values| match values {
        Values::U64(keys) => Ok(keys),
        _ => Err(Error::Format(format!(
            "{}: keys that are not u64",
            path.display()
        ))),
    };
    let scores = |values: Values| match values {
        Values::F32(scores) => Ok(scores),
        _ => Err(Error::Format(format!(
            "{}: scores that are not f32",
            path.display()
        ))),
    };
    let levels = (0..order)
        .map(|n| {
            Ok(Level {
                keys: keys(take(&format!("k{n}"))?.values)?,
                scores: scores(take(&format!("v{n}"))?.values)?,
                histories: keys(take(&format!("hk{n}"))?.values)?,
                backoffs: scores(take(&format!("hb{n}"))?.values)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ngram::new(vocabulary, bits, &levels)
        .map_err(|error| Error::Format(format!("{}: {error}", path.display())))
}

pub fn run(checkpoint_dir: &Path, out: &Path, profile: Profile) -> Result<Manifest, Error> {
    let mut writer = BundleWriter::create(out)?;
    let mut tensors = Tensors::new(&mut writer, profile);
    for (file, prefix) in NETWORKS {
        let graph = Onnx::read(&checkpoint_dir.join(file))?;
        let mut network = Network {
            prefix,
            graph: &graph,
            tensors: &mut tensors,
            written: HashSet::new(),
        };
        network.write()?;
        if prefix == "order" {
            let frequencies = &graph.tensor(ORDER_FREQUENCIES_NODE)?.values;
            tensors.vector(ORDER_FREQUENCIES, frequencies)?;
        }
    }
    for file in [META, MATH_META] {
        writer.add_blob_file(file, &checkpoint_dir.join(file))?;
    }
    for (source, part) in LANGUAGE_MODELS {
        let tables = language_model(&checkpoint_dir.join(source))?;
        writer.add_blob(part, &tables.to_bytes())?;
    }
    writer.finish(Source {
        repo: REPO.into(),
        revision: REVISION.into(),
        profile: profile.name().into(),
    })
}
