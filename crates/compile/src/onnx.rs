//! ONNX models read as checkpoints: the graph's nodes in order (operator, inputs, outputs and
//! numeric attributes), and its tensors by name, the initializers and the values of its
//! `Constant` nodes, as f32. A compiler names and lays out each weight by the node that reads it.
//!
//! The protobuf is decoded directly, only the fields this needs: `ModelProto.graph`, the graph's
//! `node` and `initializer`, a node's `input`, `output`, `op_type` and `attribute`, an attribute's
//! `name`, `f`, `i`, `t`, `floats` and `ints`, and a tensor's `dims`, `data_type`, `float_data`,
//! `int64_data`, `name` and `raw_data` (onnx.proto3).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::Error;

const MODEL_GRAPH: u64 = 7;
const GRAPH_NODE: u64 = 1;
const GRAPH_INITIALIZER: u64 = 5;
const NODE_INPUT: u64 = 1;
const NODE_OUTPUT: u64 = 2;
const NODE_OP_TYPE: u64 = 4;
const NODE_ATTRIBUTE: u64 = 5;
const ATTRIBUTE_NAME: u64 = 1;
const ATTRIBUTE_FLOAT: u64 = 2;
const ATTRIBUTE_INT: u64 = 3;
const ATTRIBUTE_TENSOR: u64 = 5;
const ATTRIBUTE_FLOATS: u64 = 7;
const ATTRIBUTE_INTS: u64 = 8;
const TENSOR_DIMS: u64 = 1;
const TENSOR_DATA_TYPE: u64 = 2;
const TENSOR_FLOAT_DATA: u64 = 4;
const TENSOR_INT64_DATA: u64 = 7;
const TENSOR_NAME: u64 = 8;
const TENSOR_RAW_DATA: u64 = 9;
/// `TensorProto.DataType` values this reads.
const FLOAT: u64 = 1;
const INT64: u64 = 7;
const CONSTANT: &str = "Constant";

/// Protobuf wire types.
const VARINT: u64 = 0;
const FIXED64: u64 = 1;
const LENGTH_DELIMITED: u64 = 2;
const FIXED32: u64 = 5;

pub struct Onnx {
    pub nodes: Vec<Node>,
    tensors: HashMap<String, OnnxTensor>,
    /// The names of the tensors that are initializers, not constants.
    initializers: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// The numeric attributes, each as its values (one for a scalar).
    pub attributes: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Default)]
pub struct OnnxTensor {
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

impl Node {
    /// Attribute `name`'s values as whole numbers, or `default` when the node has none.
    pub fn ints(&self, name: &str, default: &[usize]) -> Vec<usize> {
        match self.attributes.get(name) {
            Some(values) => values.iter().map(|&value| value as usize).collect(),
            None => default.to_vec(),
        }
    }
}

impl Onnx {
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(Error::io(path))?;
        let in_file = |error: Error| Error::Format(format!("{}: {error}", path.display()));
        let mut graph = None;
        for field in Fields::new(&bytes) {
            let (number, value) = field.map_err(in_file)?;
            if number == MODEL_GRAPH {
                graph = Some(value.bytes().map_err(in_file)?);
            }
        }
        let graph = graph.ok_or_else(|| Error::Format(format!("{}: no graph", path.display())))?;
        Self::graph(graph).map_err(in_file)
    }

    fn graph(bytes: &[u8]) -> Result<Self, Error> {
        let mut nodes = Vec::new();
        let mut tensors = HashMap::new();
        for field in Fields::new(bytes) {
            let (number, value) = field?;
            match number {
                GRAPH_NODE => nodes.push(node(value.bytes()?)?),
                GRAPH_INITIALIZER => {
                    let (name, tensor) = tensor(value.bytes()?)?;
                    tensors.insert(name, tensor);
                }
                _ => {}
            }
        }
        let initializers = tensors.keys().cloned().collect();
        for node in &nodes {
            if node.op == CONSTANT
                && let Some(value) = node.attributes.get("value")
            {
                let dims = vec![value.len()];
                let values = value.iter().map(|&value| value as f32).collect();
                tensors.insert(node.outputs[0].clone(), OnnxTensor { dims, values });
            }
        }
        Ok(Self {
            nodes,
            tensors,
            initializers,
        })
    }

    pub fn is_initializer(&self, name: &str) -> bool {
        self.initializers.contains(name)
    }

    /// Initializer or constant `name`.
    pub fn tensor(&self, name: &str) -> Result<&OnnxTensor, Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Format(format!("no tensor {name}")))
    }
}

/// A node's operator, inputs, outputs and numeric attributes (a `Constant`'s tensor flattened
/// into its `value`).
fn node(bytes: &[u8]) -> Result<Node, Error> {
    let mut node = Node::default();
    for field in Fields::new(bytes) {
        let (number, value) = field?;
        match number {
            NODE_INPUT => node.inputs.push(value.text()?),
            NODE_OUTPUT => node.outputs.push(value.text()?),
            NODE_OP_TYPE => node.op = value.text()?,
            NODE_ATTRIBUTE => {
                let (name, values) = attribute(value.bytes()?)?;
                node.attributes.insert(name, values);
            }
            _ => {}
        }
    }
    Ok(node)
}

fn attribute(bytes: &[u8]) -> Result<(String, Vec<f64>), Error> {
    let (mut name, mut values) = (String::new(), Vec::new());
    for field in Fields::new(bytes) {
        let (number, value) = field?;
        match number {
            ATTRIBUTE_NAME => name = value.text()?,
            ATTRIBUTE_FLOAT => values.push(f64::from(value.float()?)),
            ATTRIBUTE_INT => values.push(value.int()? as f64),
            ATTRIBUTE_TENSOR => {
                let (_, tensor) = tensor(value.bytes()?)?;
                values.extend(tensor.values.iter().map(|&value| f64::from(value)));
            }
            ATTRIBUTE_FLOATS => values.extend(value.floats()?.into_iter().map(f64::from)),
            ATTRIBUTE_INTS => values.extend(value.ints()?.into_iter().map(|int| int as f64)),
            _ => {}
        }
    }
    Ok((name, values))
}

/// A tensor's name and its values as f32 (from f32 or i64).
fn tensor(bytes: &[u8]) -> Result<(String, OnnxTensor), Error> {
    let (mut name, mut data_type) = (String::new(), FLOAT);
    let (mut dims, mut values, mut raw) = (Vec::new(), Vec::new(), Vec::new());
    for field in Fields::new(bytes) {
        let (number, value) = field?;
        match number {
            TENSOR_DIMS => dims.extend(value.ints()?.into_iter().map(|dim| dim as usize)),
            TENSOR_DATA_TYPE => data_type = value.int()? as u64,
            TENSOR_FLOAT_DATA => values.extend(value.floats()?),
            TENSOR_INT64_DATA => values.extend(value.ints()?.into_iter().map(|int| int as f32)),
            TENSOR_NAME => name = value.text()?,
            TENSOR_RAW_DATA => raw = value.bytes()?.to_vec(),
            _ => {}
        }
    }
    match data_type {
        FLOAT => values.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word)),
        ),
        INT64 => values.extend(
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|word| i64::from_le_bytes(*word) as f32),
        ),
        other if raw.is_empty() && values.is_empty() => {
            return Err(Error::Format(format!("{name}: data type {other} unread")));
        }
        _ => {}
    }
    Ok((name, OnnxTensor { dims, values }))
}

/// A field's value as the wire carries it.
#[derive(Clone, Copy)]
enum Value<'bytes> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'bytes [u8]),
    Fixed32(u32),
}

impl<'bytes> Value<'bytes> {
    fn bytes(self) -> Result<&'bytes [u8], Error> {
        match self {
            Value::Bytes(bytes) => Ok(bytes),
            _ => Err(Error::Format("a length-delimited field expected".into())),
        }
    }

    fn text(self) -> Result<String, Error> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| Error::Format("a string field that is not UTF-8".into()))
    }

    fn int(self) -> Result<i64, Error> {
        match self {
            Value::Varint(value) | Value::Fixed64(value) => Ok(value as i64),
            _ => Err(Error::Format("an integer field expected".into())),
        }
    }

    fn float(self) -> Result<f32, Error> {
        match self {
            Value::Fixed32(bits) => Ok(f32::from_bits(bits)),
            _ => Err(Error::Format("a float field expected".into())),
        }
    }

    /// A repeated integer field's values: one, or a packed run of varints.
    fn ints(self) -> Result<Vec<i64>, Error> {
        match self {
            Value::Bytes(mut packed) => {
                let mut ints = Vec::new();
                while !packed.is_empty() {
                    ints.push(varint(&mut packed)? as i64);
                }
                Ok(ints)
            }
            other => Ok(vec![other.int()?]),
        }
    }

    /// A repeated float field's values: one, or a packed run of little-endian floats.
    fn floats(self) -> Result<Vec<f32>, Error> {
        match self {
            Value::Bytes(packed) => Ok(packed
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()),
            other => Ok(vec![other.float()?]),
        }
    }
}

/// The fields of a message, in order: each one's number and value.
struct Fields<'bytes> {
    rest: &'bytes [u8],
}

impl<'bytes> Fields<'bytes> {
    fn new(bytes: &'bytes [u8]) -> Self {
        Self { rest: bytes }
    }

    fn field(&mut self) -> Result<(u64, Value<'bytes>), Error> {
        let key = varint(&mut self.rest)?;
        let value = match key & 0b111 {
            VARINT => Value::Varint(varint(&mut self.rest)?),
            FIXED64 => Value::Fixed64(u64::from_le_bytes(take(&mut self.rest)?)),
            LENGTH_DELIMITED => {
                let len = varint(&mut self.rest)? as usize;
                let (bytes, rest) = self.rest.split_at_checked(len).ok_or_else(truncated)?;
                self.rest = rest;
                Value::Bytes(bytes)
            }
            FIXED32 => Value::Fixed32(u32::from_le_bytes(take(&mut self.rest)?)),
            other => return Err(Error::Format(format!("wire type {other} unread"))),
        };
        Ok((key >> 3, value))
    }
}

impl<'bytes> Iterator for Fields<'bytes> {
    type Item = Result<(u64, Value<'bytes>), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let field = self.field();
        if field.is_err() {
            self.rest = &[];
        }
        Some(field)
    }
}

fn varint(bytes: &mut &[u8]) -> Result<u64, Error> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = bytes.split_first().ok_or_else(truncated)?;
        *bytes = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::Format("a varint longer than 64 bits".into()))
}

fn take<const N: usize>(bytes: &mut &[u8]) -> Result<[u8; N], Error> {
    let (head, rest) = bytes.split_first_chunk::<N>().ok_or_else(truncated)?;
    *bytes = rest;
    Ok(*head)
}

fn truncated() -> Error {
    Error::Format("truncated protobuf".into())
}
