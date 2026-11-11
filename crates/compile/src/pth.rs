//! PyTorch `torch.save` archives (`.pth`) of a state dict read as checkpoints: the pickle
//! `<archive>/data.pkl` maps each tensor's name to `torch._utils._rebuild_tensor_v2(storage,
//! offset, size, stride, …)`, its storage a persistent id `('storage', <dtype>Storage, key,
//! location, numel)` naming the member `<archive>/data/<key>`, stored uncompressed. Only
//! contiguous float tensors are read, through the pickle opcodes `torch.save` writes for them.

use std::collections::HashMap;
use std::ops::Range;

use crate::checkpoint::{Dtype, Entry};
use crate::{Error, npz};

const PICKLE_MEMBER: &str = "/data.pkl";
const DATA_MEMBER: &str = "/data/";
const STORAGE_KIND: &str = "storage";
const STORAGE_SUFFIX: &str = "Storage";
const STORAGE_DTYPES: [(&str, Dtype); 3] = [
    ("BFloat16", Dtype::Bf16),
    ("Float", Dtype::F32),
    ("Half", Dtype::F16),
];
const ORDERED_DICT: (&str, &str) = ("collections", "OrderedDict");
const REBUILD_TENSOR: (&str, &str) = ("torch._utils", "_rebuild_tensor_v2");

/// The pickle opcodes a state dict uses.
mod opcode {
    pub(super) const PROTO: u8 = 0x80;
    pub(super) const GLOBAL: u8 = b'c';
    pub(super) const BINPUT: u8 = b'q';
    pub(super) const LONG_BINPUT: u8 = b'r';
    pub(super) const BINGET: u8 = b'h';
    pub(super) const LONG_BINGET: u8 = b'j';
    pub(super) const EMPTY_TUPLE: u8 = b')';
    pub(super) const EMPTY_DICT: u8 = b'}';
    pub(super) const MARK: u8 = b'(';
    pub(super) const BINUNICODE: u8 = b'X';
    pub(super) const BININT: u8 = b'J';
    pub(super) const BININT1: u8 = b'K';
    pub(super) const BININT2: u8 = b'M';
    pub(super) const TUPLE: u8 = b't';
    pub(super) const TUPLE1: u8 = 0x85;
    pub(super) const TUPLE2: u8 = 0x86;
    pub(super) const TUPLE3: u8 = 0x87;
    pub(super) const NEWFALSE: u8 = 0x89;
    pub(super) const BINPERSID: u8 = b'Q';
    pub(super) const REDUCE: u8 = b'R';
    pub(super) const SETITEM: u8 = b's';
    pub(super) const SETITEMS: u8 = b'u';
    pub(super) const BUILD: u8 = b'b';
    pub(super) const STOP: u8 = b'.';
}

/// Every tensor of the archive `bytes`: its name, dtype, shape and byte range in `bytes`.
pub(crate) fn tensors(bytes: &[u8]) -> Result<HashMap<String, Entry>, Error> {
    let members = npz::entries(bytes)?;
    let (prefix, pickle) = members
        .iter()
        .find_map(|member| {
            let prefix = member.name.strip_suffix(PICKLE_MEMBER)?;
            Some((prefix, member))
        })
        .ok_or_else(|| Error::Format(format!("no {PICKLE_MEMBER} in the archive")))?;
    let by_name: HashMap<&str, &npz::Entry> = members
        .iter()
        .map(|member| (member.name.as_str(), member))
        .collect();
    let mut reader = Reader {
        bytes: &bytes[pickle.stored(bytes)?],
        at: 0,
    };
    let Value::Dict(state) = Machine::default().run(&mut reader)? else {
        return Err(Error::Format("the pickle is not a state dict".into()));
    };
    state
        .into_iter()
        .map(|item| {
            let (Value::Str(name), Value::Tensor(tensor)) = item else {
                return Err(Error::Format(
                    "the state dict holds more than tensors".into(),
                ));
            };
            let member = format!("{prefix}{DATA_MEMBER}{}", tensor.key);
            let member = by_name
                .get(member.as_str())
                .ok_or_else(|| Error::Format(format!("{name}: no member {member}")))?;
            let entry = tensor.entry(member.stored(bytes)?, &name)?;
            Ok((name, entry))
        })
        .collect()
}

#[derive(Clone)]
struct Storage {
    dtype: Dtype,
    key: String,
}

#[derive(Clone)]
struct Tensor {
    dtype: Dtype,
    key: String,
    /// In elements of the storage.
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl Tensor {
    /// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)`.
    fn rebuild(args: &[Value]) -> Result<Self, Error> {
        let [
            Value::Storage(storage),
            offset,
            Value::Tuple(shape),
            Value::Tuple(stride),
            ..,
        ] = args
        else {
            return Err(Error::Format(
                "a tensor rebuilt from unexpected arguments".into(),
            ));
        };
        let dimensions = |values: &[Value]| {
            values
                .iter()
                .map(Value::count)
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            dtype: storage.dtype,
            key: storage.key.clone(),
            offset: offset.count()?,
            shape: dimensions(shape)?,
            stride: dimensions(stride)?,
        })
    }

    /// The checkpoint entry of this tensor, in a member stored at `member` in the archive.
    fn entry(self, member: Range<usize>, name: &str) -> Result<Entry, Error> {
        let mut expected = 1;
        let contiguous = self
            .shape
            .iter()
            .zip(&self.stride)
            .rev()
            .all(|(&size, &step)| {
                let fits = size == 1 || step == expected;
                expected *= size;
                fits
            });
        if !contiguous {
            return Err(Error::Format(format!("{name}: not contiguous")));
        }
        let size = self
            .dtype
            .size()
            .ok_or_else(|| Error::Format(format!("{name}: {:?} is not read", self.dtype)))?;
        let start = member.start + self.offset * size;
        let end = start + self.shape.iter().product::<usize>() * size;
        if end > member.end {
            return Err(Error::Format(format!(
                "{name}: past the end of its storage"
            )));
        }
        Ok(Entry {
            dtype: self.dtype,
            shape: self.shape,
            data_offsets: [start, end],
        })
    }
}

#[derive(Clone)]
enum Value {
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(Storage),
    Tensor(Tensor),
}

impl Value {
    fn count(&self) -> Result<usize, Error> {
        match self {
            Value::Int(value) => usize::try_from(*value)
                .map_err(|_| Error::Format(format!("a negative size or offset, {value}"))),
            _ => Err(Error::Format(
                "a size or offset that is not an integer".into(),
            )),
        }
    }
}

struct Reader<'bytes> {
    bytes: &'bytes [u8],
    at: usize,
}

impl<'bytes> Reader<'bytes> {
    fn take(&mut self, len: usize) -> Result<&'bytes [u8], Error> {
        let end = self.at.checked_add(len).ok_or_else(corrupt)?;
        let taken = self.bytes.get(self.at..end).ok_or_else(corrupt)?;
        self.at = end;
        Ok(taken)
    }

    fn array<const LEN: usize>(&mut self) -> Result<[u8; LEN], Error> {
        let taken = self.take(LEN)?;
        taken.first_chunk().copied().ok_or_else(corrupt)
    }

    fn byte(&mut self) -> Result<u8, Error> {
        self.array().map(u8::from_le_bytes)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        self.array().map(u32::from_le_bytes)
    }

    /// A UTF-8 string of `len` bytes.
    fn text(&mut self, len: usize) -> Result<String, Error> {
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| Error::Format("a pickled string that is not UTF-8".into()))
    }

    /// The text up to the next newline, which is consumed.
    fn line(&mut self) -> Result<String, Error> {
        let rest = &self.bytes[self.at..];
        let len = rest
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(corrupt)?;
        let line = self.text(len);
        self.at += 1;
        line
    }
}

/// The pickle virtual machine: its stack, the stack heights `MARK` recorded and its memo.
#[derive(Default)]
struct Machine {
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
}

impl Machine {
    fn run(mut self, reader: &mut Reader) -> Result<Value, Error> {
        loop {
            let code = reader.byte()?;
            if code == opcode::STOP {
                return self.pop();
            }
            self.execute(code, reader)?;
        }
    }

    fn execute(&mut self, code: u8, reader: &mut Reader) -> Result<(), Error> {
        match code {
            opcode::PROTO => {
                reader.byte()?;
            }
            opcode::GLOBAL => {
                let module = reader.line()?;
                let name = reader.line()?;
                self.stack.push(Value::Global(module, name));
            }
            opcode::BINPUT => self.memoize(u32::from(reader.byte()?))?,
            opcode::LONG_BINPUT => self.memoize(reader.u32()?)?,
            opcode::BINGET => self.recall(u32::from(reader.byte()?))?,
            opcode::LONG_BINGET => self.recall(reader.u32()?)?,
            opcode::EMPTY_TUPLE => self.stack.push(Value::Tuple(Vec::new())),
            opcode::EMPTY_DICT => self.stack.push(Value::Dict(Vec::new())),
            opcode::MARK => self.marks.push(self.stack.len()),
            opcode::BINUNICODE => {
                let len = reader.u32()? as usize;
                self.stack.push(Value::Str(reader.text(len)?));
            }
            opcode::BININT => {
                let value = i32::from_le_bytes(reader.array()?);
                self.stack.push(Value::Int(i64::from(value)));
            }
            opcode::BININT1 => self.stack.push(Value::Int(i64::from(reader.byte()?))),
            opcode::BININT2 => {
                let value = u16::from_le_bytes(reader.array()?);
                self.stack.push(Value::Int(i64::from(value)));
            }
            opcode::NEWFALSE => self.stack.push(Value::Int(0)),
            other => self.operate(other)?,
        }
        Ok(())
    }

    fn operate(&mut self, code: u8) -> Result<(), Error> {
        match code {
            opcode::TUPLE => {
                let mark = self.marks.pop().ok_or_else(corrupt)?;
                self.tuple_from(mark)?;
            }
            opcode::TUPLE1 => self.tuple_of(1)?,
            opcode::TUPLE2 => self.tuple_of(2)?,
            opcode::TUPLE3 => self.tuple_of(3)?,
            opcode::BINPERSID => self.storage()?,
            opcode::REDUCE => self.reduce()?,
            opcode::SETITEM => {
                let value = self.pop()?;
                let key = self.pop()?;
                self.dict()?.push((key, value));
            }
            opcode::SETITEMS => self.set_items()?,
            opcode::BUILD => {
                self.pop()?;
            }
            other => {
                return Err(Error::Format(format!(
                    "pickle opcode {other:#04x} is not read"
                )));
            }
        }
        Ok(())
    }

    fn pop(&mut self) -> Result<Value, Error> {
        self.stack.pop().ok_or_else(corrupt)
    }

    fn memoize(&mut self, index: u32) -> Result<(), Error> {
        let top = self.stack.last().ok_or_else(corrupt)?.clone();
        self.memo.insert(index, top);
        Ok(())
    }

    fn recall(&mut self, index: u32) -> Result<(), Error> {
        let value = self
            .memo
            .get(&index)
            .ok_or_else(|| Error::Format(format!("pickle memo {index} is unset")))?;
        self.stack.push(value.clone());
        Ok(())
    }

    fn tuple_of(&mut self, len: usize) -> Result<(), Error> {
        let start = self.stack.len().checked_sub(len).ok_or_else(corrupt)?;
        self.tuple_from(start)
    }

    fn tuple_from(&mut self, start: usize) -> Result<(), Error> {
        if start > self.stack.len() {
            return Err(corrupt());
        }
        let items = self.stack.split_off(start);
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    fn dict(&mut self) -> Result<&mut Vec<(Value, Value)>, Error> {
        match self.stack.last_mut() {
            Some(Value::Dict(items)) => Ok(items),
            _ => Err(Error::Format(
                "pickle sets an item of what is not a dict".into(),
            )),
        }
    }

    fn set_items(&mut self) -> Result<(), Error> {
        let mark = self.marks.pop().ok_or_else(corrupt)?;
        if mark > self.stack.len() {
            return Err(corrupt());
        }
        let mut items = self.stack.split_off(mark).into_iter();
        let mut pairs = Vec::new();
        while let Some(key) = items.next() {
            let value = items.next().ok_or_else(corrupt)?;
            pairs.push((key, value));
        }
        self.dict()?.extend(pairs);
        Ok(())
    }

    /// The persistent id `('storage', <dtype>Storage, key, location, numel)` on top.
    fn storage(&mut self) -> Result<(), Error> {
        let Value::Tuple(id) = self.pop()? else {
            return Err(Error::Format("a persistent id that is not a tuple".into()));
        };
        let [
            Value::Str(kind),
            Value::Global(_, class),
            Value::Str(key),
            ..,
        ] = &id[..]
        else {
            return Err(Error::Format(
                "a persistent id that is not a storage".into(),
            ));
        };
        let dtype = class
            .strip_suffix(STORAGE_SUFFIX)
            .and_then(|name| STORAGE_DTYPES.iter().find(|(known, _)| *known == name))
            .filter(|_| kind == STORAGE_KIND)
            .map(|&(_, dtype)| dtype)
            .ok_or_else(|| Error::Format(format!("{kind} {class} is not read")))?;
        self.stack.push(Value::Storage(Storage {
            dtype,
            key: key.clone(),
        }));
        Ok(())
    }

    fn reduce(&mut self) -> Result<(), Error> {
        let (Value::Tuple(args), Value::Global(module, name)) = (self.pop()?, self.pop()?) else {
            return Err(Error::Format("pickle calls what is not a global".into()));
        };
        let value = match (module.as_str(), name.as_str()) {
            ORDERED_DICT => Value::Dict(Vec::new()),
            REBUILD_TENSOR => Value::Tensor(Tensor::rebuild(&args)?),
            _ => {
                return Err(Error::Format(format!(
                    "pickle calls {module}.{name}, which is not read"
                )));
            }
        };
        self.stack.push(value);
        Ok(())
    }
}

fn corrupt() -> Error {
    Error::Format("a truncated or corrupt pickle".into())
}
