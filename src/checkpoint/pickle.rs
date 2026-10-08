use std::io::Cursor;

use candle_core::DType;
use candle_core::pickle::{Object, Stack};

use super::Result;
use crate::Error;

const MAX_DEPTH: usize = 64;

const STORAGE_CLASSES: [(&str, DType); 6] = [
    ("FloatStorage", DType::F32),
    ("DoubleStorage", DType::F64),
    ("HalfStorage", DType::F16),
    ("BFloat16Storage", DType::BF16),
    ("ByteStorage", DType::U8),
    ("LongStorage", DType::I64),
];

fn checkpoint_error(message: impl Into<String>) -> Error {
    Error::Checkpoint(message.into())
}

fn storage_dtype(module: &str, class: &str) -> Option<DType> {
    if module != "torch" {
        return None;
    }
    STORAGE_CLASSES
        .iter()
        .find(|(name, _)| *name == class)
        .map(|(_, dtype)| *dtype)
}

fn is_rebuild_tensor(object: &Object) -> bool {
    matches!(
        object,
        Object::Class { module_name, class_name }
            if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2"
    )
}

fn check_allowed(object: &Object, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(checkpoint_error("pickle is nested too deeply"));
    }
    match object {
        Object::Int(_)
        | Object::Long(_)
        | Object::Float(_)
        | Object::Unicode(_)
        | Object::Bool(_)
        | Object::None => Ok(()),
        Object::Class {
            module_name,
            class_name,
        } => match storage_dtype(module_name, class_name) {
            Some(_) => Ok(()),
            None => Err(checkpoint_error(format!(
                "pickle references the disallowed global `{module_name}.{class_name}`"
            ))),
        },
        Object::Tuple(items) | Object::List(items) => items
            .iter()
            .try_for_each(|item| check_allowed(item, depth + 1)),
        Object::Dict(entries) => entries.iter().try_for_each(|(key, value)| {
            check_allowed(key, depth + 1)?;
            check_allowed(value, depth + 1)
        }),
        Object::Reduce { callable, args } => {
            if !is_rebuild_tensor(callable) {
                return Err(match &**callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } => checkpoint_error(format!(
                        "pickle calls the disallowed global `{module_name}.{class_name}`"
                    )),
                    other => checkpoint_error(format!(
                        "pickle calls an unsupported object: {}",
                        describe(other)
                    )),
                });
            }
            check_allowed(args, depth + 1)
        }
        Object::PersistentLoad(inner) => check_allowed(inner, depth + 1),
        Object::Build { .. } => Err(checkpoint_error(
            "pickle uses the BUILD opcode, which is not supported",
        )),
        Object::Mark => Err(checkpoint_error("pickle contains a dangling MARK")),
    }
}

fn describe(object: &Object) -> &'static str {
    match object {
        Object::Class { .. } => "class",
        Object::Int(_) | Object::Long(_) => "integer",
        Object::Float(_) => "float",
        Object::Unicode(_) => "string",
        Object::Bool(_) => "bool",
        Object::None => "None",
        Object::Tuple(_) => "tuple",
        Object::List(_) => "list",
        Object::Mark => "mark",
        Object::Dict(_) => "dict",
        Object::Reduce { .. } => "call",
        Object::Build { .. } => "build",
        Object::PersistentLoad(_) => "persistent id",
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Object> {
    let outcome = std::panic::catch_unwind(|| {
        let mut cursor = Cursor::new(bytes);
        let mut stack = Stack::empty();
        stack.read_loop(&mut cursor)?;
        stack.finalize()
    });
    match outcome {
        Ok(Ok(object)) => {
            check_allowed(&object, 0)?;
            Ok(object)
        }
        Ok(Err(error)) => Err(checkpoint_error(format!("malformed pickle: {error}"))),
        Err(_) => Err(checkpoint_error("malformed pickle: the parser panicked")),
    }
}

fn as_usize(object: &Object) -> Option<usize> {
    match object {
        Object::Int(value) => usize::try_from(*value).ok(),
        Object::Long(value) => usize::try_from(*value).ok(),
        _ => None,
    }
}

fn as_f64(object: &Object) -> Option<f64> {
    match object {
        Object::Float(value) => Some(*value),
        Object::Int(value) => Some(f64::from(*value)),
        Object::Long(value) => Some(*value as f64),
        _ => None,
    }
}

pub(crate) struct Dict<'a> {
    entries: &'a [(Object, Object)],
    path: String,
}

impl<'a> Dict<'a> {
    pub(crate) fn new(object: &'a Object, path: &str) -> Result<Self> {
        match object {
            Object::Dict(entries) => Ok(Self {
                entries,
                path: path.to_string(),
            }),
            other => Err(checkpoint_error(format!(
                "`{path}` is a {}, expected a dict",
                describe(other)
            ))),
        }
    }

    fn empty(path: String) -> Self {
        Self { entries: &[], path }
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn entries(&self) -> &'a [(Object, Object)] {
        self.entries
    }

    pub(crate) fn get(&self, key: &str) -> Option<&'a Object> {
        self.entries
            .iter()
            .rev()
            .find(|(k, _)| matches!(k, Object::Unicode(name) if name == key))
            .map(|(_, value)| value)
    }

    fn present(&self, key: &str) -> Option<&'a Object> {
        self.get(key).filter(|value| !matches!(value, Object::None))
    }

    fn wrong_type(&self, key: &str, expected: &str, found: &Object) -> Error {
        checkpoint_error(format!(
            "`{}.{key}` is a {}, expected {expected}",
            self.path,
            describe(found)
        ))
    }

    fn missing(&self, key: &str) -> Error {
        checkpoint_error(format!("`{}` has no entry `{key}`", self.path))
    }

    fn convert<T>(
        &self,
        key: &str,
        expected: &str,
        convert: impl Fn(&Object) -> Option<T>,
    ) -> Result<Option<T>> {
        match self.present(key) {
            None => Ok(None),
            Some(value) => convert(value)
                .map(Some)
                .ok_or_else(|| self.wrong_type(key, expected, value)),
        }
    }

    pub(crate) fn dict(&self, key: &str) -> Result<Dict<'a>> {
        let value = self.present(key).ok_or_else(|| self.missing(key))?;
        Dict::new(value, &format!("{}.{key}", self.path))
    }

    pub(crate) fn dict_or_empty(&self, key: &str) -> Result<Dict<'a>> {
        match self.present(key) {
            None => Ok(Dict::empty(format!("{}.{key}", self.path))),
            Some(value) => Dict::new(value, &format!("{}.{key}", self.path)),
        }
    }

    pub(crate) fn usize(&self, key: &str) -> Result<usize> {
        self.usize_opt(key)?.ok_or_else(|| self.missing(key))
    }

    pub(crate) fn usize_opt(&self, key: &str) -> Result<Option<usize>> {
        self.convert(key, "a non-negative integer", as_usize)
    }

    pub(crate) fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        Ok(self.usize_opt(key)?.unwrap_or(default))
    }

    pub(crate) fn f64(&self, key: &str) -> Result<f64> {
        self.f64_opt(key)?.ok_or_else(|| self.missing(key))
    }

    pub(crate) fn f64_opt(&self, key: &str) -> Result<Option<f64>> {
        self.convert(key, "a number", as_f64)
    }

    pub(crate) fn f64_or(&self, key: &str, default: f64) -> Result<f64> {
        Ok(self.f64_opt(key)?.unwrap_or(default))
    }

    pub(crate) fn bool(&self, key: &str) -> Result<bool> {
        let value = self.present(key).ok_or_else(|| self.missing(key))?;
        match value {
            Object::Bool(flag) => Ok(*flag),
            other => Err(self.wrong_type(key, "a bool", other)),
        }
    }

    pub(crate) fn bool_or(&self, key: &str, default: bool) -> Result<bool> {
        match self.present(key) {
            None => Ok(default),
            Some(Object::Bool(flag)) => Ok(*flag),
            Some(other) => Err(self.wrong_type(key, "a bool", other)),
        }
    }

    pub(crate) fn string_or(&self, key: &str, default: &str) -> Result<String> {
        let parsed = self.convert(key, "a string", |value| match value {
            Object::Unicode(text) => Some(text.clone()),
            _ => None,
        })?;
        Ok(parsed.unwrap_or_else(|| default.to_string()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TensorRecord {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) storage_key: String,
    pub(crate) storage_numel: usize,
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
}

fn usize_tuple(object: &Object) -> Option<Vec<usize>> {
    match object {
        Object::Tuple(items) => items.iter().map(as_usize).collect(),
        _ => None,
    }
}

fn is_c_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1usize;
    for (&dim, &step) in shape.iter().zip(stride).rev() {
        if dim != 1 && step != expected {
            return false;
        }
        expected = expected.saturating_mul(dim);
    }
    true
}

fn parse_tensor(name: &str, path: &str, object: &Object) -> Result<TensorRecord> {
    let bad = |what: &str| checkpoint_error(format!("`{path}.{name}`: {what}"));
    let Object::Reduce { callable, args } = object else {
        return Err(bad(&format!(
            "is a {}, expected a tensor",
            describe(object)
        )));
    };
    if !is_rebuild_tensor(callable) {
        return Err(bad("is not a plain tensor"));
    }
    let Object::Tuple(args) = &**args else {
        return Err(bad("has malformed tensor arguments"));
    };
    if args.len() < 4 {
        return Err(bad("has too few tensor arguments"));
    }
    let Object::PersistentLoad(persistent) = &args[0] else {
        return Err(bad("does not reference a storage"));
    };
    let Object::Tuple(storage) = &**persistent else {
        return Err(bad("has a malformed storage reference"));
    };
    let [tag, class, key, _location, numel] = storage.as_slice() else {
        return Err(bad("has a malformed storage reference"));
    };
    if !matches!(tag, Object::Unicode(tag) if tag == "storage") {
        return Err(bad("has an unexpected persistent id"));
    }
    let Object::Class {
        module_name,
        class_name,
    } = class
    else {
        return Err(bad("has a malformed storage type"));
    };
    let dtype = storage_dtype(module_name, class_name)
        .ok_or_else(|| bad(&format!("uses the unsupported storage type `{class_name}`")))?;
    let Object::Unicode(storage_key) = key else {
        return Err(bad("has a malformed storage key"));
    };
    if storage_key.is_empty() || !storage_key.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad("has a malformed storage key"));
    }
    let storage_numel = as_usize(numel).ok_or_else(|| bad("has a malformed storage size"))?;
    let offset = as_usize(&args[1]).ok_or_else(|| bad("has a malformed storage offset"))?;
    let shape = usize_tuple(&args[2]).ok_or_else(|| bad("has a malformed shape"))?;
    let stride = usize_tuple(&args[3]).ok_or_else(|| bad("has a malformed stride"))?;
    if shape.len() != stride.len() {
        return Err(bad("has a shape and stride of different ranks"));
    }
    if !is_c_contiguous(&shape, &stride) {
        return Err(bad(&format!(
            "is not contiguous (shape {shape:?}, stride {stride:?})"
        )));
    }
    Ok(TensorRecord {
        name: name.to_string(),
        dtype,
        storage_key: storage_key.clone(),
        storage_numel,
        offset,
        shape,
    })
}

pub(crate) fn tensor_records(dict: &Dict<'_>) -> Result<Vec<TensorRecord>> {
    let mut records: Vec<TensorRecord> = Vec::with_capacity(dict.entries().len());
    for (key, value) in dict.entries() {
        let Object::Unicode(name) = key else {
            return Err(checkpoint_error(format!(
                "`{}` has a non-string key",
                dict.path()
            )));
        };
        if name == "_metadata" && matches!(value, Object::Dict(_)) {
            continue;
        }
        if records.iter().any(|record| &record.name == name) {
            return Err(checkpoint_error(format!(
                "`{}` contains the tensor `{name}` twice",
                dict.path()
            )));
        }
        records.push(parse_tensor(name, dict.path(), value)?);
    }
    Ok(records)
}
