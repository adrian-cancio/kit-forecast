use std::collections::HashMap;
use std::io::{self, Read, Seek};

use candle_core::{Device, Tensor};
use zip::ZipArchive;
use zip::result::ZipError;

use super::Result;
use super::kit_config::KitConfig;
use super::loader::{CheckpointInfo, SourceKind};
use super::pickle::{self, Dict, TensorRecord};
use super::weights::{Found, ModelWeights, validate_layout};
use crate::Error;

const PICKLE_SUFFIX: &str = "data.pkl";
const MAX_PICKLE_BYTES: u64 = 64 << 20;
const F32_BYTES: usize = 4;

fn zip_error(context: &str, error: ZipError) -> Error {
    Error::Checkpoint(format!("{context}: {error}"))
}

fn find_pickle_entry<R: Read + Seek>(archive: &ZipArchive<R>) -> Result<String> {
    let mut candidates = archive
        .file_names()
        .filter(|name| *name == PICKLE_SUFFIX || name.ends_with("/data.pkl"));
    let first = candidates
        .next()
        .ok_or_else(|| Error::Checkpoint("archive has no `data.pkl` entry".to_string()))?;
    if candidates.next().is_some() {
        return Err(Error::Checkpoint(
            "archive has more than one `data.pkl` entry".to_string(),
        ));
    }
    Ok(first.to_string())
}

fn read_pickle<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let entry = archive
        .by_name(name)
        .map_err(|e| zip_error("cannot open the pickle entry", e))?;
    if entry.size() > MAX_PICKLE_BYTES {
        return Err(Error::Checkpoint(format!(
            "pickle entry is {} bytes, more than the {MAX_PICKLE_BYTES} byte limit",
            entry.size()
        )));
    }
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry
        .take(MAX_PICKLE_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Checkpoint(format!("cannot read the pickle entry: {e}")))?;
    Ok(bytes)
}

fn check_byteorder<R: Read + Seek>(archive: &mut ZipArchive<R>, prefix: &str) -> Result<()> {
    let name = format!("{prefix}byteorder");
    let Ok(entry) = archive.by_name(&name) else {
        return Ok(());
    };
    let mut text = String::new();
    entry
        .take(16)
        .read_to_string(&mut text)
        .map_err(|e| Error::Checkpoint(format!("cannot read the byteorder entry: {e}")))?;
    if text.trim() != "little" {
        return Err(Error::Checkpoint(format!(
            "unsupported byte order {:?}, only little-endian checkpoints can be read",
            text.trim()
        )));
    }
    Ok(())
}

struct Source<'a> {
    state: Dict<'a>,
    kind: SourceKind,
    ema_decay: Option<f64>,
}

fn select_source<'a>(root: &Dict<'a>) -> Result<Source<'a>> {
    let ema = root.dict_or_empty("ema")?;
    if !ema.entries().is_empty() {
        return Ok(if ema.get("shadow").is_some() {
            Source {
                state: ema.dict("shadow")?,
                kind: SourceKind::Ema,
                ema_decay: ema.f64_opt("decay")?,
            }
        } else {
            Source {
                state: ema,
                kind: SourceKind::Ema,
                ema_decay: None,
            }
        });
    }
    let model = root.dict_or_empty("model")?;
    if !model.entries().is_empty() {
        return Ok(Source {
            state: model,
            kind: SourceKind::Model,
            ema_decay: None,
        });
    }
    Err(Error::Checkpoint(
        "checkpoint has neither `ema` nor `model` weights".to_string(),
    ))
}

fn element_count(shape: &[usize], name: &str) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::Checkpoint(format!("tensor `{name}` has an overflowing shape")))
}

fn read_tensor<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    record: &TensorRecord,
    device: &Device,
) -> Result<Tensor> {
    let name = record.name.as_str();
    let fail = |what: String| Error::Checkpoint(format!("tensor `{name}`: {what}"));
    let numel = element_count(&record.shape, name)?;
    let end = record
        .offset
        .checked_add(numel)
        .ok_or_else(|| fail("storage range overflows".to_string()))?;
    if end > record.storage_numel {
        return Err(fail(format!(
            "needs elements {}..{end} but its storage holds only {}",
            record.offset, record.storage_numel
        )));
    }
    let entry_name = format!("{prefix}data/{}", record.storage_key);
    let mut entry = archive
        .by_name(&entry_name)
        .map_err(|e| fail(format!("cannot open storage `{entry_name}`: {e}")))?;
    let storage_bytes = record
        .storage_numel
        .checked_mul(F32_BYTES)
        .ok_or_else(|| fail("storage size overflows".to_string()))?;
    if entry.size() != storage_bytes as u64 {
        return Err(fail(format!(
            "storage `{entry_name}` holds {} bytes but the pickle declares {storage_bytes}",
            entry.size()
        )));
    }
    let io_error = |e: io::Error| fail(format!("cannot read storage `{entry_name}`: {e}"));
    io::copy(
        &mut entry.by_ref().take((record.offset * F32_BYTES) as u64),
        &mut io::sink(),
    )
    .map_err(io_error)?;
    let mut bytes = vec![0u8; numel * F32_BYTES];
    entry.read_exact(&mut bytes).map_err(io_error)?;
    io::copy(&mut entry, &mut io::sink()).map_err(io_error)?;
    let values: Vec<f32> = bytes
        .chunks_exact(F32_BYTES)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(fail(format!("non-finite value at flat index {index}")));
    }
    Ok(Tensor::from_vec(values, record.shape.clone(), device)?)
}

pub(crate) fn read_pt<R: Read + Seek>(
    reader: R,
    device: &Device,
) -> Result<(KitConfig, ModelWeights, CheckpointInfo)> {
    let mut archive =
        ZipArchive::new(reader).map_err(|e| zip_error("not a readable zip archive", e))?;
    let pickle_name = find_pickle_entry(&archive)?;
    let prefix = pickle_name[..pickle_name.len() - PICKLE_SUFFIX.len()].to_string();
    check_byteorder(&mut archive, &prefix)?;
    let bytes = read_pickle(&mut archive, &pickle_name)?;
    let object = pickle::parse(&bytes)?;
    drop(bytes);
    let root = Dict::new(&object, "checkpoint")?;

    let config = KitConfig::from_pickle(&root.dict("config")?)?;
    let source = select_source(&root)?;
    let mut records = pickle::tensor_records(&source.state)?;
    let specs = config.model.tensor_specs()?;
    validate_layout(
        &specs,
        records.iter().map(|r| Found {
            name: &r.name,
            shape: &r.shape,
            dtype: r.dtype,
        }),
    )?;

    records.sort_by_key(|r| (r.storage_key.len(), r.storage_key.clone()));
    let mut tensors = HashMap::with_capacity(records.len());
    for record in &records {
        let tensor = read_tensor(&mut archive, &prefix, record, device)?;
        tensors.insert(record.name.clone(), tensor);
    }

    let info = CheckpointInfo {
        step: root.usize_opt("step")?.map(|step| step as u64),
        ema_decay: source.ema_decay,
        source_kind: source.kind,
    };
    Ok((
        config,
        ModelWeights::from_validated(tensors, device.clone()),
        info,
    ))
}
