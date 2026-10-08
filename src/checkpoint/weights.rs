use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::Result;
use super::config::{ModelConfig, TensorSpec};
use crate::Error;

const MAX_REPORTED_NAMES: usize = 20;

fn describe_names(names: &[String]) -> String {
    let shown: Vec<&str> = names
        .iter()
        .take(MAX_REPORTED_NAMES)
        .map(String::as_str)
        .collect();
    if names.len() > shown.len() {
        format!("{shown:?} and {} more", names.len() - shown.len())
    } else {
        format!("{shown:?}")
    }
}

pub(crate) struct Found<'a> {
    pub(crate) name: &'a str,
    pub(crate) shape: &'a [usize],
    pub(crate) dtype: DType,
}

pub(crate) fn validate_layout<'a>(
    specs: &[TensorSpec],
    found: impl IntoIterator<Item = Found<'a>>,
) -> Result<()> {
    let found: HashMap<&str, Found<'a>> = found.into_iter().map(|f| (f.name, f)).collect();
    if let Some(spec) = specs.iter().find(|s| !found.contains_key(s.name.as_str())) {
        return Err(Error::MissingTensor(spec.name.clone()));
    }
    let expected: std::collections::HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let mut unexpected: Vec<String> = found
        .keys()
        .filter(|name| !expected.contains(*name))
        .map(|name| name.to_string())
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(Error::Checkpoint(format!(
            "{} unexpected tensors: {}",
            unexpected.len(),
            describe_names(&unexpected)
        )));
    }
    for spec in specs {
        let entry = &found[spec.name.as_str()];
        if entry.shape != spec.shape.as_slice() {
            return Err(Error::ShapeMismatch {
                name: spec.name.clone(),
                actual: entry.shape.to_vec(),
                expected: spec.shape.clone(),
            });
        }
        if entry.dtype != DType::F32 {
            return Err(Error::Checkpoint(format!(
                "tensor `{}` has dtype {:?}, expected F32",
                spec.name, entry.dtype
            )));
        }
    }
    Ok(())
}

fn ensure_finite(tensor: &Tensor, name: &str) -> Result<()> {
    let residual = tensor.sub(tensor)?.sum_all()?.to_scalar::<f32>()?;
    if residual.is_finite() {
        Ok(())
    } else {
        Err(Error::Checkpoint(format!(
            "tensor `{name}` contains non-finite values"
        )))
    }
}

/// Validated model parameters as `f32` candle tensors, keyed by `state_dict` name.
///
/// Every instance holds exactly the tensors listed by [`ModelConfig::tensor_specs`], with the
/// expected shapes, `f32` dtype and only finite values.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl ModelWeights {
    /// Validates `tensors` against `config` and wraps them.
    ///
    /// Errors: [`Error::MissingTensor`] for an absent tensor, [`Error::Checkpoint`] for
    /// unexpected tensors (all listed), a dtype other than `f32`, non-finite values or tensors on
    /// different devices, and [`Error::ShapeMismatch`] for a wrong shape.
    pub fn from_tensors(tensors: HashMap<String, Tensor>, config: &ModelConfig) -> Result<Self> {
        let specs = config.tensor_specs()?;
        validate_layout(
            &specs,
            tensors.iter().map(|(name, tensor)| Found {
                name,
                shape: tensor.dims(),
                dtype: tensor.dtype(),
            }),
        )?;
        let device = tensors
            .values()
            .next()
            .map(|t| t.device().clone())
            .ok_or_else(|| Error::Checkpoint("checkpoint holds no tensors".to_string()))?;
        for spec in &specs {
            let tensor = &tensors[&spec.name];
            if !tensor.device().same_device(&device) {
                return Err(Error::Checkpoint(format!(
                    "tensor `{}` is on a different device than the other tensors",
                    spec.name
                )));
            }
            ensure_finite(tensor, &spec.name)?;
        }
        Ok(Self { tensors, device })
    }

    pub(crate) fn from_validated(tensors: HashMap<String, Tensor>, device: Device) -> Self {
        Self { tensors, device }
    }

    /// Device the tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are held.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total number of scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(Tensor::elem_count).sum()
    }

    /// Tensor by `state_dict` name such as `blocks.3.attn.qkv.weight`.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// All tensors keyed by `state_dict` name.
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Consumes the weights and returns the tensor map.
    pub fn into_tensors(self) -> HashMap<String, Tensor> {
        self.tensors
    }

    /// A [`VarBuilder`] over the tensors with `f32` dtype on the weights' device.
    ///
    /// Names follow the Python `state_dict`, for example
    /// `vb.pp("blocks").pp("0").pp("attn").pp("qkv").get((3 * d, d), "weight")`.
    pub fn var_builder(&self) -> VarBuilder<'static> {
        VarBuilder::from_tensors(self.tensors.clone(), DType::F32, &self.device)
    }
}
