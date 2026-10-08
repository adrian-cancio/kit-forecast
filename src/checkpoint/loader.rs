use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use candle_core::Device;
use serde::{Deserialize, Serialize};

use super::Result;
use super::kit_config::KitConfig;
use super::pt::read_pt;
use super::weights::ModelWeights;
use crate::Error;

/// File name of the weights inside a directory written by [`Checkpoint::save_dir`].
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// File name of the [`KitConfig`] JSON inside a checkpoint directory.
pub const CONFIG_FILE: &str = "config.json";

/// File name of the optional [`CheckpointInfo`] JSON inside a checkpoint directory.
pub const INFO_FILE: &str = "info.json";

/// Which state of the training checkpoint the weights come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// Exponential moving average of the parameters (`ema.shadow`), the preferred source.
    Ema,
    /// Raw model parameters (`model`), used when the checkpoint has no EMA state.
    Model,
}

impl SourceKind {
    /// `"ema"` or `"model"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Model => "model",
        }
    }
}

/// Provenance of the loaded weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// Training step stored in the checkpoint, if present.
    pub step: Option<u64>,
    /// EMA decay of the stored weights, if the weights are EMA weights and the decay is stored.
    pub ema_decay: Option<f64>,
    /// Which state the weights were taken from.
    pub source_kind: SourceKind,
}

/// A loaded KiT checkpoint: inference configuration, validated weights and provenance.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Architecture, windows, normalization constants and sampling defaults.
    pub config: KitConfig,
    /// Validated `f32` parameters on the target device.
    pub weights: ModelWeights,
    /// Where the weights came from.
    pub info: CheckpointInfo,
}

impl Checkpoint {
    /// Loads an official PyTorch `.pt` training or slim checkpoint directly.
    ///
    /// The file is a zip archive holding a pickle (`data.pkl`) and one raw storage per tensor.
    /// The pickle is parsed symbolically and never executed; only plain tensors, the
    /// `float`/`half`/`double`/`bfloat16`/`byte`/`long` storage classes and basic containers are
    /// accepted. Weights are taken from `ema.shadow` (or a flat `ema` dictionary) and otherwise
    /// from `model`. Tensors are read one at a time straight onto `device`, so peak host memory
    /// stays at the size of the largest tensor.
    ///
    /// Fails with [`Error::Checkpoint`] if the file is not a valid checkpoint, `config.model` has
    /// no `yday_dim`, neither `ema` nor `model` exists, or a tensor is non-contiguous, not `f32`,
    /// non-finite or unexpected, with [`Error::MissingTensor`] if a required tensor is absent and
    /// with [`Error::ShapeMismatch`] if a shape disagrees with the configuration.
    pub fn load_pt(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        Self::load_pt_reader(BufReader::with_capacity(1 << 20, file), device)
    }

    /// Like [`Checkpoint::load_pt`], reading the archive from any seekable source.
    pub fn load_pt_reader<R: Read + Seek>(reader: R, device: &Device) -> Result<Self> {
        let (config, weights, info) = read_pt(reader, device)?;
        Ok(Self {
            config,
            weights,
            info,
        })
    }

    /// Writes `model.safetensors`, `config.json` and `info.json` into `dir`, creating it.
    pub fn save_dir(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        candle_core::safetensors::save(self.weights.tensors(), dir.join(WEIGHTS_FILE))?;
        std::fs::write(dir.join(CONFIG_FILE), self.config.to_json()?)?;
        std::fs::write(
            dir.join(INFO_FILE),
            serde_json::to_string_pretty(&self.info)?,
        )?;
        Ok(())
    }

    /// Loads a directory written by [`Checkpoint::save_dir`], with the same strict validation as
    /// [`Checkpoint::load_pt`].
    ///
    /// `info.json` is optional; without it the info reports no step, no decay and
    /// [`SourceKind::Model`]. The safetensors file is read into memory in one piece.
    pub fn load_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join(CONFIG_FILE);
        let weights_path = dir.join(WEIGHTS_FILE);
        for path in [&config_path, &weights_path] {
            if !path.is_file() {
                return Err(Error::Checkpoint(format!(
                    "file not found: {}",
                    path.display()
                )));
            }
        }
        let config = KitConfig::from_json(&std::fs::read_to_string(&config_path)?)?;
        let tensors = candle_core::safetensors::load(&weights_path, device)?;
        let weights = ModelWeights::from_tensors(tensors, &config.model)?;
        let info_path = dir.join(INFO_FILE);
        let info = if info_path.is_file() {
            serde_json::from_str(&std::fs::read_to_string(&info_path)?)?
        } else {
            CheckpointInfo {
                step: None,
                ema_decay: None,
                source_kind: SourceKind::Model,
            }
        };
        Ok(Self {
            config,
            weights,
            info,
        })
    }
}
